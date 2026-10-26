"""Publish a standalone seven-input report; no inference or training changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import statistics
import urllib.request

MODES = ('energy', 'coherence', 'envelope', 'both', 'energy_envelope', 'coherence_envelope', 'all')
NEW = {'coherence', 'energy_envelope', 'coherence_envelope'}
LABELS = dict(zip(MODES, ('E', 'C', 'N', 'E+C', 'E+N', 'C+N', 'E+C+N')))
METHODS = ('vision_only', 'acoustic_only', 'equal_weight_fusion', 'visual_topk_acoustic_rerank')
RECALLS = ('recall_0.1m', 'recall_0.5m', 'recall_1m', 'recall_1m_30deg',
           'yaw_within_10deg', 'yaw_within_30deg', 'yaw_within_50deg')
METRICS = RECALLS + ('median_xy_m', 'median_yaw_deg', 'repair', 'harm')
HEADERS = ('R@0.1m', 'R@0.5m', 'R@1m', 'R@1m·30°', 'Yaw≤10°', 'Yaw≤30°', 'Yaw≤50°',
           'Median XY (m)', 'Median yaw (°)', 'Repair', 'Harm')
ASSETS = 'https://ledersonlee.github.io/html-reports/echoloc/hssd-unloc-subset-failures/'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def once(page, old, new):
    if page.count(old) != 1:
        raise ValueError('Unexpected source template: '+old[:90])
    return page.replace(old, new, 1)


def recompute(records, method):
    rows = [r['methods'][method] for r in records]
    xy = [r['xy_error_m'] for r in rows]
    yaw = [r['yaw_error_deg'] for r in rows]
    n = len(rows)
    result = {f'recall_{t:g}m': sum(x <= t for x in xy)/n for t in (.1, .5, 1)}
    result['recall_1m_30deg'] = sum(x <= 1 and y <= 30 for x, y in zip(xy, yaw))/n
    result.update({f'yaw_within_{t}deg': sum(y <= t for y in yaw)/n for t in (10, 30, 50)})
    result.update(median_xy_m=statistics.median(xy), median_yaw_deg=statistics.median(yaw),
                  repair=sum(r['vs_vision_1m'] == 'repair' for r in rows),
                  harm=sum(r['vs_vision_1m'] == 'harm' for r in rows))
    return result


def table(rows):
    header = '<tr><th>Input</th><th>Method</th>'+''.join('<th>'+x+'</th>' for x in HEADERS)+'</tr>'
    body = []
    for label, method, values in rows:
        cells = [label, method]
        cells += [f'{100*values[k]:.2f}' if k in RECALLS else
                  str(values[k]) if k in ('repair', 'harm') else f'{values[k]:.2f}' for k in METRICS]
        body.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
    return '<div class="scroll"><table>'+header+''.join(body)+'</table></div>'


def detail_page(page):
    helper = '''const ASSET_ROOT=ASSET_LITERAL;
async function reportBytes(url){const r=await fetch(url);if(!r.ok)throw Error('Data '+r.status+' '+url);return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer())}
async function reportJson(name){if(name==='summary.json'){const r=await fetch(name);if(!r.ok)throw Error(r.status);return r.json()}return JSON.parse(new TextDecoder().decode(await reportBytes(name+'.gz')))}
'''.replace('ASSET_LITERAL', json.dumps(ASSETS))
    page = once(page, '<script>', '<script>\n'+helper)
    page = once(page, "$('rgb').src='rgb/'+r.scene_id+'_'+r.sample_index+'.png';",
                "$('rgb').src=ASSET_ROOT+'rgb/'+r.scene_id+'_'+r.sample_index+'.webp';")
    page = once(page, "image=new Image();image.src='maps/'+r.scene_id+'.png';",
                "image=new Image();image.crossOrigin='anonymous';image.src=ASSET_ROOT+'maps/'+r.scene_id+'.png';")
    page = once(page, "let response=await fetch(r.heatmap);if(!response.ok)throw Error('점수 파일 로드 실패');let values=new Uint8Array(await response.arrayBuffer());",
                "let values=await reportBytes(r.heatmap+'.gz');")
    page = once(page, "fetch(x).then(r=>{if(!r.ok)throw Error('결과 파일 로드 실패: '+x);return r.json()})", 'reportJson(x)')
    old = "['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','median_xy_m','median_yaw_deg','repair','harm']"
    page = once(page, old, json.dumps(METRICS))
    old = "['R@0.1m','R@0.5m','R@1m','R@1m/30°','Median m','Median yaw°','Repair','Harm']"
    page = once(page, old, json.dumps(HEADERS, ensure_ascii=False))
    page = once(page, "k.startsWith('recall')?", "(k.startsWith('recall')||k.startsWith('yaw_within'))?")
    return once(page, '<h1>', '<p><a href="../../">입력 7조건 비교</a></p><h1>')


def build(source, destination, shared_manifest):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refuse to overwrite an existing report')
    destination.mkdir(parents=True, exist_ok=True)
    analysis = read(source/'analysis.json')
    independent = read(source/'independent_verification.json')
    if independent['status'] != 'verified' or set(analysis['conditions']) != set(MODES):
        raise ValueError('Source report is not independently verified')
    # Compare the previously published asset manifest to the actual public copy.
    with urllib.request.urlopen(ASSETS+'publication.json', timeout=30) as response:
        public_manifest = response.read()
    shared = read(shared_manifest)
    if json.loads(public_manifest) != shared:
        raise ValueError('Public asset provenance differs from local proof')
    rgb = {r['original_name']: r for r in shared['rgb_receipts']}
    files = {}; all_rows = []; reference = None
    for mode in MODES:
        folder = source/'dense'/mode if mode in NEW else source.parent/'input-ablation-dense'/mode
        records = read(folder/'records.json'); summary = read(folder/'summary.json'); banks = read(folder/'banks.json')
        identities = [(r['scene_id'], r['sample_index'], r['gt_pose_xzyaw']) for r in records]
        if len(records) != 1472 or len(banks) != 3 or (reference is not None and identities != reference):
            raise ValueError('Seven conditions must use exactly the same query/pose order')
        reference = identities
        if digest((folder/'records.json').read_bytes()) != independent['proofs'][mode]['records_sha256']:
            raise ValueError('Source records changed after independent verification')
        for method in METHODS:
            computed = recompute(records, method)
            for key, value in computed.items():
                if abs(value-analysis['conditions'][mode]['micro'][method][key]) > 1e-10:
                    raise ValueError('Source/recomputed metric differs: '+str((mode, method, key)))
            summary['micro'][method] = computed
            for scene in banks:
                summary['per_scene'][scene][method].update(recompute([r for r in records if r['scene_id'] == scene], method))
            if mode == MODES[0] or method != 'vision_only':
                all_rows.append((LABELS[mode] if method != 'vision_only' else 'Common vision', method, computed))
        target = destination/'details'/mode; target.mkdir(parents=True)
        save(target/'summary.json', summary)
        for name in ('metrics.csv', 'results.md'):
            shutil.copy2(folder/name, target/name)
        def pack(path):
            blob = path.read_bytes(); compressed = gzip.compress(blob, compresslevel=6, mtime=0)
            if gzip.decompress(compressed) != blob:
                raise ValueError('Lossless compression failed')
            out = target/(str(path.relative_to(folder))+'.gz'); out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(compressed)
            return str(out.relative_to(destination)), dict(source_sha256=digest(blob), published_sha256=digest(compressed))
        paths = [folder/'records.json', folder/'banks.json']+[folder/r['heatmap'] for r in records]
        if len(set(paths)) != 1474:
            raise ValueError('Heatmap reference coverage changed')
        with ThreadPoolExecutor(max_workers=8) as pool:
            files.update(pool.map(pack, paths))
            def verify_rgb(r):
                path = folder/'rgb'/f'{r["scene_id"]}_{r["sample_index"]}.png'
                if digest(path.read_bytes()) != rgb[path.name]['original_sha256']:
                    raise ValueError('Reused RGB differs: '+path.name)
            list(pool.map(verify_rgb, records))
        for scene in banks:
            blob = (folder/'maps'/f'{scene}.png').read_bytes()
            if digest(blob) != shared['files_sha256'][f'maps/{scene}.png']:
                raise ValueError('Reused floorplan differs')
        (target/'index.html').write_text(detail_page((folder/'index.html').read_text()))
        print(dict(mode=mode, queries=len(records), status='verified'), flush=True)
    for name in ('summary.json', 'analysis.json', 'analysis.html', 'val.html', 'val_summary.json',
                 'conditional-yaw.html', 'independent_verification.json'):
        shutil.copy2(source/name, destination/name)
    val_page = destination/'val.html'
    val_page.write_text(val_page.read_text().replace('href="dense/"', 'href="./"'))
    rows = io.StringIO(); writer = csv.writer(rows)
    writer.writerow(['input', 'method']+list(HEADERS))
    for label, method, values in all_rows:
        writer.writerow([label, method]+[100*values[k] if k in RECALLS else values[k] for k in METRICS])
    (destination/'metrics.csv').write_text(rows.getvalue())
    links = ''.join(f'<li><a href="details/{m}/">{LABELS[m]} · RGB / rays / likelihood</a></li>' for m in MODES)
    intro = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HSSD · 입력 7조건 dense test 비교</title><style>body{font:16px system-ui;margin:28px;color:#17314d}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccd6e5;white-space:nowrap;text-align:right}td:first-child,td:nth-child(2){text-align:left}th{background:#edf3fc}.scroll{overflow:auto}.note{background:#fff1cf;padding:18px;line-height:1.7}</style>
<h1>HSSD · 입력 7조건 dense test 비교</h1><div class="note">Train12 / val3 / test3, 동일한 test 1,472 query · 0.2m XY / yaw10° 후보 bank.
E=학습용 STFT energy, C=complex coherency, N=마이크 공통 성분 제거 후 envelope.
각 조건은 val loss로 선택된 best이며, test로 checkpoint·K·가중치를 선택하지 않았습니다.
Single seed42, 조건별 파라미터 수가 다르고 기존 test subset을 재사용한 탐색적 비교입니다.
Recall과 yaw 지표는 전체 query micro 평균(%). Yaw 정확도는 위치 성공 여부와 무관하며 R@1m·30°만 결합 조건입니다.
1:1 fusion은 두 score를 각각 z-score 정규화한 뒤 동일 가중치로 결합합니다. Top-K는 visual top-50 후보의 acoustic 재정렬입니다.
Repair/Harm은 vision의 1m 위치 성공을 기준으로 한 개선/악화 query 개수입니다.</div>
<p><a href="metrics.csv">전체 지표 CSV (%)</a> · <a href="analysis.html">입력별 분석</a> · <a href="val.html">Val best/last</a> · <a href="conditional-yaw.html">조건부 yaw 분석</a> · <a href="publication.json">게시 검증</a></p>'''
    (destination/'index.html').write_text(intro+table(all_rows)+'<h2>조건별 상세</h2><ul>'+links+'</ul></html>')
    manifest = dict(status='verified', queries_per_condition=1472, scenes=3, conditions=7, policies=4,
                    source_analysis_sha256=digest((source/'analysis.json').read_bytes()),
                    source_verification_sha256=digest((source/'independent_verification.json').read_bytes()),
                    source_files_modified=False, raw_RIR_or_checkpoints_included=False,
                    numerical_data_lossless_gzip=True, all_metrics_recomputed=True,
                    reused_asset_root=ASSETS, reused_asset_manifest_sha256=digest(public_manifest),
                    reused_rgb_hash_checks=1472*7, files=files)
    save(destination/'publication.json', manifest)
    print(dict(status='verified', bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--shared-manifest', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.destination, args.shared_manifest)
