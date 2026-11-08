#!/usr/bin/env python3
"""Export existing research records; never execute experiments or alter source studies.
Requires Python3 + beautifulsoup4. Run: python export_local_studies.py /path/to/track1
Generated Markdown is a historical snapshot; review interpretation before committing.
"""
from pathlib import Path
import hashlib,json,re,sys,os
from urllib.parse import unquote
from bs4 import BeautifulSoup,NavigableString,Doctype
ROOT=Path(sys.argv[1]).resolve();OUT=Path(__file__).resolve().parents[1];SRC=ROOT/'experiments'
SPECS=[
('astra_fusion_20261107_0200','2026-11-07-fusion-audit','초기 fusion 감사','results_summary.md','저비용 gate·yaw 개입에서 강한 기준을 넘는 개선을 확인하지 못했다.','analysis.json'),
('astra_fusion_v2_20261107_0225','2026-11-07-support-refinement','국소 지지도와 정밀화','results_summary.md','Val8에서 Support+refinement가 B2보다 joint +1.587%p였다. 학습형 gate 개선이 아닌 후보 결정 규칙의 효과이며 재사용 개발 결과다.','final_summary.json'),
('fusion_vs_posterior_20261107_1300','2026-11-07-fusion-vs-posterior','Fusion과 posterior 결정 분리','RESEARCH_REPORT.md','고정 support/refinement 효과와 adaptive weighting을 분리했다. Val11은 당시 decoder 조건부 확인이었지만 visual teacher 미노출 평가는 아니다.','confirmation_summary.json'),
('hypothesis_discrimination_20261107_1538','2026-11-07-hypothesis-discrimination','후보 추출과 utility ranking','RESEARCH_REPORT.md','같은 Top20 후보에서 utility 차이 회귀가 선택을 개선했다. Val8 +0.895pp, 재사용 Val11 +0.597pp이며 coverage 증가와 구분한다.','summary.json'),
('modality_coverage_20261107_1700','2026-11-07-modality-coverage','Modality별 coverage와 판별','RESEARCH_REPORT.md','관측 기반 음향 coverage는 정보 없는 NMS보다 높지만 Top-K 포함과 실제 Top1 판별 사이 격차가 크다.','summary.json'),
('structural_evidence_20261107_1730','2026-11-07-geometry-discrimination','후보 geometry와 GEOQ','RESEARCH_REPORT.md','기존 점수와 후보 형상의 결합이 utility ranking을 개선했다. 새 관측의 physical matching 이득이라는 초기 해석은 대조 실험으로 수정했다.','final_evaluation.json'),
('acoustic_cues_20261107_1830','2026-11-07-acoustic-cues','음향 단서와 물리적 해석','RESEARCH_REPORT.md','시간·envelope 및 초기 복소 채널 관계의 판별 정보를 확인했다. 고정 직접 경로·정규화 때문에 절대 음량/직접 도달 시간 해석은 지지되지 않았다.','metric_comparisons.json'),
('pairwise_acoustic_evidence_20261107_2100','2026-11-07-pairwise-acoustic','반복 음향과 후보 쌍','RESEARCH_REPORT.md','모호한 쌍의 acoustic 정보는 있으나 GEOQ 이후 추가 joint 개선은 미확인이다. C3−GEOQ −0.104pp, CI[−0.334,+0.114].','paired_statistics.json'),
('alignment_discrimination_20261107_2221','2026-11-07-alignment-discrimination','Domain alignment와 discrimination','RESEARCH_REPORT.html','양성 거리 감소만으로 판별 개선을 설명할 수 없다. Positive-only alignment는 표현을 수축시켰고 matched downstream은 GEOQ를 넘지 못했다.','domain_summary.json'),
('candidate_sampling_20261107_2337','2026-11-07-candidate-sampling','경쟁 후보와 negative sampling','RESEARCH_REPORT.html','학습 negative와 실제 경쟁 후보의 차이를 확인했지만 candidate-aware sampling이 강한 GEOQ를 개선하지 못했다. Multi-positive의 상대 회복을 최종 개선과 구분한다.','downstream_seed_summary.json'),
('directional_disambiguation_20261108_0123','2026-11-08-directional-interaction','방향성 정보 교환','RESEARCH_REPORT.html','A→V는 일부 visual 판별을 돕지만 같은 정보의 concat/scalar를 안정적으로 넘지 못했다. E0+GEOQ 개발셋 신호는 HSSD 연구의 전체query 후속 검증에서 확정되지 않았다.','full_downstream_summary.json'),
('residual_spatial_20261108_1144','2026-11-08-residual-spatial','잔여 정보와 공통 공간 표현','RESEARCH_REPORT.html','잔여 feature·angular/Fourier·capacity·내부 검증 대조를 거쳤으나 GEOQ 이후 안정적인 추가 이득은 미확인이다. 좌표 정합 성공은 실제 DoA 판별 증거와 다르다.','phase_a_full_analysis.json'),
('hssd_scaling_20261108_1332','2026-11-08-hssd-scaling','HSSD 준비도·확대·학습 조건 복원','RESEARCH_REPORT.html','135개 non-Test 장면 감사와478 fits. 후보 geometry의 matched 이득은 유지됐으나 기존 장면별 가중치 복원 후12→24 확대 이득은 불확실했다.','FINAL_FINDINGS.json'),
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sanitize(s):
 return s.replace(str(ROOT),'${WORKSPACE}').replace('/workspace','${WORKSPACE}').replace('/file2/jeongeon/AV-FPLoc/datasets/HSSD','${HSSD_ROOT}').replace('/ssd1/AV-FPLoc/datasets/HSSD','${HSSD_SSD_ROOT}')
selected={};maps={};records={}
for src,slug,title,report,finding,evidence in SPECS:
 base=SRC/src;dest=OUT/'studies'/slug;dest.mkdir(parents=True,exist_ok=True)
 files={base/report:dest/'results.md'}
 for f in base.glob('*.md'):
  if f.name!=report and '.template.' not in f.name:files[f]=dest/'archive'/f.name
 names=[evidence,'final_verification.json','FINAL_VERIFICATION.json','stopping_decision.json','protocol.json','split_protocol.json','SCENE_WEIGHTING_PROTOCOL.json','OOF_ALIGNMENT_PROTOCOL.json']
 if src.startswith('hssd_scaling'):names+=['recipe_control_results.json','full_query_results.json','oof_results.json','EXPERIMENT_PROTOCOL.json','EXPANDED_TRAINING_PROTOCOL.json']
 for n in names:
  f=base/n
  if f.exists():files[f]=dest/'evidence'/n
 selected[src]=files;maps.update({f.resolve():d for f,d in files.items()})
 # Source-root inventory plus original recursive manifest pointer; no heavyweight outputs copied.
 entries=[]
 for f in sorted(base.iterdir()):
  if f.is_file() and f.suffix in ['.md','.html','.json','.py','.sh','.txt']:
   entries.append(dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f),git_copy=str(files[f].relative_to(dest)) if f in files else None))
 records[src]=dict(schema_version=1,study_id=src,source_root='${WORKSPACE}/experiments/'+src,scope='Root records/scripts only. Complete nested artifact inventory remains in the source artifact_manifest.json or MANIFEST.json. External raw data/checkpoints are not included in Git.',source_files=entries)

def resolve_link(url,base,dest):
 if url.startswith(('https://','http://','mailto:','#')):return url
 raw=unquote(url.split('#')[0]);p=Path(raw)
 if raw.startswith('/workspace/'):p=ROOT/raw[len('/workspace/'):]
 elif raw.startswith('/output/'):p=base.parent/raw[len('/output/'):]
 elif not p.is_absolute():p=base.parent/p
 p=p.resolve()
 if p in maps:return os.path.relpath(maps[p],dest.parent)
 # External links deliberately point to the inventory, with original target kept in visible prose.
 return os.path.relpath(dest.parent/'artifacts.json' if dest.name=='results.md' else dest.parent.parent/'artifacts.json',dest.parent)

def rewrite_md(s,base,dest):
 def change(m):
  label,url=m.group(1),m.group(2).strip('<>');new=resolve_link(url,base,dest)
  if new.endswith('artifacts.json'):return '['+label+']('+new+') (외부 산출물: `'+sanitize(url)+'`)'
  return '['+label+']('+new+')'
 return sanitize(re.sub(r'!?\[([^\]]*)\]\(([^\n)]+)\)',change,s))

def html_md(base,dest):
 soup=BeautifulSoup(base.read_text(),'html.parser')
 for n in soup(['style','script','head','title']):n.decompose()
 def conv(n):
  if isinstance(n,Doctype):return ''
  if isinstance(n,NavigableString):return re.sub(r'\s+',' ',str(n))
  name=n.name
  if name=='table':
   rows=[]
   for tr in n.find_all('tr'):
    cells=[re.sub(r'\s+',' ',c.get_text(' ',strip=True)).replace('|','\\|') for c in tr.find_all(['th','td'],recursive=False)]
    if cells:rows.append(cells)
   if not rows:return ''
   width=max(map(len,rows));rows=[r+['']*(width-len(r)) for r in rows]
   return '\n\n'+'\n'.join('| '+' | '.join(r)+' |' for r in [rows[0],['---']*width]+rows[1:])+'\n\n'
  if name=='img':return '\n\n그림: '+n.get('alt','')+' — 외부 산출물 `'+sanitize(n.get('src',''))+'` ([목록](artifacts.json)).\n\n'
  if name=='a':
   label=''.join(conv(c) for c in n.children).strip();url=n.get('href','');target=resolve_link(url,base,dest)
   return '['+label+']('+target+')'+(' (`'+sanitize(url)+'`)' if target.endswith('artifacts.json') else '')
  if name=='pre':return '\n\n```\n'+n.get_text().strip()+'\n```\n\n'
  text=''.join(conv(c) for c in n.children)
  if re.fullmatch('h[1-6]',name or ''):return '\n\n'+'#'*int(name[1])+' '+re.sub(r'\s+',' ',text).strip()+'\n\n'
  if name in ['strong','b']:return '**'+text.strip()+'**'
  if name=='code':return '`'+text.strip()+'`'
  if name=='li':return '\n- '+text.strip()+'\n'
  if name=='br':return '\n'
  if name in ['p','section','header','div','figure','ul','ol']:return '\n\n'+text.strip()+'\n\n'
  return text
 return sanitize(re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',conv(soup)).strip())+'\n'

for i,(src,slug,title,report,finding,evidence) in enumerate(SPECS):
 base=SRC/src;dest=OUT/'studies'/slug;files=selected[src]
 for f,d in files.items():
  d.parent.mkdir(parents=True,exist_ok=True)
  if d.suffix=='.json':
   # Numerical contents retained, machine-local paths parameterized. Original hash retained separately.
   obj=json.loads(f.read_text());selection='All source JSON fields retained; machine paths parameterized.'
   if src.startswith('pairwise_acoustic') and f.name=='paired_statistics.json':
    obj['records']=[r for r in obj['records'] if r['group']=='all'];selection="Only group=='all' records, all arms/splits/metrics; subgroup records remain external and in the report."
   if src.startswith('structural_evidence') and f.name=='final_evaluation.json':
    obj={k:v for k,v in obj.items() if k in ['metrics_order','selected','contrasts']};selection='metrics_order, selected, contrasts only; detailed arms/examples remain external and in the report.'
   if f.name=='oof_results.json':
    obj['summary']=[r for r in obj['summary'] if r['group']=='all'];selection="summary restricted to group=='all'; all contrasts, scene_rows, training, scope retained. Subgroups remain external."
   assert isinstance(obj,dict),f
   obj['_archive_export']=dict(source=f.name,selection=selection)
   d.write_text(sanitize(json.dumps(obj,ensure_ascii=False,indent=2))+'\n')
  else:
   body=html_md(f,d) if f.suffix=='.html' else rewrite_md(f.read_text(),f,d)
   d.write_text('> 역사적 연구 기록. 원본: `'+src+'/'+f.name+'`. 원본 SHA256은 [산출물 목록]('+os.path.relpath(dest/'artifacts.json',d.parent)+')에 있습니다.\n> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론]('+os.path.relpath(OUT/'current_findings.md',d.parent)+')을 먼저 확인하세요.\n\n'+body.rstrip()+'\n')
 rec=records[src]
 for e in rec['source_files']:
  if e['git_copy']:e['git_copy_sha256']=sha(dest/e['git_copy'])
 (dest/'artifacts.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 prev='[이전 연구](../'+SPECS[i-1][1]+'/README.md)' if i else '이 연속 연구의 시작'
 nxt='[다음 연구](../'+SPECS[i+1][1]+'/README.md)' if i+1<len(SPECS) else '최신 연구'
 (dest/'README.md').write_text(f'''# {title}

- 연구 ID: `{src}`
- 판정: {finding}
- 평가 범위: 연구별 query/scene/aggregation을 확인해야 한다. Val8/Val11 재사용 및 visual teacher 노출을 독립 평가로 해석하지 않는다. Test 관측·결과는 연구에 사용하지 않았다.

## 읽기 순서

1. [상세 결과와 해석](results.md)
2. [프로토콜과 변경 이력](protocol.md)
3. [재현과 원본 근거](reproduction.md)
4. [수치 근거](evidence/{evidence}) · [원본 위치·해시](artifacts.json)

## 연구 흐름

{prev} → 이 연구 → {nxt}

[전체 연구 목록](../../experiment_index.md) · [현재 결론과 후속 반증](../../current_findings.md)

이 문서는 기존 결과의 Git 아카이브이며 새 실험이나 독립 재검증을 뜻하지 않는다. 당시 보고서의 해석과 현재 누적 결론을 구분한다.
''')
 protocols=[d for f,d in files.items() if 'protocol' in f.name.lower() or f.name in ['iterations.md','execution_events.md','execution_history.md','hypothesis_register.md','INFORMATION_AUDIT.md']]
 (dest/'protocol.md').write_text('# 프로토콜과 변경 이력\n\n원본 문서/설정에서 실제 학습·평가 조건을 확인한다. 후속 변경은 최초 사전 설정과 구분하며, Git에 옮긴 날짜를 실험 설계 날짜로 해석하지 않는다.\n\n'+ '\n'.join('- ['+d.name+']('+str(d.relative_to(dest))+')' for d in protocols)+'\n\n[상세 결과](results.md)에 matched controls, 실패와 한계가 포함되어 있다. 원본 코드의 해시는 [artifacts.json](artifacts.json)에 기록했다.\n')
 repro=[d for f,d in files.items() if 'reproduc' in f.name.lower()]
 scripts=[e['path'] for e in rec['source_files'] if e['path'].endswith(('.py','.sh'))]
 (dest/'reproduction.md').write_text('# 재현과 원본 근거\n\n원본 연구 폴더: `${WORKSPACE}/experiments/'+src+'`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.\n\n'+('\n'.join('- [원본 재현 문서: '+d.name+']('+str(d.relative_to(dest))+')' for d in repro) if repro else '별도 원본 재현 Markdown은 없다. 아래 실행/분석 스크립트, 프로토콜 및 결과 문서를 함께 사용한다. 이 Git 문서만으로 완전한 실행 환경이 복원된다고 주장하지 않는다.')+'\n\n## 원본 실행·분석 코드\n\n'+ '\n'.join('- `'+s+'`' for s in scripts)+'\n\n전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.\n\n## 검증 경계\n\n[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.\n')
print('Exported',len(SPECS),'studies')
