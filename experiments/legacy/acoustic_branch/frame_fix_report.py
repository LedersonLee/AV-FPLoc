"""Report corrected-frame results and prominently flag invalid legacy evidence."""
import argparse
from pathlib import Path

from .stage_comparison_report import run
from .planar_analysis import bootstrap
from .contracts import atomic_json
from .data import read


def publish(source,report,root):
    run(source,report,root)
    data=read(source/'results.json')['methods']
    corrected=data['corrected_extruded_FP'];native=data['native_stage']
    comparisons=dict(
        corrected_STFT_vs_native_STFT=bootstrap(corrected['STFT_L1']['records'],native['STFT_L1']['records'],'xy'),
        corrected_common_vs_corrected_STFT_xy=bootstrap(corrected['common_arrival_L1']['records'],corrected['STFT_L1']['records'],'xy'),
        corrected_common_vs_corrected_STFT_joint=bootstrap(corrected['common_arrival_L1']['records'],corrected['STFT_L1']['records'],'joint'),
        caveat='exploratory VAL, pose-cluster bootstrap in 3 scenes, no multiplicity correction')
    atomic_json(report/'corrected_comparison.json',comparisons)
    page=report/'index.html';text=page.read_text()
    text=text.replace('<h1>Native 3D stage vs 압출 2DFP — 동일 조건 비교</h1>',
        '<h1>압출 FP 좌표 프레임 수정 후 재평가</h1><p style="background:#ffe0d2;padding:16px">중요: extruded_FP는 잘못 회전되어 로딩된 기존 데이터이며 정상 성능 기준이 아닙니다. corrected_extruded_FP는 원래 GLB와 map을 바꾸지 않고 stage up=+Y, front=−Z를 명시해 다시 렌더링한 결과입니다. 학습 없이 같은 Val 관측·후보를 비교했습니다.</p>')
    text=text.replace('후보 geometry만 바꿨습니다.', '후보 geometry와 로딩 프레임을 대조했습니다. 수정 FP는 모든 렌더링 task에서 로딩 후 bounding box와 source 위치를 검증했습니다.')
    text=text.replace('<h2>전체 비교</h2>', '<h2>전체 비교</h2><p><a href="corrected_comparison.json">수정 FP와 native의 paired 비교 및 매칭 방식 효과</a></p>',1)
    page.write_text(text)


def flag_prior_pages(report_root):
    notice='<div style="background:#ffdbce;color:#521500;border:2px solid #c64b28;padding:18px;margin:20px 0"><strong>중요 정정: 압출 FP RIR 로딩 좌표계 오류 확인.</strong> 이 페이지의 기존 extruded FP 관련 수치와 해당 FP로 학습한 모델은 정상적인 도면 기반 성능으로 해석하면 안 됩니다. 기존 GLB가 (X,Y,Z)→(X,Z,−Y)로 회전되어 로딩됐습니다. Native stage와 순수 수평 경로 대조군은 이 오류에 해당하지 않습니다. 데이터는 감사용으로 보존했습니다. <a href="/frame-fix/">좌표계 수정 후 재평가</a></div>'
    for folder in ('diagnostics','stage-comparison','planar-probe','planar-common'):
        page=report_root/folder/'index.html'
        if not page.exists():continue
        text=page.read_text()
        if '중요 정정: 압출 FP RIR 로딩 좌표계 오류 확인.' in text:continue
        end=text.find('</h1>')
        if end>=0:text=text[:end+5]+notice+text[end+5:]
        temp=page.with_name('index.frame-warning.tmp');temp.write_text(text);temp.replace(page)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--flag-root',type=Path)
    a=p.parse_args()
    if a.flag_root:flag_prior_pages(a.flag_root)
    if a.source:publish(a.source,a.report,a.root)
