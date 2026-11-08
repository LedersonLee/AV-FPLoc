# 평가 원칙과 비교 시 주의점

이 문서는13개 연구의 공통 해석 규칙이다. 실제 설정의 권위 있는 기록은 각 연구의 protocol/result/source hash다. 아래 설명으로 서로 다른 실험 조건을 통일해 덮어쓰지 않는다.

## 단위와 지표

- R@1m: XY translation error ≤1m.
- Joint: XY ≤1m AND circular yaw error ≤30°.
- Fine/R@0.1m: XY ≤0.1m. Yaw 조건을 추가하지 않는다.
- Top-K coverage/oracle: GT가 성공 후보의 존재를 사후 판정한 값. 실제 Top1 성공과 다르다.
- Repair/harm: 동일 관측에서 기준의 실패를 복구/성공을 훼손한 수 또는 비율. 기준·분모를 함께 읽는다.
- Utility score의 softmax mass/entropy는 별도 calibration 없이 성공 확률로 해석하지 않는다. T=.05 evidence와 T=1 진단을 혼합하지 않는다.
- Query-micro와 scene-macro를 구분한다. 초기 v2의 query 평균 차이와 HSSD의 scene-macro 차이를 한 표에서 동등한 성능으로 순위 매기지 않는다.

## Split와 노출

초기 연구는 Val3, 이후 Train12(6,387query)/Val8(4,190)/Val11(5,775)을 주로 사용했다. Val11이 처음 decoder 확인에 쓰였을 때도 visual teacher의 Val19 노출이 남았다. 이후 연구에서 Val8/Val11은 반복 사용한 development다. Acoustic historical checkpoint는 Val3 선택 이력이 있다.

HSSD 연구의 새 head-held11은 이전 Val11과 **다른 집합**이다. Dense Train35를 기존 학습12+추가 학습12+head-held11로 나누고 기존 Val19를 development로 썼다. Pilot은11×128=1,408query, 후속은 같은11scene의6,072query다. Pilot을 포함하며 독립 replication이 아니다. 후속 가설 설계 이후에는 평가11을 탐색적으로 재사용했다. Visual teacher Train116/Val19 노출은 남는다.

Test 관측/성능을 탐색에 사용하지 않았다. HSSD의 Test ID/family 제외를 위한 행정 메타데이터 확인은 Test 관측 접근과 구분한다. 같은 원본 furnishing family의 split 중복을 보수적으로 제외했다.

## 통계

대부분 시드42/43과 scene 단위 paired bootstrap10,000회를 사용한다. 같은 query의 시드 또는 pair 수를 독립 scene으로 세지 않는다. 연구별 micro/macro bootstrap estimator와 표본 수는 원본 결과를 따른다. 구간은 다중 탐색/선택 편향을 제거한 확증 구간이 아니며 작은 scene 수의 일반화를 보장하지 않는다.

## Matched controls와 데이터 계약

후보 bank, query ID, GT, checkpoint, score normalization, threshold, utility와 학습량을 확인한다. Candidate extraction 변경과 fixed-candidate discrimination 변경을 구분한다. 같은 parameter 수라도 유효 경로·연산량·입력 정보량이 동일한 것은 아니다.

GEOQ 입력은 기존 S26+candidate geometry26+observation geometry26 및 candidate0 상대값으로156차원이다. S와 B2 proposal에는 acoustic 정보가 남는다. 새 branch의 no-audio/donor를 전체 시스템의 음향 제거로 해석하지 않는다. Donor query를 recipient reference와 다시 비교하는지 확인한다.

HSSD 후속 대조는 fullB2 old12 in-sample → old12 B2 OOF → scene-balanced weighting을 분리했다. 초기 pilot128/scene은 scene 수별 query가 같지만 full-query에서는 장면별 query 수가 다르다. Original GEOQ threshold42=0/43=.01과 새 primary threshold0를 구분한다. OOF 복원은 teacher-disjoint 또는 완전히 nested된 전체 pipeline CV가 아니다.

## 기록의 권위와 변경

현재 판정은 [current_findings.md](current_findings.md)에서 후속 반증과 함께 확인한다. 당시 보고서의 수치는 해당 cohort의 역사적 결과로 보존한다. Raw output은 외부 manifest가 추적하며 Git 사본은 경로 치환/HTML 변환 때문에 원본과 해시가 다를 수 있다. 두 해시는 study별 `artifacts.json`에 구분해 기록한다.
