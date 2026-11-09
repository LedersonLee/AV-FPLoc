# M1 후보 판별기 연결 준비

상태: 특징 계산의 구현/단위 검증 완료. M1 OOF 후보 생성과 판별기 학습은 아직 미실행이며, 성능 결과가 아니다.

## 발견과 수정

기존 `candidate_features`는 자체적으로 `.85V+.15A`를 재구성해 support, 상대 점수, entropy를 계산했다. 이는 study23의 독립 F1 scorer에는 맞지만, M1이 생성한 Top20에 그대로 사용하면 다른 posterior를 특징으로 사용하게 된다. 선택적 keyword `score_field`로 실제 full-bank scorer 출력을 전달하도록 확장했다. 생략 시 기존 F1 계산을 그대로 유지한다. 후보 개수와 일치하지 않는 일부 점수/비유한 점수는 거부한다. 133차원 구조와 label-free signature는 유지했다.

코드 SHA: `b41e6462003a6b37806d368e79e5757f6a34138a`. 검증은 해당 변경을 포함한 작업 상태에서 실행했고 이후 이 SHA로 확정했다. 무관한 Track A 기존 두 파일의 변경은 포함하지 않았다.

## 검증과 한계

서로 3m 떨어진 합성 후보25개에서 support가 해당 candidate의 정확한 softmax 확률과 일치하는 독립 oracle을 사용했다. M1용 임의 점수장을 주면 support/상대 점수/entropy가 그 점수장을 따르고, 기존 균일 F1 결과와 구별됨을 확인했다. 명시적 F1 점수 전달과 기존 호출은 bitwise 동일했다. 부분 점수장/NaN 입력은 예상대로 거부했다. 기존 좌표 변환/GT-free 특징/gradient 검사도 통과했다.

전체 `track_b/tests`: **44 passed, 2 기존 autocast deprecation warnings**, 1.80초. 재현: `avfploc:cu124-clean`, repo를 `/work:ro`로 mount, `PYTHONPATH=/work OMP_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider track_b/tests`.

실제 M1 후보, OOF 학습 분리, 분류기의 효과는 이 테스트로 검증되지 않는다. 다음 연결에서는 M1 full field에서 후보를 추출하고 동일 field를 이 함수에 전달해야 한다. 모든 locally trained precursor/interaction은 fold-Train만 사용해야 하며, 기존 전체 Train checkpoint를 OOF scorer로 재사용할 수 없다. GT는 특징과 후보 확정 이후 label/metric에만 사용한다. 경쟁 설명과 최종 성공 여부는 본 연구 README의 가설 및 실제 후속 평가로 판단한다. 본 단계는 구현 검증이라 dataset 성능 baseline/ablation/일반화 결론은 해당하지 않는다.

문서 검증 정정: README에 후단 준비 링크를 추가한 직후 기존 문서 해시가 달라 archive 검사가 한 차례 실패했다. 원본 실험 파일 변경은 없으며 README의 문서 해시만 갱신한 뒤 재검증했다.
