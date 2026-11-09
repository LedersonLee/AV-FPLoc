# Related-work verification status

2026-11-09에 사용자 프로토콜의4개 primary proceedings URL을 직접 조회했다.

- [UnLoc, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59883a1524bf58cf2dd86d0fc8d499e2-Abstract-Conference.html): 공식 abstract 확인. Depth uncertainty를 분포로 모델링하고 pretrained monocular depth를 사용하는 sequential floorplan localization이다. 이번 single-query tiny-scratch pilot과 동일조건 비교가 아니므로 paper-reported 성능을 결과표에 옮기지 않았다. 전문·동일조건 재현은 미완료.
- F3Loc CVPR2024, SemRayLoc ICCV2025, Ye et al.CVPR2026의 제공 CVF URL은 이번 도구 조회에서403이었다. 논문 내용/재현 상태를 확인했다고 주장하지 않는다. URL은 원문 protocol에 보존했다.

Full SE(2) posterior, ray matching, cross-attention 자체를 신규 기여로 주장하지 않는다. Novelty/accuracy-cost Pareto 검증은 아직 수행되지 않았다.
