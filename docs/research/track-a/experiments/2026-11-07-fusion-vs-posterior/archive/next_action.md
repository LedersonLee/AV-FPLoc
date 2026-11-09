> 역사적 연구 기록. 원본: `fusion_vs_posterior_20261107_1300/next_action.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Research cycle complete

All launched research jobs are finished; existing user servers preserved. Full goal completion audit: completion_audit.md. Final report: RESEARCH_REPORT.md. Primary frozen decoder confirmation on11scenes/5775unique observations/2seeds gives joint68.078→70.078%, +2pp CI[1.258,2.839]pp; R.1 +.485pp; conditional visual success +.647pp. All11 scene-mean joint gains positive. Whole-pipeline untouched generalization is NOT claimed: upstream visual training monitored allVal19. Test stays sealed.

New knowledge: negative gate×decoder interaction repeats; changing training target improves development support but does not establish superiority over fixed support in confirmation. Continuous weight oracle84.248 vs separated Top20~95.883% indicates hypothesis selection opportunity. Spatial aggregation rather than yaw or neighbor count alone explains most gain. NLL/ECE worsen while decision success improves; no probability-calibration improvement claim.

No further automatic experiment is queued. Future falsification questions (not pending required work): connected/wall-aware kernels, matched separated-hypothesis learning, label-smoothing controls, truly fresh end-to-end data with a new predeclared protocol. Already evaluated Val11 cannot be reused as untouched confirmation.
