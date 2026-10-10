# t49: t48 리뷰 (read)

t48(Claude)이 t44를 고쳤다: lane_trust에 exclude_lanes(폭이 2.0~4.0 m 밖일 때만 true), 영점 실패, 검산 불가는 LDW만 끔, 근접 보험 축척에서 차선 뺀 구간은 class 3 제외. 결과 runs_out/nexar_t48/(compare.md, verify.json), t48_moto_regress/, t48_work/. 수정 후 해시 engine 67d108a238dc / geom_car ae9bd055612f. 기준 t40: runs_out/nexar_t40, t44: runs_out/nexar_t44.

## 확인할 것
1. 영점 실패, 검산 불가 구간에서 보정(수평선, 소실점, H, 교차 확인, 보닛 상한, 근접 보험)이 t40과 같은 경로를 타는가. pos_00845의 LDW 외 판정이 t40과 전 프레임 같은가.
2. exclude_lanes 구간에서 차선이 남아 있는 경로가 있는가(t47 지적한 근접 보험 포함 전부).
3. 신뢰 12클립, --no-lane-trust, 오토바이 mixkit이 t40과 바이트 동일한가.
4. 숫자 재검산: 미탐 3/20, 사고 밖 bin≥2 21.16%, 근접 보험 수정으로 바뀐 프레임(기록 509, 경보 상태 12, FCW 레벨 1: pos_00936 t=19.2 L1→L2).
5. 현재 엔진 파일 해시가 67d108a238dc / ae9bd055612f인지(테스트셋 실행 전 고정 확인).
6. runs_in/testset 접근 흔적 없음. 단위 테스트 `python -m unittest tools.test_lane_trust tools.test_ldw_variants tools.test_eval_nexar web.test_server`(폴더 생성 권한 오류는 따로 표시).

## 금지
파일 수정 금지. GPU, 엔진 재실행 금지. testset 사용 금지.

## 보고
심각도 순. 테스트셋을 돌려도 되는 상태인지 한 줄로 결론.
