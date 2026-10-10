# t47: t44(차선 신뢰 2.0~4.0 m) 리뷰 (read)

Claude가 t44에서 geom_car.py(LANE_TRUST_W_RANGE_M, lane_trust(), calib_geom()), engine.py(_apply_lane_trust, 기본 켬, --no-lane-trust)를 고쳤다. 결과 runs_out/nexar_t44/(compare.md, verify.json, check/), runs_out/t44_moto_regress/, runs_out/t44_work/. 기준 t40 결과는 runs_out/nexar_t40.

## 확인할 것
1. 기본 동작: 신뢰 클립(12클립 13구간)은 프레임 출력이 t40과 같은가. `--no-lane-trust`가 t40과 완전히 같은가(코드 경로로도).
2. 순환 문제 처리: 검산 폭(차선 없는 수평선 + 독립 H) 계산이 맞는가. 독립 H가 없을 때 차선 H를 쓰는 부분이 불신 클립의 거리에 차선을 다시 끌어들이지 않는가.
3. 불신 클립에서 차선이 수평선, 소실점, H, 보닛 상한, 교차 확인 신뢰도에 정말 안 들어가는가. 빠뜨린 경로가 있는가.
4. LDW 외 변화가 불신 클립에만 있는가. 신뢰 클립에서 FCW, BSW, HEAD가 하나라도 바뀌었으면 원인.
5. compare.md 숫자 재검산: 미탐 3/20, 사고 밖 bin≥2 21.12→21.16%, pos_00845 사고 3초 창 위험 프레임 16→5, pos_00411 FCW 98프레임 상승.
6. pos_00845: 영점 실패(근거 부족, 38/40)로 차선이 교차 확인에서 빠져 수평선 신뢰도 high→low, 보닛 상한이 꺼진 경로를 코드로 확인. "영점 실패"와 "폭 범위 밖"을 다르게 처리하면(실패는 LDW만 끄고 보정에는 차선 유지) pos_00845가 t40과 같아지는지 코드상으로 추정.
7. runs_in/testset 접근 흔적 없음. 단위 테스트 `python -m unittest tools.test_lane_trust tools.test_ldw_variants tools.test_eval_nexar web.test_server`(읽기 전용이라 폴더 생성 오류는 따로 표시).

## 금지
파일 수정 금지. GPU, 엔진 재실행 금지. testset 사용 금지.

## 보고
문제를 심각도 순으로. 숫자는 맞음/다름(값). 6번은 예상 결과와 근거.
