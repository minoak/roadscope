# 지시서 t30: t29 결과 평가 (쓰기, 평가 결과 파일만)

## 배경
Claude가 t29에서 자동차 모드에 YOLOPv2(차선)와 내 차 영역 사전 점검(Mask2Former + 반복 검출 + 정지 영역)을 넣고, 보정 안전장치와 침범 기준(1.25 m)을 바꾼 뒤 30개를 `runs_out/nexar_t29/`에 돌렸다. 보고는 `.orch/runs/20261009-133427-claude-t29-car-models-integrate/report.md`. objects에 `interior: true` 박스가 섞여 있다(판정에서 뺀 실내 박스, 거리, track_id null).

## 할 것
1. `tools/eval_nexar.py --runs runs_out/nexar_t29 --no-timelines`로 30개 평가를 `runs_out/nexar_t29/eval.md`, `eval.json`에 낸다. interior 박스는 평가 근거 추출에서 실제 객체와 구분한다.
2. t24(`runs_out/nexar_t24/`)와 30개 전체를 나란히 비교한다: 미탐, 사고 전 3초 창 선행, 사고 밖 위험 이상, 정상 주의, 보정 신뢰도별, 낮/밤.
3. 미탐 사고 영상(Claude 간이 집계로는 pos_00004, pos_00877, 새로 pos_00905)의 사고 전 3초 원인을 t25처럼 한두 줄씩 적는다.
4. 정상 영상 주의(L1)가 어디서 나오는지 경고 사유별로 집계한다. Claude 보고로는 침범이 줄고 "측면 나란히"와 FCW 주의로 옮겨갔다고 한다.
5. 사고 밖 위험 이상 상위 5개 원인. neg_01711은 실내 person이 빠진 뒤에도 위험이 81%라고 한다(보정 low로 주차 차량을 3~4 m로 잡음).

## 범위
- tools/eval_nexar.py만 고쳐도 된다. engine.py, geom_car.py, car_models.py, viz.py, app.py 금지. engine 실행과 GPU 사용 금지.
