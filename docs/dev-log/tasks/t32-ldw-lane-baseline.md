# 지시서 t32: LDW에 차로 안 평소 위치 기준 넣기 (쓰기, GPU 금지)

## 배경
민옥 제안으로 Codex가 t31에서 저장 결과로 실험했다. 사전 점검에서 "차가 차로 안 어디쯤을 평소에 달리는지"(카메라 설치 치우침)를 구해 LDW 기준으로 쓰면, 정상 영상 LDW가 31→8프레임으로 줄고 미탐은 그대로였다. 옆 거리 X 보정은 효과가 없어서 넣지 않는다. 결정은 `.orch/DECISIONS.md` 마지막 줄. 실험 코드와 결과: `tools/rejudge_lane_baseline.py`, `runs_out/rejudge_t31/report.md`(계산식, 최소 근거 max(30프레임, 10%), 기준 행 0.95H, 부호 규칙이 적혀 있다).

## 할 것 (자동차 모드만, 오토바이는 그대로)
1. 보정 단계(사전 점검)에서 YOLOPv2 차선 두 줄이 잡힌 프레임으로 평소 위치 b(미터)를 구한다. t31과 같은 식, ±0.5 m 상한, 근거가 부족하면 미적용. 장면 전환 구간마다 따로 구한다.
2. LDW 판정에서 내 차 기준 열을 소실점 열 대신 이 기준으로 옮긴다. 옆 거리 X, BSW, FCW는 바꾸지 않는다.
3. meta의 car_calib에 b, 근거 프레임 수, 상한 적용 여부를 남기고 frame_schema.md에 적는다.
4. t31 재판정과 같은 값이 나오는지 확인한다: t29 저장 결과 몇 개 영상에서 네가 구현한 b 계산이 t31 표의 값과 맞는지.

## 확인 (GPU 금지)
- 민옥이 이번에는 GPU를 쓰지 않길 원한다. `CUDA_VISIBLE_DEVICES`를 비우고 확인한다.
- 30개 재실행은 하지 않는다. CPU로 짧게만 돌려 동작을 확인한다: Nexar 영상 2개(차선이 잘 보이는 낮 영상 1개, 평소 위치가 0.5 m 상한에 걸린 영상 1개)를 앞 15초만 잘라 `--vehicle car`로 돌린다. Mask2Former와 Depth Pro가 CPU에서 너무 느리면 `--no-m2f`, `--no-depth-guard`를 써도 된다. 결과는 `runs_out/t32_smoke/`.
- 오토바이 모드 코드 경로가 바뀌지 않았는지는 코드 diff로 확인해 보고한다(실행 안 해도 됨).

## 범위
- engine.py, geom_car.py, car_models.py, `.orch/contracts/frame_schema.md`만 수정한다. tools/, app.py, viz.py 금지.
- 끝까지 확인한 뒤 보고한다. 백그라운드로 걸어두고 먼저 끝내지 마라.
