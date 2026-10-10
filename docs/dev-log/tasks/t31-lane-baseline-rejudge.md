# 지시서 t31: 차로 안 평소 위치 기준 실험, 저장 결과로 다시 판정 (쓰기, GPU 금지)

## 배경
민옥 제안: 사전 점검에서 "차가 차로 안 어디쯤을 평소에 달리는지"를 영상 단위로 구해 두고, LDW를 그 기준으로 판정하자. 블랙박스는 차 중앙이 아니라 룸미러 옆에 달리는 경우가 많아서, 카메라 기준으로 재면 정상 주행도 한쪽으로 치우쳐 보인다. 같은 치우침이 옆 거리 X에도 들어가서 차선 침범, 측면 나란히 주의가 과하게 뜨는 원인일 수 있다(t30 평가: 정상 영상 측면 나란히 주의가 오른쪽 341, 왼쪽 82프레임. 우측통행이라 주차 차량 탓일 수도 있음).

이번에는 **GPU를 쓰지 않는다.** 모델을 다시 돌리지 말고, t29 저장 결과(`runs_out/nexar_t29/<영상>/frames.jsonl`, `meta.json`)에 있는 값으로 다시 판정해서 효과를 잰다. frames.jsonl에는 프레임별 lanes(left/right 선분), vp, objects(bbox, lateral_m, in_lane, dist_geo 등), warnings가 있고, meta의 car_calib에 수평선, 소실점 x, 카메라 높이, 초점거리가 있다.

## 할 것
1. `tools/rejudge_lane_baseline.py`를 만든다.
   - 영상마다 차선 두 줄이 잡힌 프레임에서, 화면 아래쪽 기준 행에서 차로 중심과 소실점 열의 차이를 미터로 바꿔 "평소 위치"를 구한다(중앙값). 근거 프레임이 너무 적으면(네가 기준을 정해라) 기준 없음으로 둔다.
   - 보정 폭은 ±0.5 m로 제한한다. 카메라가 차 중심에서 벗어날 수 있는 범위가 그 정도다.
   - (A) LDW만 기준 적용, (B) LDW와 X(옆 거리) 모두 기준 적용, 두 경우로 LDW와 BSW(차선 침범, 측면 나란히 등) 등급을 다시 계산한다. FCW와 다른 경고는 저장된 값을 그대로 쓴다. 판정 기준값은 engine.py와 geom_car.py에 있는 값을 그대로 가져와라(복사하지 말고 가능하면 import). 엔진의 판정 순서를 다 재현하기 어려우면 어디가 근사인지 적어라.
   - 결과는 `runs_out/rejudge_t31/A/<영상>/frames.jsonl`, `.../B/...`로 저장해서 `tools/eval_nexar.py`로 평가할 수 있게 한다(meta.json도 복사).
2. `tools/eval_nexar.py`로 t29, A, B를 비교한다. 이상치 3개(pos_00905, neg_01960, pos_00754)는 본 지표에서 빼고 따로 적는다(`.orch/DECISIONS.md` 2026-10-09 끝에서 두 번째 줄). 지표: 미탐, 사고 전 3초 창 선행, 정상 주의(L1) 비율과 사유별, 사고 밖 위험 이상.
3. 영상별 평소 위치 값(미터, 근거 프레임 수)을 표로 낸다. 0.5 m 상한에 걸린 영상이 있으면 표시한다.
4. 결과와 요약을 `runs_out/rejudge_t31/report.md`에 쓴다.

## 범위
- GPU 금지(`CUDA_VISIBLE_DEVICES`를 비워서 실행). 모델 로드도 하지 마라.
- 수정 가능: `tools/rejudge_lane_baseline.py`(새 파일), `tools/eval_nexar.py`, `runs_out/rejudge_t31/`. engine.py, geom_car.py, car_models.py, viz.py, app.py는 읽기만.
