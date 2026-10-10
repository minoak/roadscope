# 지시서 t25: Nexar 30개 평가 (쓰기, 평가 결과 파일만)

## 배경
Claude가 t24에서 자동차 모드 버그를 고치고 30개를 `runs_out/nexar_t24/`에 돌렸다. 보고는 `.orch/runs/20261009-105019-claude-t24-fix-miss-bugs-run30/report.md`. `runs_out/nexar_t22/`는 버그가 있는 중간 결과라 쓰지 않는다. 판정 원칙과 결정은 `.orch/DECISIONS.md` 2026-10-09 줄을 보라(한 장면 기준, TTC는 기록만).

## 할 것
1. `tools/eval_nexar.py --runs runs_out/nexar_t24`로 30개 평가를 낸다. 결과는 `runs_out/nexar_t24/eval.md`, `eval.json`.
2. 기존 10개는 세 결과를 나란히 비교한다: `runs_out/nexar/`(이전 엔진), `runs_out/nexar_t20/`(새 기하, TTC 판정 포함), `runs_out/nexar_t24/`(현재).
3. 미탐이 난 사고 영상은 영상마다 사고 3초 전~사고 사이에 정면 객체가 어떻게 잡혔는지(박스 유무, 거리, X, FCW cause, 보정 신뢰도)를 frames.jsonl에서 뽑아 한두 줄로 원인을 적는다. Claude 간이 점검에서 pos_00004(끼어들기), pos_00877(정면 5~7 m)이 주의까지만 떴다고 했다.
4. 오경보가 큰 영상 상위 5개도 같은 방식으로 원인을 한두 줄씩 적는다(정체 중 앞차, 보정 실패, 소실점 이상 등). pos_00754는 소실점 x가 화면 밖으로 잡혔다고 한다.
5. 보정 신뢰도(low/중간/high 등 meta의 car_calib)별로 지표를 나눠 본다.

## 범위
- tools/eval_nexar.py는 필요하면 고쳐도 된다. engine.py, geom_car.py, viz.py, app.py 금지. engine 실행과 GPU 사용 금지.
- 쓰는 파일은 `runs_out/nexar_t24/eval.*`와 tools/eval_nexar.py뿐이다.

## 보고
핵심 숫자(미탐 수/사고 수, 선행 시간, 사고 밖 위험 이상 비율, 정상 영상 주의 비율)를 맨 앞에 쓴다.
