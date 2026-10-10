# 지시서 t46: 개발 세트와 시험 세트 평가 (쓰기, 평가 파일만, GPU 금지), t38 대체

## 배경
- 개발 세트: `runs_in/nexar/` 30개, 최종 엔진 결과 `runs_out/nexar_t48/`. 이전 비교 `runs_out/nexar_t34/`(t32 엔진), `runs_out/nexar_t40/`.
- 시험 세트: `runs_in/testset/` 99개(labels.csv), 결과 `runs_out/testset_t45/`(같은 엔진, 코드 변경 없음. run_log.md 맨 위 해시가 개발 세트 실행 엔진과 같은지 확인).
- 이상치 기준: 자동차 보정 실패(meta car_calib horizon_conf가 failed)는 본 지표에서 빼고 따로 집계(`.orch/DECISIONS.md` 2026-10-09).
- 포트폴리오 README에 쓸 숫자다. 정직해야 한다.

## 할 것
1. `tools/eval_nexar.py`로 두 세트를 각각 평가한다. 시험 세트는 labels.csv 경로를 인자로 받게 고쳐도 된다.
2. 각 지표에 신뢰구간: 미탐률은 Wilson 95%, 비율 지표(사고 밖 위험 이상, 정상 주의)는 영상 단위 부트스트랩 95%(반복 수, 시드 기록). 방법을 eval.md에 적는다.
3. 표 하나: 세트(개발/시험) × 지표(평가 영상 수, 이상치 수, 미탐/사고, 사고 전 3초 창 첫 위험 평균 선행, 사고 밖 위험 이상, 정상 주의, LDW 측정 가능 비율, 차선 신뢰 클립 수) + 낮/밤 분리.
4. 개발 세트와 시험 세트 차이를 해석 없이 숫자로 비교(과적합 판단은 민옥이).
5. 개발 세트 t34 → t40 → t44 → t48 변화(미탐, 사고 밖 위험 이상, 정상 LDW 프레임 좌/우)를 따로 표로.
6. 시험 세트 미탐 영상은 사고 전 3초 원인을 한두 줄씩(t25 형식). 규칙 수정 제안 금지. 시험 세트 결과로 규칙을 고치면 안 된다.
7. 결과: `runs_out/eval_t46/eval.md`, `eval.json`, `readme_table.md`(README에 붙일 표만).

## 범위
수정 가능: tools/eval_nexar.py, `runs_out/eval_t46/`. 엔진과 web/ 금지. GPU 금지. 커밋 금지.
