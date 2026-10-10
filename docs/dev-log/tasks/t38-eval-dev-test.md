# 지시서 t38: 개발 세트와 시험 세트 평가 (쓰기, 평가 파일만, GPU 금지)

## 배경
- 개발 세트: `runs_in/nexar/` 30개, 최신 결과 `runs_out/nexar_t34/`(t32 엔진). 이전 비교 대상 `runs_out/nexar_t29/`.
- 시험 세트: `runs_in/testset/` 99개(labels.csv), 결과 `runs_out/testset_t37/`(같은 엔진, 코드 변경 없음).
- 이상치 기준: 자동차 보정 실패(meta car_calib horizon_conf가 failed)는 본 지표에서 빼고 따로 집계(`.orch/DECISIONS.md` 2026-10-09).
- 포트폴리오 README에 쓸 숫자를 낸다. 숫자는 정직해야 한다.

## 할 것
1. `tools/eval_nexar.py`로 두 세트를 각각 평가한다. 시험 세트는 labels.csv 경로를 인자로 받게 고쳐도 된다.
2. 각 지표에 신뢰구간을 붙인다: 미탐률은 Wilson 95% 구간, 비율 지표(사고 밖 위험 이상, 정상 주의)는 영상 단위 부트스트랩 95% 구간. 방법을 eval.md에 적는다.
3. 표 하나로 정리한다: 세트(개발/시험) × 지표(평가 영상 수, 이상치 수, 미탐/사고, 사고 전 3초 창 첫 위험 평균 선행, 사고 밖 위험 이상, 정상 주의) + 낮/밤 분리.
4. 개발 세트와 시험 세트 차이를 해석 없이 숫자로 비교한다(과적합 여부를 민옥이 판단할 수 있게).
5. 개발 세트 t29 → t34 LDW 변화(정상 영상 LDW 프레임 좌/우)를 따로 표로 낸다.
6. 시험 세트 미탐 영상은 사고 전 3초 원인을 한두 줄씩 적는다(t25 형식). 규칙 수정 제안은 하지 마라. 시험 세트 결과로 규칙을 고치면 안 된다.
7. 결과: `runs_out/eval_t38/eval.md`, `eval.json`, `readme_table.md`(README에 붙일 표만).

## 범위
- 수정 가능: tools/eval_nexar.py, `runs_out/eval_t38/`. 엔진과 web/ 금지. GPU 금지.
