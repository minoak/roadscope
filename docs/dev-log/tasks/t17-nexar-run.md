# 지시서 t17: Nexar 자동차 영상 10개 실행 (쓰기, 작게)

`runs_in/nexar/` 에 Nexar 충돌 예측 데이터셋 영상 10개와 `labels.csv`(label, time_of_event, time_of_alert, light 등)가 있다. 사고 7개(밤 3, 낮 4), 정상 3개(밤 1, 낮 2). 1280×720, 30fps, 약 40초.

## 할 것
1. car 모드 보정 두 가지만:
   - 자동차용 카메라 높이 prior 1.3m (car 모드일 때만).
   - 2m 억제를 car 모드에서 끈 건 관리자가 승인했다. 그대로.
2. 10개 전부 `python engine.py <영상> --fps 5 --vehicle car` 로 실행. 결과는 `runs_out/nexar/<파일명>/`. 보닛 비율은 기본 0.12 로 하되, 대표 프레임을 보고 명백히 다르면 영상별로 조정하고 그 값을 기록.
3. 영상별 실행 시간과 오류를 `runs_out/nexar/run_log.md` 에 남긴다.

평가 지표 계산은 다른 실행자(Codex)가 `tools/eval_nexar.py` 로 만들고 있다. 너는 실행과 결과 파일 생성까지만.

## 범위
engine.py 의 car 모드 카메라 높이만 수정. 판정 기준, 스키마, viz.py, app.py 금지.
