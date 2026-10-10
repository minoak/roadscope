# 지시서 t18: 충돌 경고 평가 도구 (쓰기)

## 목적
처음으로 정답 라벨이 있는 테스트 세트가 생겼다: `runs_in/nexar/labels.csv` (Nexar 충돌 예측 데이터셋 10개. label 1 = 충돌, 아찔, time_of_event = 사고 시각 초, time_of_alert = 사람이 경고를 줬어야 할 시각). 엔진 결과는 다른 실행자(Claude)가 지금 `runs_out/nexar/<파일명>/frames.jsonl` 로 만드는 중이다 (스키마 `.orch/contracts/frame_schema.md`).
민옥의 최우선 원칙: 위험, 치명 미탐을 줄이는 것 (게시판 공지 참고).

## 만들 것
1. `tools/eval_nexar.py`: labels.csv + 각 frames.jsonl → 지표
   - 사고 영상: 사고 시각 이전에 처음 위험(bin≥2) 이 뜬 시각, 처음 주의(bin≥1) 가 뜬 시각, 사고 시각 대비 리드타임(초), time_of_alert 대비 빠름/늦음. 사고 전 3초 안에 위험 이상이 한 번도 없으면 "미탐".
   - 정상 영상과 사고 영상의 사고 3초 이전 구간: 위험 이상 프레임 비율 (오탐 지표).
   - 요약: 미탐 수/7, 평균 리드타임, 오탐 비율. 밤/낮 나눠서.
   - 결과를 `runs_out/nexar/eval.md` (표) 와 `eval.json` 으로.
2. 영상별 타임라인 PNG: 기존 render_timeline 을 재사용해서, 사고 시각(빨간 세로선)과 경고 기준 시각(주황 세로선)을 겹쳐 그린다. `runs_out/nexar/<파일명>/timeline_eval.png`.
3. 엔진 결과가 아직 없을 수 있으니, 가짜 frames.jsonl 로 동작을 확인하는 테스트를 `tools/` 나 `tests/` 에 남겨. 실제 결과가 생기면 관리자가 돌린다.

## 범위
tools/eval_nexar.py, viz.py 의 타임라인 확장(기존 동작 유지), 테스트. engine.py, app.py, 스키마 금지.
가상환경 Python 실행 가능하다 (샌드박스 unelevated).
