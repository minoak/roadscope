# 지시서 t21: Nexar 30개 평가 준비와 앱 차량 선택 (쓰기)

## 배경
자동차 모드를 기하 기반으로 다시 짜는 중이다(Claude가 t20에서 engine.py 담당). 평가는 Nexar 30개 영상으로 한다. 지금 `runs_in/nexar/`에는 10개만 있다. 나머지 20개는 이 PC의 Hugging Face 캐시에 이미 받아져 있다(`nexar-ai/nexar_collision_prediction`, 보통 `<HF_CACHE>\datasets--nexar-ai--nexar_collision_prediction\snapshots\<해시>\train\{positive,negative}\`). 결정 내용은 `.orch/DECISIONS.md`의 2026-10-09 줄을 보라.

추가할 20개 영상 id:
- negative: 01211, 01489, 01539, 01584, 01711, 01811, 01966
- positive: 00004, 00071, 00142, 00310, 00602, 00655, 00754, 00845, 00864, 00872, 00877, 00896, 01019

## 할 것
1. **영상 추가.** 위 20개를 캐시에서 `runs_in/nexar/`로 복사한다. 이름은 기존 규칙(`pos_00004.mp4`, `neg_01211.mp4`)을 따른다. 캐시의 `train/positive/metadata.csv`와 `train/negative/metadata.csv`로 `labels.csv`에 20줄을 추가한다(기존 열 그대로). 다시 할 수 있게 작은 스크립트로 만든다(`tools/add_nexar_from_cache.py`). 네트워크 다운로드는 하지 않는다. 캐시에 없는 파일이 있으면 그 목록만 보고해.
2. **평가 도구.** `tools/eval_nexar.py`가 결과 폴더를 인자로 받게 한다(예: `--runs runs_out/nexar_t20`). 30개 기준으로 아래를 낸다.
   - 기존 지표: 미탐(사고 3초 전~사고 사이에 위험 이상 없음), 선행 시간, 사고 밖 구간의 위험 이상 비율
   - 주의(L1) 비율: 정상 영상과 사고 영상의 사고 밖 구간 각각
   - 밤/낮, 도로 유형(scene)별 묶음
   - 기존 10개는 이전 결과(`runs_out/nexar/`)와 나란히 비교
   - frames.jsonl에 TTC나 보정값 같은 새 필드가 있으면 경고 이유별(예: TTC로 올라간 프레임 수) 집계. 새 필드 이름은 t20이 `.orch/contracts/frame_schema.md`에 추가한다. 없거나 아직 안 정해졌으면 건너뛰게 만든다.
   - 결과 폴더가 일부만 있어도 있는 것만으로 돌아가야 한다.
3. **앱.** app.py에 차량 선택(오토바이/자동차)을 넣고 engine 호출에 `--vehicle`을 넘긴다. fps 기본값은 오토바이 5로 두고, 자동차는 engine의 자동차 모드 기본값을 따르게 한다(값을 넘기지 않음). 기존 오토바이 흐름은 그대로 동작해야 한다.

## 확인
- `labels.csv`가 30줄이고 30개 영상 파일이 열리는지(프레임 수, fps, 해상도) 표로 보고한다.
- `tools/eval_nexar.py --runs runs_out/nexar`로 기존 10개 결과가 이전과 같은 숫자로 나오는지 확인한다.
- app.py는 임포트와 화면 구성까지만 확인해. 실제 영상 분석은 돌리지 않는다.

## 범위
- 수정 가능: `tools/add_nexar_from_cache.py`(새 파일), `tools/eval_nexar.py`, `app.py`, `runs_in/nexar/`(영상 복사와 labels.csv).
- 금지: engine.py, viz.py, `.orch/contracts/`. engine 실행과 GPU 사용도 금지(Claude가 GPU로 실행 중).
