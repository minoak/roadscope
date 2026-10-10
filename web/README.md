# RoadScope web

FastAPI 서버 하나와 정적 화면(`static/`)으로 이루어진 로컬 시연용 웹이다. 실행 방법은 리포 루트의 README를 본다. 서버는 단일 worker로 실행하고, 분석은 한 번에 하나씩 처리한다. 엔진은 `python -m roadscope.engine` 하위 프로세스로 실행하므로 이 모듈을 import해도 모델을 올리지 않는다.

## 환경변수

| 변수 | 뜻 |
|---|---|
| `ROADSCOPE_DISABLE_ENGINE=1` | 업로드와 엔진 실행을 막는다 (결과 조회 전용) |
| `ROADSCOPE_DEV=1` | 로컬 요청에 한해 `/dev/load?run=<runs_out 아래 상대 경로>`로 저장된 결과를 연다 |

`/dev/load`는 `runs_out/` 안의 `meta.json`과 `frames.jsonl`이 있는 폴더만 받는다. 저장된 결과를 바꾸지 않고, `hud.mp4`가 없으면 새로 만들지 않는다.

## API

- `POST /api/jobs`: multipart `video`(MP4/MOV, 최대 500 MiB), `vehicle`(`car` 또는 `moto`). 202 `{id, url}`. 업로드 저장이 끝나면 백그라운드 큐에 넣는다.
- `GET /api/jobs/{id}`: `status`, `progress`(0~100), `stage`, `stage_index`, `completed_frames`, `video`, `error`. 완료되면 `summary`, `files`가 붙는다. 서버 파일 경로는 응답하지 않는다.
- `GET /api/jobs/{id}/files/{hud|frames|timeline}`: H.264 MP4, JSONL, PNG. MP4는 byte-range 재생을 지원한다. `?attachment=true`면 내려받는다.
- `GET /api/config`: 업로드 가능 여부와 파일 크기 제한.

업로드와 작업 상태는 `web/.data/jobs/<id>/`, 엔진 출력은 `web/.data/results/job_<id>/`에 저장한다. 서버가 재시작되면 완료된 결과는 남고, 진행 중이던 작업은 중단으로 표시하며 다시 실행하지 않는다. 자동차는 엔진 기본 FPS, 오토바이는 5 fps로 평가한다.

진행률은 엔진이 쓴 마지막 프레임 시각을 입력 영상 길이로 나눠 추정한다. 남은 시간은 표시하지 않는다.

## 요약 계산 (`summary.py`)

- `segments`: `[start, end)` 동안 직전 프레임 등급을 유지한다. 같은 등급은 합친다.
- `danger`는 등급 2 이상, `caution`은 등급 1이다. `count`와 `intervals`는 아래의 표시용 묶기를 적용한 구간이고, `duration`은 원래 기록의 등급별 누적 시간이다.
- `peak`: 최고 등급이 처음 나온 시각과 그 프레임의 `top_reason`.
- `events`: 표시용 경고 종류별로 시작, 등급이나 방향이 바뀐 시각, 해제를 기록한다.
- 영상 길이: 업로드는 실제 영상 길이를 쓴다. 저장 결과는 자동차 보정의 `end_frame`(포함) + 1을 `fps_src`로 나누고, 없으면 마지막 평가 간격으로 추정해 `duration_source`에 적는다.
- `cameras`: 모든 보정 구간을 보여준다. `ldw_measurable=false`인 구간은 차선 영점 실패로 `lane_baseline_status=failed`로 표시한다.
- 신뢰 낮음 띠는 자동차 결과에서 보정 구간 중 하나라도 `horizon_conf=failed`일 때만 표시한다.

### 짧은 경고 끊김 묶기

경고 목록과 요약 카드의 구간 수에만 쓰는 화면 표시용 처리다. 기준은 `DISPLAY_MERGE_GAP_SECONDS`(0.5초) 하나다. 같은 경고 종류와 방향에서 같은 등급 이상이 다시 나오면, 그 사이의 낮은 등급이나 해제가 0.5초 미만일 때 높은 등급으로 잇는다. 0.5초 이상 끊긴 경우, 반대 방향이 끼어든 경우, 다른 경고 종류는 묶지 않는다. 원본 `frames.jsonl`, 타임라인 색 띠, 최고 위험 시각, 등급별 누적 시간은 바꾸지 않는다.

## 남은 일

- 업로드와 결과를 자동으로 지우는 정책이 없다. 저장된 파일은 직접 지워야 한다.
- 외부 공개 배포를 전제로 만들지 않았다. `127.0.0.1`에서 실행한다.
