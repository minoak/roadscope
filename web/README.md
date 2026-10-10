# RoadScope web

로컬 시연을 위해 FastAPI 서버 하나와 정적 화면(`static/`)으로 구성했다. 실행 방법은 리포 루트의 README에 있다.

서버는 단일 worker로 실행하며, 영상을 한 번에 하나씩 분석한다. 엔진은 `python -m roadscope.engine` 하위 프로세스로 실행한다. 이 모듈을 import하는 것만으로는 모델을 메모리에 올리지 않는다.

## 환경변수

| 변수 | 뜻 |
|---|---|
| `ROADSCOPE_DISABLE_ENGINE=1` | 업로드와 엔진 실행을 막는다 (결과 조회 전용) |
| `ROADSCOPE_DEV=1` | 로컬 요청에 한해 `/dev/load?run=<runs_out 아래 상대 경로>`로 저장된 결과를 연다 |

`/dev/load`로 열 수 있는 폴더는 `runs_out/` 안에서 `meta.json`과 `frames.jsonl`을 갖춘 폴더뿐이다. 저장된 결과를 바꾸지 않고, `hud.mp4`가 없으면 새로 만들지 않는다.

## API

- `POST /api/jobs`: multipart `video`(MP4/MOV, 최대 500 MiB), `vehicle`(`car` 또는 `moto`). 202 `{id, url}`. 업로드 저장이 끝나면 백그라운드 큐에 넣는다.
- `GET /api/jobs/{id}`: `status`, `progress`(0~100), `stage`, `stage_index`, `completed_frames`, `video`, `error`. 완료되면 응답에 `summary`, `files`를 추가한다. 서버의 파일 경로는 응답에 넣지 않는다.
- `GET /api/jobs/{id}/files/{hud|frames|timeline}`: H.264 MP4, JSONL, PNG. MP4는 byte-range 재생을 지원한다. `?attachment=true`면 내려받는다.
- `GET /api/config`: 업로드 가능 여부와 파일 크기 제한을 반환한다.

업로드와 작업 상태는 `web/.data/jobs/<id>/`, 엔진 출력은 `web/.data/results/job_<id>/`에 저장한다. 서버를 재시작해도 완료된 결과는 남는다. 진행 중이던 작업은 중단으로 표시하며 다시 실행하지 않는다.

자동차는 엔진 기본 FPS, 오토바이는 5 fps로 평가한다.

진행률은 엔진이 기록한 마지막 프레임의 시각을 입력 영상 길이로 나누어 추정한다. 남은 시간은 표시하지 않는다.

## 요약 계산 (`summary.py`)

- `segments`: `[start, end)` 동안 직전 프레임 등급을 유지한다. 같은 등급은 합친다.
- `danger`는 등급 2 이상, `caution`은 등급 1이다. `count`와 `intervals`는 아래의 표시용 묶기를 적용한 구간이고, `duration`은 원래 기록의 등급별 누적 시간이다.
- `peak`: 최고 등급이 처음 나온 시각과 해당 프레임의 `top_reason`을 기록한다.
- `events`: 화면에 표시하는 경고 종류별로 시작과 해제 시각을 기록한다. 등급이나 방향이 바뀐 시각도 남긴다.
- 영상 길이: 업로드는 실제 영상 길이를 쓴다. 저장 결과는 자동차 보정의 `end_frame`(포함) + 1을 `fps_src`로 나누고, 없으면 마지막 평가 간격으로 추정해 `duration_source`에 적는다.
- `cameras`: 모든 보정 구간을 보여준다. `ldw_measurable=false`인 구간은 차선 영점 실패로 `lane_baseline_status=failed`로 표시한다.
- 신뢰 낮음 띠는 자동차 결과에서 보정 구간 중 하나라도 `horizon_conf=failed`일 때만 표시한다.

### 짧은 경고 끊김 묶기

짧게 끊긴 경고는 목록에서 이어서 보여주고, 요약 카드의 구간 수도 이에 맞춰 센다. 이 처리는 화면 표시에만 쓰며, 기준은 `DISPLAY_MERGE_GAP_SECONDS`(0.5초) 하나다.

경고 종류와 방향이 같고 같은 등급 이상이 다시 나오면, 그사이 낮은 등급이나 해제 상태가 0.5초 미만으로 이어진 구간을 높은 등급으로 묶는다. 끊긴 시간이 0.5초 이상이거나 중간에 반대 방향 경고가 나오면 묶지 않는다. 경고 종류가 달라도 묶지 않는다.

원본 `frames.jsonl`과 타임라인 색 띠는 바꾸지 않는다. 최고 위험 시각과 등급별 누적 시간도 그대로 둔다.

## 남은 일

- 업로드와 결과를 자동으로 지우는 정책이 없다. 저장된 파일은 직접 지워야 한다.
- 외부 공개 배포를 전제로 만들지 않았다. `127.0.0.1`에서 실행한다.
