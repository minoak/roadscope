# 지시서 t05: 영상 엔진 구현 (쓰기)

## 목적
서비스 MVP의 엔진. 블랙박스 영상 → 프레임 추출 → 프레임별 위험도 판정 → `.orch/contracts/frame_schema.md` 형식의 frames.jsonl.
너는 t01, t03에서 노트북과 mvp.py를 다뤘어. 판정 로직은 mvp.py를 기반으로 한다.
먼저 `.orch/DECISIONS.md` 와 `.orch/contracts/frame_schema.md` 를 읽어.

## 구성 (여왕 결정)
- 차선: 기존 lane_focus U-Net R34 5-fold. 가중치 `<ARCHIVE>/lane_focus_fold0~4_best.pth` (복사하지 말고 경로 참조). 로딩 코드는 노트북 §8 참고. 프레임마다 다수결 마스크 → 차선/VP/ego 마스크.
- 객체: YOLO 계열 (ultralytics). 마스크 덩어리 기반 extract_objects 를 검출 박스로 교체. ego(내 오토바이) 조각 제거 로직은 마스크 기준으로 유지하거나 검출 박스에 맞게 고쳐.
- 거리: Depth Pro (`apple/DepthPro-hf`, transformers). 노트북 load_depth_pro 참고. 처음 실행 때 HF에서 받는다 (약 1.9GB, 허용).
- 트래킹: src/simple_tracker.py 의 IoU 트래커를 연결해서 track_id 를 붙인다. 필요하면 ultralytics 내장 트래커를 써도 되는데, 그 판단은 보고에 적어.
- GPU 사용 (RTX 4070 Laptop 8GB). 메모리가 부족하면 모델 순차 실행이나 fold 수 축소를 고려하되 결정은 보고로 올려.

## 만들 것
- `engine.py`: `python engine.py <영상경로> --fps 5` → `runs_out/<영상이름>/meta.json`, `frames.jsonl`
- 프레임 단위 함수도 노출: `assess_frame(bgr_image) -> 레코드 dict` (Gradio 화면이 쓸 수 있게)
- 원본 영상이 아직 없다. 테스트는 archive 의 `www.acmeai.tech ODataset 1 - Motorcycle Night Ride Dataset/images/night ride (N).png` 원본 이미지들(`___fuse`, `___save` 붙은 파일은 제외)을 번호 순으로 이어 붙인 임시 mp4 로 해. 이 임시 영상 생성 스크립트도 `tools/frames_to_video.py` 로 남겨.
- v6 영상 파이프라인 코드가 archive 어딘가에 있을 수 있다 (`review_video/*_v6_tracked_hud.mp4` 결과가 있음). 찾으면 참고하고 경로를 보고에 적어. 찾는 데 오래 쓰지 마.

## 실행 환경
- Python: `python`
- 관리자가 지금 torch(CUDA), ultralytics, transformers, gradio 를 설치 중이다. `<VENV>-install2.log` 끝에 DONE 이 찍히면 완료. 그 전에는 코드 읽기와 작성부터 해.
- 패키지 추가 설치 금지. 필요하면 보고에 올려.

## 범위
- src/, 노트북, samples/, mvp.py 의 기존 동작은 수정 금지 (mvp.py 에서 함수를 import 하는 건 괜찮다).
- 시각화(HUD, 영상 합성, 화면)는 만들지 마. Codex 담당.

## 완료 조건
임시 영상으로 `python engine.py` 가 끝까지 돌고, frames.jsonl 이 스키마를 따른다. 프레임당 처리 시간과 GPU 메모리 사용량을 보고에 적어.
