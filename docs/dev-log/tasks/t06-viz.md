# 지시서 t06: 시각화와 화면 구현 (쓰기)

## 목적
서비스 MVP의 시각화와 화면은 네 담당이야. 블랙박스 영상을 넣으면 HUD 가 얹힌 영상과 위험도 타임라인을 보여주는 Gradio 화면까지.
너는 t04에서 시각화 설계를 했어. 그 안을 기준으로 해. 여왕 결정에 따라 객체는 YOLO 검출 박스, 거리는 Depth Pro 기반이 된다.
먼저 `.orch/DECISIONS.md` 와 `.orch/contracts/frame_schema.md` 를 읽어. 시각화는 이 스키마만 보고 그린다.

## 만들 것
1. `viz.py`
   - `render_frame(bgr, record) -> bgr`: 프레임 1장에 HUD. t04 설계안(등급 배지, 활성 경고와 사유, 경고 박스와 거리, 얇은 차선, 방향별 거리, 차선 측정 불가 표시)
   - `render_video(video_path, run_dir, out_path)`: 원본 영상 + frames.jsonl → HUD 영상 mp4. 평가 fps 가 원본보다 낮으면 사이 프레임은 직전 레코드를 유지
   - `render_timeline(run_dir, out_path)`: 시간축 위험도(bin) 그래프 + 경고 종류별 구간 표시 PNG
2. `app.py` (Gradio): 영상 업로드 → engine 실행 → HUD 영상, 타임라인, 경고 이벤트 표(시간, 경고, 사유, 거리) 표시. engine 은 다른 실행자가 `engine.py` 로 만들고 있다 (`python engine.py <영상> --fps 5` → `runs_out/<영상이름>/`). 아직 없으면 subprocess 호출 인터페이스만 맞춰두고 화면은 mock 데이터로 확인해.
3. `tools/mock_frames.py`: 엔진 없이 시각화를 확인할 mock frames.jsonl 생성기. outputs/*.json (mvp.py 결과) 과 samples 이미지를 이용해도 좋다.

## 참고
- archive `<ARCHIVE>/review_video/*_v6_hud.mp4`, `final_report/` 의 이전 HUD 결과. 디자인 기준으로 보되 그대로 복제할 필요는 없다.
- 한글 텍스트가 깨지지 않게 폰트를 처리해 (Windows 맑은 고딕 등).

## 실행 환경
- Python: `python`
- 관리자가 지금 gradio 등을 설치 중이다. `<VENV>-install2.log` 끝에 DONE 이 찍히면 완료.
- 패키지 추가 설치 금지.

## 범위
- engine.py, mvp.py, src/, 노트북, samples/ 수정 금지. 스키마 변경이 필요하면 보고로.
- Gradio 서버를 띄워둔 채로 끝내지 마. 확인 후 종료.

## 완료 조건
mock 데이터로 HUD 영상 mp4 1개와 타임라인 PNG 1개가 생기고, app.py 가 import 에러 없이 뜬다. 결과 파일 경로를 보고에 적어.
