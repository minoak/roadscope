# 지시서 t28: 자동차용 사전학습 모델 두 개 시험 (쓰기, 실험 결과 파일만)

## 배경
자동차 모드에 사전학습 모델 두 개를 넣기로 했다(`.orch/DECISIONS.md` 2026-10-09 마지막 두 줄). 엔진에 넣기 전에 Nexar 영상에서 실제로 쓸 만한지 확인한다. 엔진 통합은 이 결과를 보고 Claude가 한다.

- **YOLOPv2:** 가중치 `<MODELS_DIR>\yolopv2.pt`(TorchScript), 원본 코드 `<MODELS_DIR>\YOLOPv2\`(demo.py, utils/utils.py의 전처리, 후처리 참고). 출력: 차량 검출, 주행 가능 영역 마스크, 차선 마스크.
- **Mask2Former (Mapillary Vistas semantic):** HF 캐시의 `facebook/mask2former-swin-large-mapillary-vistas-semantic`. transformers로 로드한다(`Mask2FormerForUniversalSegmentation`, `AutoImageProcessor`). 클래스 이름은 config의 id2label에서 확인한다. Ego Vehicle, Car Mount, Lane Marking 계열, Road를 쓴다.

## 할 것
1. `tools/probe_car_models.py`를 만든다. `runs_in/nexar/`의 30개 영상에서 영상당 프레임 3장(10%, 45%, 75% 시점)을 뽑아 두 모델을 돌린다.
2. 영상마다 아래를 낸다.
   - Mask2Former 내 차 영역(Ego Vehicle + Car Mount): 화면 비율, 영역의 가장 위 행. t24 보정의 보닛 행(`runs_out/nexar_t24/<파일명>/meta.json`의 car_calib hood_row)과 비교한다.
   - neg_01711: 화면 왼쪽 끝 x 0~80, y 280~560 영역이 내 차 영역으로 잡히는지(민옥 테스트에서 이 영역이 person으로 잡혀 치명이 떴다). pos_00413의 화면 하단 넓은 박스 영역도 같은 방식으로 본다.
   - YOLOPv2 차선 마스크로 구한 차선 두 줄과 교점(소실점): 몇 개 영상에서 잡히는지. 연구 결과 수평선(`.orch/research/car_dashcam_2026-10-09/data/per_clip.csv`의 hz_y, 화면 높이 비율)과 비교한다. 기존 lane_focus는 30개 중 10개에서만 차선 소실점이 잡혔다.
   - 프레임당 처리 시간(CPU). GPU 시간은 추정만 적는다.
3. 영상 6개 정도(낮, 밤, 실내가 많이 보이는 것, neg_01711, pos_00413 포함)는 두 모델 마스크를 겹친 이미지를 `runs_out/model_probe/overlays/`에 남긴다. 관리자가 직접 본다.
4. 결과 표와 요약을 `runs_out/model_probe/probe.md`에 쓴다.

## 범위
- **GPU 쓰지 마라.** Claude가 t27로 GPU를 쓰며 30개를 돌리는 중이다. CPU로만 돌린다(`device='cpu'`). 프레임 수가 적어서 충분하다.
- 수정 가능: `tools/probe_car_models.py`(새 파일), `runs_out/model_probe/`. engine.py, geom_car.py, viz.py, app.py 금지.
- 네트워크 다운로드 금지. 모델은 이미 받아져 있다. 로드가 안 되면 원인만 보고해.
