# 지시서 t07: 차선 검출 검증 + 정리 (쓰기)

## 배경
관리자가 viz.py 로 runs_out/night_ride_tmp 의 HUD 영상을 렌더해서 확인했다 (runs_out/night_ride_tmp/hud.mp4).
- frame_idx 0 (LDW L1 "좌측 편향 19%"): 차선 두 줄이 거의 수평에 가깝고, 내 바이크 핸들 높이를 가로지른다. 도로 차선이 아니라 바이크 쪽 경계를 잡은 것으로 보인다. LDW 판정이 이 선을 근거로 나왔다.
- frame_idx 2, 8: 차선이 화면 하단 바이크 위로 길게 이어진다. 바이크(ego) 영역에서 나온 선분이 섞였을 가능성.

## 할 것
1. 프레임 0, 2, 8 에서 lane 마스크(class 3)와 ego 마스크, Hough 선분, 최종 left/right 를 확인해서 원인을 찾아. 확인용 이미지는 `runs_out/debug/` 에 저장해도 된다.
2. 원인에 맞는 최소한의 검증을 engine 쪽에 넣어 (예: ego 마스크 영역과 겹치는 선분 제외, 기울기 범위, VP 수렴 조건). 노트북/mvp.py 원본 로직은 고치지 말고 engine.py 에서 감싸거나 전처리로.
3. 검증에 걸려 차선을 못 쓰는 프레임은 lane_measurable=false 로 두고 LDW 를 판정하지 않는다.
4. YOLO 가중치를 archive 에 내려받았는데, archive 는 읽기 전용으로 쓰기로 했다. 관리자가 `<MODELS_DIR>/yolo11m.pt` 로 옮겼다. engine.py 의 경로를 거기로 바꾸고, 앞으로 모델 다운로드는 그 폴더로 한다.
5. 수정 후 임시 영상으로 engine.py 를 다시 돌려 runs_out/night_ride_tmp 를 갱신.

## 범위
src/, 노트북, samples/, mvp.py, viz.py, app.py 수정 금지.

## 완료 조건
세 프레임의 원인 설명(보고에 짧게), 수정 전후 LDW 활성 프레임 수 비교, engine 재실행 성공.
