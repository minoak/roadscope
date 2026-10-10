# 보고 20261007-235724-claude-t07-lane-sanity

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 517.5 초
- 지시서: [tasks/t07-lane-sanity.md](../tasks/t07-lane-sanity.md)

## 바꾼 것
- `engine.py`: 차선 검증을 추가했다(`find_lanes`, `lane_input_mask`). 검증에 걸리면 ego 차선을 비워서 LDW를 판정하지 않는다. YOLO 경로는 `<MODELS_DIR>/yolo11m.pt`로 바꿨다.
- `tools/debug_lanes.py` (새 파일): 프레임별 차선 오버레이를 수정 전/후로 나란히 그린다.
- `runs_out/night_ride_tmp/`를 다시 만들었다. engine 재실행은 성공했고 스키마 위반은 0건이다. 디버그 이미지는 `runs_out/debug/lanes_frame*.png` 7장이다.

## 내가 내린 판단
- 원인:
  - 프레임 0, 2는 횡단보도, 정지선 같은 가로 노면표시가 class 3으로 잡혔다. 그래서 거의 수평인 선(|dx/dy| 5~30)이 차선이 됐다. VP도 수렴하지 않았는데(conf 0.24/0.25) LDW가 판정됐다.
  - 프레임 2의 좌측 선과 프레임 8의 우측 선은 바이크 영역 안의 짧은 lane 조각이다. 화면 중앙에 가깝다는 이유로 뽑혔다.
- 검증 규칙 3가지를 넣었다.
  - 바이크 영역(각 열에서 ego 마스크 최상단보다 아래)의 lane 픽셀을 지운다.
  - 기울기 |dx/dy| ≤ 3.0만 남긴다.
  - VP conf ≥ 0.3이어야 차선을 쓴다.
- 검증에 걸리면 차선 침범 판정과 corridor도 차선 없이 처리한다.
- Depth Pro는 기존 HF 캐시를 그대로 쓴다.

## 확신 없는 부분
- LDW 활성 프레임 수는 23 → 23으로 같지만, 같은 프레임은 8개뿐이다(15개가 빠지고 15개가 새로 생김). 새로 생긴 것 중 32, 98만 실제 차선인지 눈으로 확인했다.
- 기울기 분포에 뚜렷한 경계가 없다. 3.0은 카메라 높이 1.0m를 가정한 기하 계산에 근거한다.
- 차선이 한쪽만 보이면(프레임 8) LDW를 판정하지 않는다. 바이크 틈으로 보이는 실제 차선 일부도 이때 함께 버려진다.
- lane_measurable은 51 → 58, lanes가 있는 프레임은 92 → 72가 됐다.

## 관리자 판단 필요
- 프레임 134: 29.4m 떨어진 차량이 sidelobe 안에 있다는 이유로 BSW L2(위험)가 됐다. sidelobe는 VP를 꼭짓점으로 하는 측면 삼각형이다. 노트북의 zone boost 로직이 거리를 보지 않기 때문인데, 이번 수정으로 이 프레임에 corridor가 생기면서 드러났다. 원본 로직을 고칠지 정해줘.
- Depth Pro 캐시를 models 폴더로 옮길지(약 1.9GB를 다시 받아야 함).

## 다른 실행자에게
- 차선 검증에 걸리면 `lanes`, `corridor`, `vp`가 모두 null이고 `offset_ratio`도 null이다. LDW 사유는 "[측정 불가] 차선 없음" 또는 "[측정 불가] 차선 수렴 안 됨 (VP conf x)"이다.
- 사용법: `python tools/debug_lanes.py <영상> <frame_idx...>`