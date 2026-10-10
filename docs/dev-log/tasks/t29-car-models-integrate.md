# 지시서 t29: 자동차 모드에 사전학습 모델 통합 + t27 마무리, 30개 실행 (쓰기)

## 배경
t27을 하던 중 13:10쯤 데스크톱 앱이 재시작되면서 실행이 끊겼다(네 잘못이 아니다). 마지막 흔적은 geom_car.py 13:08, `runs_out/t27_work/`(try1~4, test_interior.py). 현재 파일 상태를 먼저 확인하고 이어서 해라.

그 사이 민옥이 자동차 모드에 사전학습 모델 두 개를 쓰기로 했다(`.orch/DECISIONS.md` 2026-10-09 마지막 다섯 줄). Codex가 CPU로 시험한 결과가 `runs_out/model_probe/probe.md`, `tools/probe_car_models.py`(로드, 전처리, 후처리 참고), `runs_out/model_probe/overlays/`에 있다. 요약은 이렇다.
- YOLOPv2: 차선 쌍과 소실점이 30개 중 19개 영상에서 잡혔다(lane_focus는 10개). 소실점 y가 연구 수평선과 중앙값 0.007H 차이였다. CPU 0.4초/프레임.
- Mask2Former(Vistas): Ego Vehicle + Car Mount가 30개 모두에서 나왔고 보닛, 대시보드를 잘 덮는다. pos_00413 하단 오검출 박스는 이 영역과 거의 겹친다. neg_01711 왼쪽 끝 실내 물체는 Mask2Former도 Person으로 봐서 이것만으로는 못 뺀다. CPU 2.6초/프레임이라 매 프레임에는 못 쓴다.

## 할 것 (자동차 모드만, 오토바이 모드는 lane_focus 그대로)
1. **t27 내용 마무리:** 소실점 x 20~80%, 신뢰도 low면 hood_bound 안 씀, 침범 |X| < 1.25 m.
2. **차선과 주행 가능 영역은 YOLOPv2로:** 자동차 모드에서 lane_focus 대신 YOLOPv2의 차선 마스크와 주행 가능 영역을 쓴다. 영상 단위 보정의 차선 소실점과 LDW가 이걸 쓰게 한다. YOLOPv2의 차량 검출은 쓰지 않는다(YOLO11m 유지). 가중치 `<MODELS_DIR>\yolopv2.pt`, 원본 코드 `envs\models\YOLOPv2\`(MIT).
3. **내 차 영역은 사전 점검으로:** 판정 전에 영상을 한 번 빠르게 훑어서 내 차 영역을 정한다. 근거는 세 가지다.
   - Mask2Former Ego Vehicle + Car Mount: 보정 구간에서 몇 프레임만 돌려 합친다.
   - YOLO 반복 검출: 영상 전체를 낮은 fps(1~2)로 훑어서, 같은 자리에 같은 크기로 영상 대부분 동안 잡히는 박스를 실내 물체로 본다. neg_01711 왼쪽 끝 person이 이 경우다. 일정 거리로 따라가는 앞차는 크기와 위치가 변하고 사라지므로 구분한다. 구분 기준을 보고해라.
   - t27에서 하던 "움직이는 동안 정지한 영역" 마스크가 쓸 만하면 같이 쓴다.
   - 박스의 상당 부분이 내 차 영역 안에 있으면 판정에서 빼고 objects에 `interior: true` 표시만 남긴다. 사전 점검이 실패하면 박스를 지우지 않는다(미탐 우선).
   - 이건 판정이 아니라 측정 준비라서 한 장면 기준 원칙과 부딪히지 않는다.
4. **확인 이미지:** neg_01711, pos_00413, 낮 고속도로 1개, 밤 1개에 내 차 영역과 YOLOPv2 차선을 겹친 이미지를 `runs_out/nexar_t29/check/`에 남긴다. 관리자가 직접 본다.

## 실행
- 30개 전부를 자동차 모드 기본값으로 돌려 `runs_out/nexar_t29/`에 저장한다. run_log.md에 영상별 실행 시간과 사전 점검 결과(내 차 영역 비율, 반복 검출로 뺀 박스 수)를 남긴다.
- GPU 8GB 안에서 돌아가야 한다. Mask2Former는 사전 점검 때만 올리고 끝나면 내린다. 메모리가 부족하면 Mask2Former를 CPU로 돌려도 된다.
- 오토바이 회귀는 mixkit 하나만 bin이 같은지 확인한다.
- 평가 지표는 Codex가 낸다.

## 범위
- engine.py, geom_car.py, 새 보조 모듈(필요하면 `car_models.py` 같은 파일), `.orch/contracts/frame_schema.md`만 수정한다. app.py, viz.py, tools/ 금지(probe_car_models.py는 읽기만).
- 네트워크 다운로드 금지. 모델은 이미 받아져 있다.
- 30개 실행이 다 끝난 걸 확인한 뒤에 보고한다. 백그라운드로 걸어두고 먼저 끝내지 마라.
- FCW 거리 기준(3/5/10 m), TTC 기록만, 유령 박스는 바꾸지 않는다.
