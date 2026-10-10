# 보고 20261009-124617-codex-t28-car-model-probe

- 실행자: codex, 모델: gpt-6-astra, 상태: done, 종료 코드: 0, 소요: 833.8 초
- 지시서: [tasks/t28-car-model-probe.md](../tasks/t28-car-model-probe.md)

## 바꾼 것

- `tools/probe_car_models.py`: 오프라인 CPU 전용 실험 스크립트를 추가했사와요.
- probe.md: 30영상, 90프레임 결과와 비교표를 기록했사와요.
- `runs_out/model_probe/`: 중첩 이미지 18장, 원시 마스크 90세트, CSV, JSON, 검산 기록을 저장했사와요.

## 내가 내린 판단

- 비교 좌표는 1280×720, pos_00413 관심 영역은 t24의 인접 시각 실제 검출 박스로 정했사와요.
- 소실점 후보는 19/30영상, 40/90프레임에서 얻었답니다.
- neg_01711 문제 영역은 Person 오분류가 남아 Mask2Former 단독 억제 근거가 부족하다고 판단했사와요.
- CPU 중앙값은 YOLOPv2 0.398초, Mask2Former 2.652초이며 차선 기하 후처리는 별도랍니다.

## 확신 없는 부분

GPU 시간은 가정에 따른 추정뿐이랍니다. 기존 lane_focus와 평가 조건이 달라 정확도 개선을 단정할 수 없으며, 영상당 3장으로 전체 영상의 억제 안전성을 검증하지 못했사와요.

## 관리자 판단 필요

없음

## 다른 실행자에게

Mask2Former 캐시 기본 입력은 384×384, Ego Vehicle=64, Car Mount=63이랍니다. 연구 CSV의 추가 영상은 `xneg_`, `xpos_` 접두사 연결이 필요하와요. `--refresh-artifacts`로 추론 없이 그림과 보고서를 재생성할 수 있사와요.