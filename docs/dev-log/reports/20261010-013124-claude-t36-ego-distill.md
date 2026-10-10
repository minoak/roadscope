# 보고 20261010-013124-claude-t36-ego-distill

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 5873.0 초
- 지시서: [tasks/t36-ego-distill.md](../tasks/t36-ego-distill.md)

## 바꾼 것
- tools/distill/make_labels.py: 엔진 calibrate_car를 import해 영상별 내 차 마스크와 프레임 12장을 저장하고 제외 기준을 적용
- tools/distill/train_ego.py: U-Net R34 학습 (영상 단위 240/30, 겹침 없음)
- tools/distill/eval_ego.py: IoU(낮, 밤), 처리 시간, 확인 그림
- tools/distill/eval_parts.py (추가): 보닛 띠와 그 위 실내를 나눠 재현율 측정
- runs_out/distill/: labels.csv, md, train_log.csv, learning_curve.png, ego_unet_r34.pt, eval.md, eval_parts.md, check/ 10장

## 내가 내린 판단
- 제외 기준: Mask2Former 표본 마스크와 투표 결과의 IoU 평균 0.75 미만, 또는 마스크 면적 2~70% 밖. 학습 270개 중 50개(투표 48, 면적 2), 시험 3개가 빠짐. 움직임 판정 실패는 빼지 않고 표시만 함
- 입력 512×288. 증강은 밝기, 대비, 감마, ±4% 이동, 0.93~1.07 확대, 좌우 반전 0.5. 손실 BCE+Dice, 40 epoch 중 검증 최고인 epoch 13을 저장
- 인코더: 로컬 캐시의 smp-hub/resnet34.imagenet (torchvision ImageNet 가중치와 같은 값인지 확인함)
- 확인 그림: 30개 전체의 최저 5개(이상치, 제외 영상 3개 포함), 본 집계의 최저 5개, 지정 2개를 합쳐 10장

## 확신 없는 부분
- 시험 IoU 중앙값 0.862 (낮 0.887, 어두움 0.847, n=24), 검증 0.886. 한 장 처리 시간은 GPU 8.5 ms, CPU 70 ms(4스레드 93 ms)
- 보닛 띠는 재현율 0.93~0.96으로 잘 잡지만, 그 위 실내는 0.43~0.58. neg_01711 왼쪽 끝 실내 물체를 놓치고, A필러, 룸미러도 놓침
- 보닛 위 흐린 반사 띠를 더 칠함. 이 띠 너머로 도로가 보이는 프레임이 있어서, 이 마스크로 박스를 지우면 미탐 위험이 있음
- 학생이 나아 보인 경우: pos_00283, pos_00413. 선생은 수평 띠인데 학생은 실제 보닛, 대시보드 곡선을 따라감
- 제외 기준이 너무 엄격했을 수 있음. 빠진 영상 2개를 직접 보니 정답이 멀쩡했고, 검증 쪽 제외 영상에서 학생 IoU가 0.87~0.93
- 검증의 어두운 영상은 1개뿐이라 검증 낮, 밤 비교는 의미가 약함. 검증 손실은 epoch 9부터 오름(과적합 기미)

## 관리자 판단 필요
- 엔진에 넣을지(지시대로 아직 안 넣음). 넣는다면 보닛 띠만 학생으로 바꾸고 위, 옆 실내는 사전 점검을 유지할지
- 제외 기준을 완화해 다시 학습할지

## 다른 실행자에게
- 내 차 영역 마스크, 정보는 nexar_t34와 nexar_t40이 30/30 같음(t40 영향 없음). 정답은 엔진 출력 interior_seg0과 IoU 1.0
- 학생 사용법: smp.Unet('resnet34', classes=1)에 state_dict를 불러옴. 입력은 RGB 512×288, ImageNet 정규화. sigmoid > 0.5면 내 차. 예시는 eval_ego.student()
- 정답 생성은 영상당 약 35초, 프로세스당 GPU 1.7GB. 3개를 같이 돌리면 6.3GB