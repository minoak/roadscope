# 보고 20261009-162357-claude-t32-ldw-lane-baseline

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 823.6 초
- 지시서: [tasks/t32-ldw-lane-baseline.md](../tasks/t32-ldw-lane-baseline.md)

## 바꾼 것
- geom_car.py: 평소 위치 계산 함수 추가(t31과 같은 식: 0.95H 기준 행, 중앙값, ±0.5 m 상한, 최소 근거 max(30, 10%)), meta `lane_baseline` 출력
- engine.py: 사전 점검에서 구간별로 YOLOPv2 차선 마스크를 모아 평소 위치 b 계산(`_lane_baseline`). LDW의 내 차 기준 열만 `vp_x − b/k`로 옮김
- frame_schema.md: `car_calib[].lane_baseline` 필드와 LDW 설명 추가
- 결과: runs_out/t32_smoke/ (스모크 2개, baseline_check.md, summary.md, diff)

## 내가 내린 판단
- 근거 프레임: 보정 구간 프레임과 그 뒤 평가 프레임마다 YOLOPv2를 돌림. 구간이 끝나 내 차 영역이 정해진 뒤, 판정과 같은 차선 고르기로 표본을 만듦. 최소 근거 10%의 분모는 본 프레임 수
- 차선 고르기(find_lanes)의 기준 열은 소실점 열 그대로 두고, LDW 판정만 옮김. t31 재판정과 같은 범위
- GPU 차단은 `CUDA_VISIBLE_DEVICES=-1`로 함. Windows에서는 빈 값을 넣으면 변수가 삭제돼 GPU가 그대로 보였음. 첫 시도는 확인 구문에서 멈췄고, 모델, 연산은 GPU에 올라가지 않음

## 확신 없는 부분
- 계산 함수 대조: t29 저장 결과를 넣으면 30/30 영상이 t31 표와 정확히 일치
- 엔진 경로(앞 15초, CPU): 같은 구간 기록으로 계산한 값과 비교
  - neg_01539: −0.138 대 −0.155 m
  - pos_00004: 원값 −0.679로 같고, −0.5 m 상한 적용
- 새 LDW 후보: t31은 새 후보를 만들 수 없었지만 엔진은 만듦
  - pos_00004: 왼쪽 55 → 14프레임
  - neg_01539: 왼쪽 7 → 5, 오른쪽 3 → 8. 반대쪽이 늘었음
- 30개 재실행은 하지 않아 전체 효과는 확인 못 함
- 사전 점검이 늘어남(구간 뒤 평가 프레임마다 YOLOPv2: CPU 0.23초/프레임, GPU는 약 20ms로 예상)

## 관리자 판단 필요
- neg_01539처럼 반대쪽에 새 LDW가 생기는 효과를 감안해, GPU로 30개를 재실행해 확인할지

## 다른 실행자에게
- meta `car_calib[].lane_baseline`이 추가됨: `baseline_m`이 null이면 미적용, `ldw_shift_px`는 기준 열 이동량
- Windows에서 GPU를 끄려면 `CUDA_VISIBLE_DEVICES=-1`을 써야 함(빈 값은 무효)
- 오토바이 경로: 공용 코드 중 바뀐 건 judge_lane뿐이고, 오토바이는 기준 열이 그대로라 동작이 같음. 나머지 변경은 자동차 전용 함수(diff: moto_path_diff_judge_lane.txt)