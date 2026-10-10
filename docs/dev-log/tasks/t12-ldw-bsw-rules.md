# 지시서 t12: LDW/BSW 판정 기준 변경 (쓰기)

`.orch/DECISIONS.md` 의 2026-10-08 마지막 네 줄이 이번 기준이다. 먼저 읽어.

## LDW
- 편향 비율만으로 L1 을 띄우는 경로(현재 15% 기준)를 끈다. 비율 계산과 offset_ratio 기록은 유지.
- t10 의 "차선 위 주행"(이탈 쪽 실제 차선이 바이크 아래까지 이어짐)만 L1. 시간 축 보정은 이 판정이 끊길 때만 이어주도록 제한.

## BSW
- 측면 구역(sidelobe) 겹침만으로 L0→L2 올리는 zone boost 를 폐지.
- 객체마다 옆 간격(횡방향, m)을 계산: 박스 하단 중심 x 와 VP x 의 차이, 종방향 거리(distance_m), f_px 로 lateral ≈ (cx - vp_x) × Z / f. VP 가 없으면 기본 VP(노트북 fallback)를 쓰되 레코드에 그 사실이 남게.
- 나란히 판정: |lateral| 이 차선 1폭 + 옆차선 절반(1.5 × LANE_WIDTH, LANE_WIDTH 는 노트북 상수 3.5m)이내이고 내 차로 안(0.5 × LANE_WIDTH 미만)이 아니며, 종방향 거리 ≤ BSW_SIDE_MAX_M(기본 8.0, 상수로 노출)이면 BSW L1 "측면 나란히".
- 기존 근접 기준으로 L2 이상이 되는 경로는 유지.
- 차선 침범 차량(check_lane_viol_strict / x1 로 잡히는 것)은 무조건 L1 이상. 지금 이 경로가 L0 으로 떨어지는 경우가 있으면 막아.
- 스키마: objects 에 `lateral_m` (소수 1자리, 계산 불가면 null) 를 추가해도 된다 (관리자 승인). `.orch/contracts/frame_schema.md` 의 objects 예시에 이 키를 한 줄 추가해.

## 확인
mixkit, night_ride_tmp 두 영상으로 engine 재실행. 수정 전후: 등급 분포, LDW 활성 수, BSW 레벨별 수, 40m 이상 차량으로 L2 이상 나오는 프레임 수(0 이어야 함). 측면 나란히로 잡힌 프레임 몇 개의 lateral_m / distance_m 값.

## 범위
engine.py 와 frame_schema.md 의 objects 한 줄만. mvp.py, src/, 노트북, viz.py, app.py 금지 (app.py 는 지금 Codex 가 고치는 중).
