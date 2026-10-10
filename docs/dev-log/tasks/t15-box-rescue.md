# 지시서 t15: 놓친 박스 되살리기 (쓰기)

`.orch/DECISIONS.md` 2026-10-08 마지막 두 줄이 이번 결정이다. 미탐 우선 원칙(게시판 공지)을 기준으로.

## 배경
accident_0029: 5.4s 에 차를 0.82 로 잡았는데 5.6s 부터 미검출 → 안전. 6.0s 차는 거리 1.2m 로 정확하지만 박스가 없어 판정에 안 들어감. 5.8s 에는 화면 전체를 person 0.26 으로 오검출.

## 할 것 (위에서 아래로 겹친다. 각 겹은 플래그로 끌 수 있게)
1. 낮은 확신도 구제: conf 0.25 미만이라도 "크고 가까운" 박스(면적 비율, 하단 위치 기준. 값은 네가 근거와 함께 정해)는 conf 0.10 까지 받는다. 클래스는 믿지 말고 `cls` 를 "unknown_near" 같은 일반 근접 객체로 표시. 거리와 근접 판정에만 쓴다.
2. 마스크 보조 박스: lane_focus 의 class 4(Moveable) 연결 성분 중, 어떤 YOLO 박스와도 겹치지 않는 큰 덩어리를 보조 박스로 추가 (cls "mask_object"). ego(class 5, 6) 와 겹치는 건 제외. 작은 덩어리는 버린다(기준은 네가 정해).
3. 트래커 유령 반영: simple_tracker 의 ghost(최근 놓친 track)를 직전 움직임으로 예측한 박스로 판정에 넣는다. 유지 시간은 기존 ghost 설정(0.5초 안팎). 다가오던(박스가 커지던) track 이 사라지면 직전 판정 이상을 유지. 유령은 레코드에 표시(`ghost: true`).
4. 거리 성분이 두 개만 남고 2배 이상 차이 나면 가까운 쪽 사용.
5. 스키마: objects 에 `source`("yolo" | "yolo_lowconf" | "mask" | "ghost") 와 `ghost`(bool) 추가 허용. `.orch/contracts/frame_schema.md` 에 반영. viz.py 는 지금 이 키를 몰라도 그리는 데 문제없어야 한다 (모르는 키 무시).

## 확인
세 영상 `--fps 5 --dump-estimates`. t14 대비: 등급 분포, 위험 이상 프레임 목록과 사유, 겹별로 추가된 객체 수, accident 5.4~6.4s 구간 프레임별 판정. night 8.6s 보행자와 12.2s 프레임 결과. 새로 생긴 위험, 치명이 오탐인지 각각 한 줄.

## 범위
engine.py, frame_schema.md 의 objects 설명, tools/eval_distance.py. 판정 기준(LDW/BSW/FCW/HEAD 레벨 규칙), mvp.py, src/, viz.py, app.py 금지. 박스 없는 깊이 경보와 TTC 는 다음 작업이라 이번엔 넣지 마.
