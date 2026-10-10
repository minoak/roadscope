# 지시서 t14: 거리 추정 개선 적용 (쓰기)

`.orch/DECISIONS.md` 2026-10-08 마지막 두 줄이 이번 결정이다. 너의 t13 실험(`runs_out/distance_eval.md`)을 바탕으로 적용한다.

## 할 것
1. 융합을 fused_gated 로 교체: 박스가 화면 끝에 잘렸으면 size 제외, 기본 VP 를 쓰면 ground 제외. 남은 성분이 하나면 그 값. 제외 사유는 estimates 기록에 남긴다.
2. Depth Pro 초점거리 영상 단위 고정 (두 번 훑기):
   - 1회차: 지금처럼 프레임마다 모델을 돌리고, 판정에 필요한 원재료(객체 박스, 성분별 거리, 그 프레임 Depth Pro 초점거리, 차선/VP 등)를 메모리나 임시 파일에 저장.
   - 고정값 = 구간 내 초점거리 중앙값. depth 성분은 depth × (f_fixed / f_frame) 로 보정. ground/size 가 쓰는 f_px 도 f_fixed 로.
   - 2회차: 모델 재실행 없이 보정된 값으로 판정(트래커 포함)을 다시 돌려 frames.jsonl 을 쓴다.
   - 장면 전환 감지 시 구간을 나눠 구간별로 고정값을 잡는다. 감지 방법은 네가 정하고 근거를 보고 (예: 프레임 간 히스토그램 차이). night_ride_tmp 처럼 사진을 이어 붙인 영상에서 동작이 합리적인지 확인.
   - `--no-focal-lock` 로 끌 수 있게. meta.json 에 구간별 f_fixed 와 구간 수를 기록.
3. app.py 가 쓰는 인터페이스(`python engine.py <영상> --fps N`, 출력 파일 위치, 진행률 로그 형식)는 바꾸지 마. 2회차 동안에도 진행률 로그가 이어지게. `assess_frame(bgr)` 단일 프레임 함수는 고정값이 없으므로 기존 방식 유지하되 docstring 에 명시.

## 확인
세 영상을 `--fps 5 --dump-estimates` 로 돌려 t13 기준과 비교: 등급 분포, 위험 이상 프레임 목록, 의심 사례 두 개(사고 6.0초 차, night 8.6초 사람)의 최종 거리, 시간 안정성 지표, 처리 시간 증가분. tools/eval_distance.py 를 재사용해.
위험, 치명 프레임이 줄어든 경우 각각 왜 줄었는지 한 줄씩 (미탐 우선 원칙).

## 범위
engine.py, tools/eval_distance.py. 판정 규칙(LDW/BSW/FCW/HEAD 기준), 스키마, mvp.py, src/, viz.py, app.py 금지.
