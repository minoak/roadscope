# 지시서 t13: 거리 추정 방식 비교 실험 (쓰기, 판정 로직 변경 금지)

## 배경
민옥 결정: Depth Pro 를 거리 주력으로 바꾸는 것을 검토한다. 바꾸기 전에 기존 방식(ground / size / depth 융합)과 Depth Pro 단독이 얼마나 다르고, 어느 쪽이 더 믿을 만한지 먼저 실험으로 확인한다.
지금까지 관찰된 의심 사례:
- accident 영상 6.0초: 화면 26% 를 덮는 차가 8.8m (실제로는 1~2m 로 보임)
- night 영상 8.6초(frame 86, night ride (51).png): 도로를 건너는 사람 9.4m
- f_px 차선 폭 역산이 자주 실패해 W×1.2 기본값이 쓰임 (Depth Pro 가 내는 초점거리를 실제로 쓰고 있는지도 확인)

## 할 것
1. engine.py 에 진단 기록만 추가: 객체마다 ground / size / depth / fused 거리와 사용된 f_px 출처를 `runs_out/<영상>/estimates.jsonl` 에 따로 남긴다. frames.jsonl 과 판정 결과는 바꾸지 마. 플래그(`--dump-estimates`)로 켜고 끄게.
2. `tools/eval_distance.py`: estimates.jsonl 을 읽어 방식별로 비교한다. 정답 거리가 없으니 아래 간접 지표로:
   - 커버리지: 방식별로 값이 나온 객체 비율
   - 방식 간 일치도: depth 대비 ground, size 의 비율 분포 (중앙값, 사분위)
   - 상식 위반: 박스 면적 15% 이상이거나 박스 하단이 화면 하단 5% 안인데 5m 넘게 나온 경우의 수. 사람 박스 높이와 키 1.7m 로 계산한 거리 대비 크게 벗어난 경우
   - 시간 안정성 (연속 영상만, accident): 같은 track_id 의 연속 평가 사이 상대 변화 |Δd|/d 의 분포. 낮을수록 안정
   - 위 의심 사례 두 개의 방식별 값
3. 세 영상 모두 `--fps 5 --dump-estimates` 로 돌려서 결과를 `runs_out/distance_eval.md` 에 표로 정리.
4. Depth Pro 단독으로 바꾸면 판정(등급 분포, FCW/BSW 레벨)이 어떻게 달라지는지는 코드를 바꾸지 말고, estimates 의 depth 값으로 다시 계산한 시뮬레이션 수치만 보고해 (가능한 범위에서).

## 보고
결론은 "Depth Pro 단독 / 지금 융합 / 다른 조합" 중 무엇을 추천하는지와 근거 수치. 정답이 없어서 생기는 한계도 짧게.

## 범위
engine.py 는 진단 기록 추가만. 판정, 스키마, mvp.py, src/, viz.py, app.py 금지. 게시판의 원칙 공지(위험, 치명은 미탐 우선)를 고려해 결론을 내.
