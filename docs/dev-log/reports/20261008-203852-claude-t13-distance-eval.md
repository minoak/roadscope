# 보고 20261008-203852-claude-t13-distance-eval

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 797.3 초
- 지시서: [tasks/t13-distance-eval.md](../tasks/t13-distance-eval.md)

## 바꾼 것
- `engine.py`: `--dump-estimates` 진단 기록을 추가했다. 객체별 ground/size/depth/fused 값, Depth Pro 변형값, f_px 출처를 `estimates.jsonl`에 남긴다. 판정과 frames.jsonl은 바뀌지 않는다(mixkit, night에서 덤프 전후 바이트 단위로 같음).
- `tools/eval_distance.py` (새 파일): 간접 지표, 의심 사례, 판정 시뮬레이션을 계산한다.
- `runs_out/distance_eval.md`와 세 영상의 `estimates.jsonl`을 만들었다.

## 내가 내린 판단
- 비교 후보를 추가했다: Depth Pro 박스 중앙/하단/하위 25% 값, 영상 안 초점거리 고정, 무효 성분을 빼는 조합 fused_gated(잘린 박스면 size 제외, 기본 VP면 ground 제외).
- 시뮬레이션은 judge의 경고 단계를 도구에 복제했다. fused로 돌리면 실제 결과와 0프레임 차이로 재현된다.
- 사고 6.0s의 차는 이번 실행에서 YOLO가 검출하지 못해, 이전 실행의 박스로 직접 측정했다(`--probe`).

## 확신 없는 부분
- 정답 거리가 없다. 영상마다 한계도 있다.
  - accident는 HUD가 덮인 재인코딩본이다(852×480).
  - night는 사진을 이어 붙인 것이라 시간 지표를 낼 수 없다.
  - mixkit에는 차량이 없다.
  - 가까운 객체는 4개, 사람은 12개뿐이고, night의 사람 박스에는 라이더 신체 조각이 섞였을 수 있다.
- f_px는 세 영상 모두 Depth Pro 초점거리를 쓰고 있다(차선 폭 역산 → W×1.2는 깊이를 안 쓰는 mvp.py에서만 일어난다). 다만 같은 영상 안에서도 초점거리가 0.76~2.24W로 출렁이고, Depth Pro 깊이도 이 값에 비례해 같이 흔들린다.

## 관리자 판단 필요
- 추천: 지금 융합을 유지하되 무효 성분을 빼는 방식(fused_gated)이다. Depth Pro 단독은 추천하지 않는다.
  - 사고 6.0s(실제 1~2m): fused 10.0m, Depth Pro 1.7m, gated 1.7m. 지금 융합은 ground, size가 둘 다 틀릴 때 맞는 Depth Pro 값을 이상치로 버린다.
  - Depth Pro 단독 시뮬레이션(accident): 위험 이상이 3 → 6프레임. 늘어난 것 중 0.0s 치명과 1.2s 위험은 먼 차 전부가 2.7~3.6m로 찍힌 오탐이다(Depth Pro가 장면 전체 거리 축척을 잘못 잡음). 4.4s 위험 → 치명 상향만 타당해 보인다.
  - night 8.6s 사람: 키 1.7m 환산 8.6m, fused 9.4m, Depth Pro 12.5m. 단독으로 쓰면 더 멀게 판정된다.
  - gated는 그 밖에는 fused와 거의 같다(등급 차이 accident 1프레임, night 2프레임). 다만 성분을 빼는 방식이라 드물게 위험 경로가 약해질 가능성이 있다.
- 사고 6.0s의 더 큰 문제는 거리 추정이 아니라 미검출이다(YOLO 신뢰도가 경계선 근처).

## 다른 실행자에게
- `estimates.jsonl`은 `--dump-estimates`를 줄 때만 생긴다. 형식은 `engine.estimates_record`에 있다.