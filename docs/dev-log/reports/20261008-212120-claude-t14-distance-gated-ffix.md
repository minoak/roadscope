# 보고 20261008-212120-claude-t14-distance-gated-ffix

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 1191.4 초
- 지시서: [tasks/t14-distance-gated-ffix.md](../tasks/t14-distance-gated-ffix.md)

## 바꾼 것
- `engine.py`:
  - 거리 융합을 fused_gated로 바꿨다. 잘린 박스면 size를, 기본 VP면 ground를 빼고, 뺀 사유는 estimates의 `gated`에 남긴다.
  - judge를 judge_scene(초점과 무관한 부분)과 judge_with_focal(거리, 경고)로 나눴다. 두 기능을 끄면 t13과 레코드 차이 0건이다.
  - 두 번 훑기로 장면 구간별 초점거리를 고정한다. `--no-focal-lock`으로 끌 수 있고, meta.json의 `focal_lock`에 구간과 f_fixed를 기록한다.
- `tools/eval_distance.py`: t13 대비 비교, 등급이 내려간 이유 자동 기록, 미검출 박스의 최종값 계산을 추가했다.
- `runs_out/distance_eval_t14.md`를 만들고 세 영상 결과를 갱신했다. t13 결과는 `runs_out/debug/t14/t13/`에 백업했다.

## 내가 내린 판단
- 장면 전환 감지: 연속 원본 프레임의 64×36 회색 썸네일 정규화 상관이 0.5 미만이면 전환으로 본다. 히스토그램은 연속 영상과 사진 교체의 분포가 겹쳐서 쓸 수 없었다.
- 감지 결과: mixkit 1구간, night 86구간(비슷한 사진 2~3장이 6번 묶임), accident 11구간이다. accident는 사고 순간 카메라가 뒤집히며 쪼개졌고, 평온한 19프레임 구간은 하나로 묶였다. 잘못 합치는 것보다 잘못 쪼개는 쪽이 안전하다고 봤다.
- 1회차 결과를 frames.jsonl에 바로 쓴다(app.py가 이 줄 수로 진행률을 셈). 2회차는 임시 파일에 쓴 뒤 교체하고, Windows 파일 잠금에 대비해 교체를 재시도한다.

## 확신 없는 부분
- 등급 분포 t13 → 지금: accident 7/26/4/0(t13 7/27/3/0), night 46/45/1/1(t13 45/46/1/1), mixkit 동일.
- 위험 이상 프레임: accident는 4.4, 4.6, 5.0s에 0.0s가 추가됐다(잘린 왼쪽 차가 3.9 → 2.9m가 되며 BSW L2). night는 9.4, 16.8s로 같다. 위험, 치명이 줄어든 프레임은 없다.
- 의심 사례 최종 거리:
  - 사고 6.0s 차: 1.2m(t13 융합 10.0m). 다만 이번에도 YOLO가 검출하지 못해 판정에는 반영되지 않는다.
  - night 8.6s 사람: 9.4m로 변화 없음.
- 시간 안정성(accident, 중앙/75%/90%): 0.41/0.59/0.94 → 0.28/0.44/0.80
- 처리 시간 증가: 2회차는 영상당 0.01~0.05초이고, 1회차 썸네일 비교는 무시할 수준이다.

## 관리자 판단 필요
- night 12.2s가 주의 → 안전이 됐다. 잘린 박스라 size(4.6m)를 빼자, 원래 이상치로 버려지던 ground 18.5m가 살아남아 거리가 4.1 → 8.7m로 멀어졌다. 성분이 두 개만 남으면 이상치를 걸러내지 못한다. 이번엔 L1이지만 같은 구조로 위험, 치명도 약해질 수 있다. 남은 두 성분이 2배 이상 차이 나면 가까운 쪽을 쓰는 규칙을 넣을지 정해줘.

## 다른 실행자에게
- meta.json에 `focal_lock`이 추가됐다(enabled, n_segments, segments[start_frame, end_frame, n_eval, f_fixed], pass2_sec).
- frames.jsonl은 1회차 값으로 먼저 쓰이고, 영상 처리가 끝날 때 2회차 값으로 통째로 교체된다. 진행 중에 읽은 값은 최종값이 아닐 수 있다.
- `assess_frame(bgr)`은 초점 고정 없이 그 프레임의 초점거리를 쓴다.