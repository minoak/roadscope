# 지시서 t09: 차선 검증 변경 리뷰 (읽기)

다른 실행자(Claude)가 t07(`.orch/tasks/t07-lane-sanity.md`)에서 engine.py 에 차선 검증을 넣었다. 게시판 새 글과 engine.py 의 `find_lanes`, `lane_input_mask` 및 호출부를 보고 리뷰해줘.
engine.py 는 git 미추적 새 파일이라 diff 가 없다. 수정 전 상태는 `.orch/runs/*claude-t05*/` 기록과 t07 보고를 참고해.

볼 것:
1. 검증 규칙 3가지(ego 영역 lane 픽셀 제거, |dx/dy| ≤ 3.0, VP conf ≥ 0.3)가 의도대로 구현됐는지, 놓친 경계 조건
2. 검증 실패 시 lanes/corridor/vp 를 모두 null 로 만드는 처리가 BSW/FCW 등 다른 경고에 주는 부작용
3. `runs_out/debug/lanes_frame*.png` 를 볼 수 있으면 보고, 규칙이 실제로 바이크 영역 선분과 가로 노면표시를 걸러냈는지
4. 프레임 134 BSW L2(29.4m 차량이 sidelobe 안이라 위험) 문제의 원인이 t07 보고 설명과 맞는지

수정 금지. 1500자 이내.
