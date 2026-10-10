# 지시서 t08: 실제 엔진 결과로 시각화 다듬기 (쓰기)

## 배경
엔진(engine.py)이 실제 결과를 냈다: runs_out/night_ride_tmp/ (frames.jsonl 93프레임, 원본 runs_in/night_ride_tmp.mp4).
관리자가 너의 viz.py 로 렌더해서 확인했다 (runs_out/night_ride_tmp/hud.mp4, timeline.png). 검출 박스, 등급 배지, 하단 거리 표시는 잘 나왔다. 고칠 것:
1. 차선 선분이 화면 하단 상태 바 위까지 그려지고, 바이크(핸들, 계기판) 위를 가로지른다. 차선은 도로 영역에만 그려. 상태 바 영역 침범 금지. (차선 좌표 자체가 이상한 프레임은 엔진 쪽에서 따로 고치고 있다. 너는 그리기 쪽만.)
2. 출력 영상이 960x540 으로 줄어든다. 기본은 원본 해상도로, 축소는 옵션으로.
3. 타임라인 PNG 제목이 위쪽에서 잘린다.
4. 실제 결과에서 "차선 측정 불가"와 차선 선분이 함께 보이는 프레임이 있다. lane_measurable=false 일 때 선분을 흐리게 하거나 숨길지 정해서 일관되게.

## 실행 관련
너의 샌드박스는 가상환경 Python 을 실행할 수 없다 (원본 Python 이 허용 경로 밖). 코드는 고치고, 확인은 `tools/check_viz.py` 에 실제 결과 렌더(runs_out/night_ride_tmp → hud.mp4, timeline.png, 그리고 상태별 대표 프레임 4장 그리드 PNG)를 추가해둬. 관리자가 대신 실행해서 결과를 알려준다.

## 범위
engine.py, mvp.py, src/, 노트북, samples/ 수정 금지. 스키마 변경 필요하면 보고로.

## 완료 조건
수정된 viz.py 와 check_viz.py. 보고에 무엇을 어떻게 바꿨는지.
