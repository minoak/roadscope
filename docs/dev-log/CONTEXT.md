# CONTEXT

관리자가 세션 시작 때 제일 먼저 읽는 파일.

## 프로젝트 방향
블랙박스(야간 오토바이) 영상 위험도 평가의 서비스 MVP. 영상 → 프레임 추출 → 판정(차선 세그 + YOLO 객체 + Depth Pro 거리 + 경고 4종 FCW/LDW/BSW/HEAD) → HUD 영상, 위험도 타임라인, Gradio 화면.
이 작업은 관리자와 실행자를 나눈 작업 구조(Cowork 관리자, Claude/Codex 실행자)의 시험도 겸한다. 민옥은 실행자들이 MVP를 얼마나 잘 만드는지를 본다.

## 재료
- 리포: 이 폴더. 판정 로직 원본은 src/RISK_ASSESSOR_V5.ipynb, 추출본은 mvp.py
- archive: <ARCHIVE> (읽기 전용. 가중치 lane_focus_fold0~4_best.pth 최상위, v6 HUD 결과 영상 review_video/, v6 렌더 스크립트 final_report/_render_video_v6_tracked.py, 데이터셋 이미지 600장)
- Python: <VENV> (리포 밖 가상환경, requirements 버전 고정 + ultralytics, gradio, imageio-ffmpeg)
- 모델 다운로드 위치: <MODELS_DIR>\ (yolo11m.pt). Depth Pro 는 HF 기본 캐시
- GPU: RTX 4070 Laptop 8GB. 사용량 약 7.6GB 로 여유 적음
- 원본 블랙박스 영상 없음. v6 가 참조하던 로컬 사고 영상 폴더는 현재 없음. 테스트는 데이터셋 이미지를 이어 붙인 runs_in/night_ride_tmp.mp4

## 파일 담당
- Claude: engine.py, tools/frames_to_video.py, tools/debug_lanes.py
- Codex: viz.py, app.py, tools/mock_frames.py, tools/check_viz.py
- 엔진-시각화 약속: .orch/contracts/frame_schema.md

## 현재 상태 (2026-10-08 저녁)
- engine.py: 프레임당 1.0초(Depth Pro 0.8초). t07 차선 검증, t10 차선 선택 검증 + 차선 위 주행 LDW L1 + 시간 축 보정 0.8초(--no-temporal-lane) + 사용자 사유 한국어화
- 민옥이 app.py 로 mixkit 영상 실사용 테스트 (테스트 영상: runs_in/mixkit_pov_31974_720.mp4 낮 산길, accident_0029_v6_hud.mp4 HUD 박힌 사고 영상, night_ride_tmp.mp4)
- t11 화면 안정화(H.264 변환, 백그라운드 작업+진행률, mock 분리): Codex 샌드박스 고장으로 두 번 실패 후 t11b 재실행 중
- 민옥 판단 대기: (1) LDW 편향 기준 15% 가 오토바이 정상 주행을 거의 항상 주의로 만듦 (mixkit 105 중 92), (2) BSW sidelobe 승격 거리 상한 (40~59m 차량으로 위험 5프레임), (3) 디버그 키 추가 여부

## 합의한 원칙
- 커밋은 민옥 승인 후. 기존 src/, 노트북, samples/ 는 수정하지 않는다
- archive 는 읽기 전용 참고 자료. 가중치는 복사하지 않고 경로로 참조
- 방향 구분은 당분간 화면 중앙 고정 띠 (DECISIONS 참고)
- Screenshot (322) LDW 불일치: 노트북이 v6 쪽으로 바뀐 흔적. v6 코드를 찾으면 그쪽을 기준으로 검토

## 자동차 블랙박스 기하 실측 (2026-10-09 새벽)
- 보고서: .orch/research/car_dashcam_2026-10-09/car_dashcam_geometry.md (부록, 그림, CSV, 분석 스크립트 포함)
- 재료: Nexar 30영상(기존 10 + 추가 20)의 3초 구간 90개, 오토바이 21구간. 원자료는 로컬 폴더(리포에 없음)
- 실측 요지: 수평선 영상별 0.20~0.69H(0.4H 고정 가정 틀림, 영상 안은 안정), 소실점 가로 위치가 중앙에서 최대 253 px 벗어남, Depth Pro 원값은 기하 거리의 약 0.23배, 크기 변화 TTC는 충돌 4/5에서 0.6~1.4초 전 경보
- 자동차 모드 설계안(보고서 8절)은 제안 상태. 민옥 결정 전에는 엔진에 반영하지 않는다
- 2026-10-09 아침: 민옥이 설계안 8개, TTC 판정 반영, Nexar 30개 평가를 승인(DECISIONS). t20(Claude, engine 자동차 모드 기하 재설계)과 t21(Codex, Nexar 30개 준비 + eval_nexar 확장 + 앱 차량 선택) 병렬 실행 중. 그다음은 t20 결과로 30개 실행, Codex 평가, 교차 리뷰
- 정리용 문서(Claude Docs): (내부 문서 링크, 비공개)
- 2026-10-09 오후: t24(미탐 버그 4개 수정) 30개 평가 t25 = 미탐 2/20, 사고 밖 위험 이상 25.5%, 정상 L1 38%. t26은 실내 요청으로 중단, t27은 13:10 실행 환경이 재시작되며 중단됨. t28(Codex) 모델 시험: YOLOPv2 차선 소실점 19/30, Mask2Former 내 차 영역 30/30. 지금 t29(Claude)가 t27 마무리 + YOLOPv2, Mask2Former, YOLO 반복 검출 통합 + 30개 실행 중

## 정리 (2026-10-09 저녁)
### 지금 자동차 모드 구조
- 사전 점검(측정 준비): 소실점과 수평선(광류 + YOLOPv2 차선 + 차량 박스 회귀), 카메라 높이(차량 박스), 초점거리 600 px 기본, 내 차 영역(Mask2Former + YOLO 반복 검출 + 정지 영역), 차로 안 평소 위치(LDW 기준만, t32)
- 판정(한 장면 기준): 거리 = 바닥 위치와 차폭 중 가까운 쪽(차폭 1.6~2.5 m 게이트), 차로 = 소실점 열 ±2.5 m, 침범 |X| < 1.25 m, FCW 3/5/10 m, TTC는 기록만, LDW 시간 보정 없음, 유령 박스 유지, Depth Pro는 박스 없는 근거리 보험
- 오토바이 모드는 lane_focus 그대로
### 평가 (t29, 이상치 3개 제외)
- 미탐 2/18(pos_00004 끼어들기, pos_00877 정면 5.6 m), 이상치: pos_00905, neg_01960, pos_00754(보정 실패)
- 남은 문제: 정상 영상 주의 약 41%(차선 침범, 측면 나란히, 보류 중인 고질병), 정체 중 앞차 위험(민옥이 기준 유지로 결정)
### 웹 MVP
- web/server.py(FastAPI) + web/static(시안 기반), run_web.bat(7861). 기존 run_app.bat(Gradio 7860)은 그대로
- GPU 없이 확인: 개발 모드 서버 + /dev/load?run=nexar_t29/<영상>. 데스크톱, 휴대폰 폭, 신뢰 낮음 띠 관리자 확인 완료
### 남은 일
1. GPU 허용 시: Nexar 30개 재실행(t32 LDW 효과 확인) + 웹 실제 업로드→엔진→HUD 흐름 확인
2. 경고 목록에서 0.5초 미만 깜빡임 묶어 보이기(화면만)
3. run_web_dev.bat(개발 모드 실행 파일) 여부, 영상 보관 기간 정책
4. 커밋: Quest02 새 파일 전부, 민옥 승인 대기
5. 정리: web/tmp* 7개
6. 다음 축: 새 저장소로 독립(30개 재실행 후), 민옥 직접 라벨링으로 평가 축 만들기
- 2026-10-09 19:07 순차 실행 시작: `.orch/overnight_chain.ps1`(로그 `.orch/overnight_chain.log`). 순서: t34(개발 30개 재실행, 진행 중) → t36(내 차 증류 학습) → t37(시험 세트 99개 실행, 코드 변경 금지) → t38(개발, 시험 평가, 95% 신뢰구간, README 표) → t39(웹 실제 업로드 흐름 확인). t35(웹 깜빡임 묶기, run_web_dev.bat) 완료
- 2026-10-09 19:40 민옥 요청으로 순차 실행과 t36 중지(GPU 작업은 사용자가 정한 시간에만). t34(개발 30개 재실행) 완료: LDW 정상 영상 34→21(좌 26→1, 우 8→20). t36은 라벨 만들기 초반에 멈춤(runs_out/distill에 일부 로그만)
- 민옥 관찰: 자동차 블랙박스는 대시보드, 보닛이 화면 아래를 넓게 차지해 근거리 차선이 안 보이는데, LDW 판정 기준 행이 0.95H(오토바이 기준 그대로, geom_car LANE_BASE_ROW, engine y_check)라 대부분 가려진 영역에서 외삽된 차선으로 판정 → 거의 모든 영상에서 차선 경고가 뜸. 소실점처럼 영상 초반에 차선 영점(보이는 가장 아래 행, 그 행의 차로 폭 px, 평소 위치)을 잡아야 한다는 제안. LDW는 민옥이 직접 몇 번 더 테스트하고 정하기로 함
- 2026-10-10 00:11 순차 실행 2 시작(`.orch/overnight_chain2.ps1`, 로그 `.orch/overnight_chain.log`): t40(차선 영점 + 개발 30개 재실행) → t36(내 차 증류, t40 세션 이어받음) → t39(웹 실제 업로드 확인). 시험 세트(t37, t38)는 LDW 확정 뒤로 보류

- 2026-10-10 03:19 순차 실행 완료(t40 차선 영점, t36 증류 IoU 0.862, t39 웹 e2e 통과). 정리 문서: (내부 문서 링크, 비공개) . 민옥 결정 대기: LDW 영점 실패 기준, 증류 모델 엔진 투입. 커밋 없음.

- 2026-10-10 새 리포 생성(민옥): https://github.com/minoak/roadscope (public). 첫 커밋 cfb0d51에 README.md(11바이트)만 있음 → 새 이력은 이 커밋을 clone해서 그 위에 쌓는다. 푸시는 올릴 목록, 라이선스 확정 후 민옥 승인.

