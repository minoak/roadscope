# 보고 20261008-192309-codex-t11b-app-robust-retry

- 실행자: codex, 모델: gpt-6-astra, 상태: done, 종료 코드: 0, 소요: 728.8 초
- 지시서: [tasks/t11b-app-robust-retry.md](../tasks/t11b-app-robust-retry.md)

## 바꾼 것

- `app.py`: 파일 업로드, 백그라운드 분석, 2초 간격 진행률 조회, 작업 ID로 결과 복원, MOCK 전용 탭 구현.
- `viz.py`: FFmpeg 자동 탐색과 H.264/yuv420p 변환 추가. 호환 H.264는 재인코딩 생략, 실패 이유는 한국어로 표시.
- `tools/check_viz.py`: FMP4 변환, 비동기 작업, ID 재조회, MOCK 분리, 서버 종료 검증 추가.

## 내가 내린 판단

GPU 작업은 단일 큐에서 순차 실행하도록 정했사와요. 완료 결과는 디스크에 보관하고, 반복 조회 시 영상을 다시 보내지 않아 재생 위치를 유지하도록 했답니다.

## 확신 없는 부분

세 파일의 문법 검사는 통과했지만 실제 변환, Gradio 기동은 미검증이랍니다. 추가 검증 중 임시 폴더 쓰기가 차단되어 중단했사와요.

`PermissionError: [Errno 13] Permission denied: 'runs_out\\t11-stdlib-5aps4qnf\\source.mp4'`

## 관리자 판단 필요

없음

## 다른 실행자에게

관리자 환경의 지정 Python으로 `tools/check_viz.py --app-only`를 실행하면 된답니다. 통과 보고서는 `runs_out/verification_t11.json`에 저장되며 확인 서버는 종료되도록 구성했사와요.

브라우저 재접속은 지원하지만, 서버 재시작 당시 진행 중이던 작업은 자동 재개하지 않는답니다. 오류가 난 임시 폴더는 정리도 권한 오류로 실패했사와요.