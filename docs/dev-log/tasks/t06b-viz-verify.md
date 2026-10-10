# 지시서 t06b: 시각화 검증 마무리 (쓰기)

t06 에서 막혔던 환경 문제를 관리자가 처리했어.
- 지정 Python 실행 차단: 샌드박스 쓰기 허용 경로에 `<ENVS>` 를 추가했다. 다시 시도해.
- FFmpeg: `imageio-ffmpeg` 를 설치했다. 경로는 `imageio_ffmpeg.get_ffmpeg_exe()` 로 얻는다 (현재 `<VENV>/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`). 하드코딩하지 말고 함수로 찾게 고쳐.

`tools/check_viz.py` 로 t06 완료 조건(mock HUD 영상 mp4, 타임라인 PNG, app.py 기동 후 종료)을 확인해. 여전히 막히면 무엇이 막혔는지 정확한 에러 한 줄과 함께 보고해.
범위는 t06 과 같다.
