# 보고 20261010-152058-claude-t45-testset-run

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 16584.6 초
- 지시서: [tasks/t45-testset-run.md](../tasks/t45-testset-run.md)

## 바꾼 것
- 코드 수정 없음. engine.py, geom_car.py, car_models.py는 실행 전후 해시가 같음.
- 새로 만든 것:
  - runs_out/testset_t45/: 99개 결과 폴더와 run_log.md
  - runs_out/t45_work/: run_batch.py(실행기), wait.py(진행 확인용), batch_stdout_1~3.txt, videos/(영상 99개 하드링크)

## 내가 내린 판단
- 엔진은 출력 폴더 이름을 영상 파일 이름으로 정함. HF 캐시 영상 이름은 `00971.mp4` 같은 숫자뿐이라, `t45_work/videos/pos_00971.mp4`로 하드링크를 걸어 그 경로로 돌림. 디스크는 더 쓰지 않음.
- 백그라운드 실행 한 번에 2시간 상한이 있어서 95분씩 3묶음으로 나눠 돌림. 끝난 영상은 건너뛰고, 묶음마다 시작과 끝에 해시를 다시 확인함(전부 같음).
- 실패 기준은 "종료 코드가 0이 아니거나 meta.json이 없음"으로 정함. 보정 실패는 `horizon_ok=False`로 기록함.
- 해시는 SHA256 전체를 적음. engine.py가 불러 쓰는 mvp.py와 src/simple_tracker.py도 참고로 적음.

## 확신 없는 부분
- 결과:
  - 99/99 성공, 99개 모두 종료 코드 0, 재시도 0회
  - 엔진 로그에 Traceback, 메모리 부족(OOM) 없음
  - 실행 시간: 합계 272.8분, 영상당 평균 165초(최소 44초, 최대 290초)
  - 프레임 수: 180~600
- 보정 결과는 run_log.md에 영상별로 적기만 했고, 오염을 막으려고 내용은 보지 않음.

## 관리자 판단 필요
없음

## 다른 실행자에게
- 해시(SHA256 앞 12자리)는 engine.py 67d108a238dc, geom_car.py ae9bd055612f, car_models.py 95a075032333. t48 수정 후 값과 같음.
- 각 결과의 meta.json `source`에는 HF 캐시 경로가 아니라 `runs_out/t45_work/videos/<file>` 하드링크 경로가 적혀 있음. 원본 경로는 manifest.csv에서 찾으면 됨.
- run_log.md의 보정 칸은 구간마다 `;`로 나뉨. 열은 수평선 출처/신뢰도, 보정 실패 여부, 차선 신뢰(trusted, exclude_lanes, reason) 순서.
- t45_work/videos/에는 영상 1.6GB가 하드링크로 들어 있음. 리포에 올릴 때 빼야 함.
- 커밋 안 함.