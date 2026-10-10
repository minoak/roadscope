# t50: 새 리포 준비: 조사와 초안만 (write, .orch/repo_prep/ 에만 쓰기)

## 배경
Quest02(블랙박스 위험도 평가, 웹 이름 RoadScope)를 새 GitHub 리포로 옮긴다. 민옥 결정(DECISIONS.md 2026-10-10): 새 이력으로 시작, 기존 AIFFEL_quest_eng Quest02(시작 커밋 b6aad74)와는 링크로만 연결. 테스트셋 숫자(t45/t46)는 아직 없다. 지금은 옮기기 전 조사와 초안만 한다. 지금 Claude가 engine.py, geom_car.py를 수정 중(t48)이니 그 파일들은 읽기만 하고, 다른 어떤 파일도 고치지 마라.

## 할 일 (결과는 전부 .orch/repo_prep/ 아래에)
1. inventory.md: 프로젝트 파일을 "올림 / 안 올림 / 결정 필요"로 분류한 표(경로, 크기, 이유). 기준:
   - 안 올림: 모델 가중치(.pt, .pth, .onnx, HF 캐시), 영상(runs_in/ 영상, 결과 mp4), runs_out 대부분, web/tmp*, __pycache__, 개인 경로가 박힌 파일.
   - 결정 필요: .orch/(지시서, 결정, 보고: 포트폴리오 가치 vs 노출), runs_out 중 README에 쓸 캡처, 표, 증류 가중치 ego_unet_r34.pt.
   - 100 MB 넘는 파일, 하드코딩된 절대 경로(<HOME> 등), 토큰, 키로 보이는 문자열을 따로 목록으로(값은 적지 말고 파일:줄만).
2. licenses.md: 쓰는 모델, 라이브러리, 데이터별 라이선스 표: YOLO11(ultralytics), YOLOPv2, Mask2Former 코드와 가중치(Mapillary Vistas 학습), Depth Pro, segmentation_models_pytorch, Nexar 데이터셋, 그 밖 requirements. 로컬에 있는 LICENSE 파일, 패키지 메타데이터, 모델 카드로 확인하고 출처 경로를 적어라. 확인 못 한 건 "미확인"으로. 리포 라이선스 선택지(예: AGPL-3.0)와 각 조건이 리포에 주는 의무를 짧게.
3. structure.md: 새 리포 폴더 구조 제안(엔진, 웹, tools, docs, 설치 스크립트), 옮길 때 바꿔야 할 경로, 설정 목록, requirements.txt 초안(실제 쓰는 패키지와 버전, 현재 환경 <VENV> 기준).
4. README_draft.md: 한국어 README 초안. 구성: 무엇을 하나(한 문단), 화면 캡처 자리, 판정 원칙(단일 장면, 위험, 치명은 놓치지 않게, 주의는 과하지 않게), 구조(보정 → 검출 → 기하 거리 → 판정), 쓰는 모델과 역할, 평가(개발 30 / 시험 99 분리, 시험은 한 번만: 숫자 자리는 [t46 결과]로 비워둠), 한계, 실행 방법, 출발점(기존 리포 링크 자리와 커밋 b6aad74), 개발 방식(Cowork 관리 + Claude Code, Codex 실행, .orch 공개 여부는 민옥 결정: 두 버전 문단 모두 써둘 것). 문체: 담백하게. 줄표(—) 남발, "이것이 ~의 핵심", "~인 셈", 가운뎃점 나열, 문학적 도입, 수사적 마무리 금지. 과장 금지, 확인 안 된 숫자 금지.

## 금지
.orch/repo_prep/ 밖 파일 쓰기, 수정 금지. git 명령으로 상태를 바꾸지 마라(조회만). 원격 리포 생성, 푸시 금지. GPU 금지. runs_in/testset 금지. 네트워크로 받아오지 마라(로컬 확인만).

## 보고
결정 필요 항목을 따로 모아서. 라이선스상 문제될 수 있는 것을 맨 위에.
