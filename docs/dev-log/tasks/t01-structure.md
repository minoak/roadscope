# 지시서 t01: 구조와 흐름 파악 (읽기)

## 목적
이 폴더는 블랙박스 영상으로 주행 위험도를 평가하는 프로젝트야. 나중에 src/ 의 .py 파일들을 조합해서 MVP를 만들 생각이라, 지금은 무엇이 있고 어떻게 이어지는지만 파악해줘.

## 볼 것
- README.md, samples/README.md, train/README.md, requirements.txt
- src/*.py 전부 (ego_corridor, geometric_distance, hud_v5_clean, simple_tracker)
- src/RISK_ASSESSOR_V5.ipynb 는 모듈을 어떻게 엮었는지 확인하는 용도로만. 출력 셀은 무시하고 코드 셀만 봐.
- 루트의 큰 .ipynb (2~3MB) 는 열지 마.

## 알려줄 것 (보고 안에 정리)
1. 각 .py 모듈의 역할, 입력과 출력, 서로 무엇을 import 하는지
2. 노트북에서 모듈들이 연결되는 순서 (입력 영상/이미지 → ... → 위험도)
3. MVP로 엮을 때 노트북에만 있고 .py 에는 없는 로직이 있다면 그게 뭔지
4. 네가 보기에 가장 단순한 MVP 구성 (어떤 모듈을 어떤 순서로)

## 범위
파일 수정 금지. 판단은 보고의 "관리자 판단 필요"로 올려.
