# 지시서 t03: mvp.py 구현 (쓰기)

## 목적
src/RISK_ASSESSOR_V5.ipynb 의 판정 로직을 torch 없이 도는 단일 파일 mvp.py 로 추출해서, samples/ 3장을 끝까지 돌린다.
너는 이전 작업(t01)에서 이 노트북 구조를 이미 파악했어. 그 결론을 따른다: GT 마스크 주입 + depth 없음, 이미지 단위, 세그 모델/트래커 제외.

## 만들 것
1. `mvp.py` (Quest02 루트)
   - 노트북 코드 셀에서 판정에 필요한 함수만 가져온다 (차선/VP, 객체, ego, corridor, 거리, 경고 4종, 보정, bin).
   - torch, transformers, 세그/깊이 모델 로딩은 넣지 않는다.
   - `assess(image_path, mask_path) -> result dict` 와 `result_to_json(result) -> dict` 를 제공. result dict 는 노트북 assess_v5 의 반환 구조를 그대로 유지한다 (`_state` 포함). 시각화 담당이 이 구조를 쓴다.
   - CLI: `python mvp.py samples` 를 실행하면 각 이미지 결과를 `outputs/<이미지이름>.json` 으로 저장하고, `outputs/compare.md` 에 expected_outputs.json 과의 비교표(등급, 경고별 레벨/문구)를 쓴다.
2. 불일치는 고치려 하지 말고 compare.md 에 그대로 기록. Screenshot (322) LDW 불일치는 이미 알려진 것.

## 실행 환경
- Python: `python` (numpy, opencv-python-headless, pillow, matplotlib). 관리자가 지금 설치 중이라 import 가 실패하면 잠시 뒤 다시 해.
- 패키지 추가 설치 금지. 더 필요하면 보고에 올려.

## 범위
- src/ 의 기존 .py, 노트북, samples/ 는 수정 금지. 새로 만드는 건 mvp.py 와 outputs/ 뿐.
- 시각화(PNG/HUD)는 만들지 마. 다른 실행자 담당이다.

## 완료 조건
`python mvp.py samples` 가 에러 없이 끝나고 outputs/ 에 json 3개와 compare.md 가 생긴다.
