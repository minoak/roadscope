# 지시서 t04: 시각화 설계 (읽기)

## 목적
위험도 평가 MVP 의 시각화를 네가 맡는다. 지금 다른 실행자가 판정 로직을 mvp.py 로 추출하고 있어 (노트북 assess_v5 의 result dict 구조를 그대로 유지, `_state` 포함). 그게 끝나면 너가 viz.py 를 구현할 거라, 지금은 설계만 해줘.

## 볼 것
- src/RISK_ASSESSOR_V5.ipynb 의 draw_hud_v5 등 시각화 코드, src/hud_v5_clean.py
- 참고 자료: <ARCHIVE> (이전 실험 폴더. 시각화 스크립트, 결과 이미지, mp4 가 있을 수 있음). 가중치나 큰 데이터는 열지 마.
- samples/images 3장

## 알려줄 것 (보고 1500자 이내)
1. 기존 HUD 가 화면에 무엇을 보여주는지, result dict 의 어떤 키를 쓰는지
2. hud_v5_clean.py 를 그대로 쓸 수 있는지, 막히는 점 (예: `from __main__ import draw_zones_on_ax`)
3. MVP 시각화 제안: 이미지 1장당 HUD PNG + 3장 갤러리. 무엇을 그리고 무엇을 뺄지
4. archive 에서 쓸 만한 시각화 자산이 있으면 경로

## 범위
수정 금지. 판단은 "관리자 판단 필요"로.
