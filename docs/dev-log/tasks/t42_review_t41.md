# t42: t41(LDW 방식 비교) 리뷰 (read)

Claude가 t41에서 engine.py에 `--ldw-variant {V0,VA,VB,VAB}`, geom_car.py에 `lane_zero(relaxed=)`, tools/rejudge_ldw_vb.py, tools/test_ldw_variants.py를 추가했고 결과는 runs_out/ldw_variants_t41/compare.md에 있다.

## 확인할 것
1. 기본값(V0)이 t40과 완전히 같은가. 옵션 없이 실행했을 때 코드 경로가 t40과 다르게 갈 수 있는 곳이 있는지 코드로 확인. compare.md의 V0_rerun 2클립 비교가 맞게 계산됐는지(runs_out/ldw_variants_t41/verify.json, runs/ 하위).
2. LDW 외 경고(FCW, BSW, HEAD 등)가 모든 방식에서 V0와 같은가. 변형이 LDW 외 경로를 건드리는 코드가 있는지.
3. tools/rejudge_ldw_vb.py의 오프라인 재판정이 엔진 `--ldw-variant VB`와 논리적으로 같은가.
4. compare.md 요약표 숫자 몇 개(정상 LDW L1 56, 사고 L1 165→13, VB가 끈 152프레임)를 저장 결과에서 직접 다시 세서 맞는지.
5. runs_in/testset을 읽거나 쓴 흔적이 없는지(코드와 runs_out 하위).
6. 단위 테스트 실행: `python -m unittest tools.test_ldw_variants tools.test_eval_nexar web.test_server`.

## 금지
파일 수정 금지(read). GPU 실행 금지. testset 사용 금지.

## 보고
문제를 심각도 순으로. 숫자 재검산 결과는 "맞음/다름(값)"으로. 문제 없으면 없다고.
