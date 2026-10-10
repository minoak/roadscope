# 보고 20261010-074745-claude-t41_ldw_variants

- 실행자: claude, 모델: claude-opus-5-5, 상태: done, 종료 코드: 0, 소요: 4774.0 초
- 지시서: [tasks/t41_ldw_variants.md](../tasks/t41_ldw_variants.md)

## 바꾼 것
- `engine.py`: `--ldw-variant {V0,VA,VB,VAB}` 옵션 추가(기본 V0). V0가 아닐 때만 meta에 `car.ldw_variant`를 남김
- `geom_car.py`: `lane_zero(relaxed=)` 추가, 완화 상수 `LANE_ZERO_RELAX_*`(15프레임, 5%)
- `tools/rejudge_ldw_vb.py`(신규): 저장된 결과에 VB를 오프라인으로 적용
- `tools/test_ldw_variants.py`(신규): 단위 테스트 15개
- `runs_out/ldw_variants_t41/`(compare.md, verify.json, check/ 6장, runs/, VB/, VAB/)와 `runs_out/t41_work/` 스크립트

## 내가 내린 판단
- VA는 최소 근거 수만 절반으로 낮춤(max(30, 10%) → max(15, 5%)). 보임 비율 완화는 저장 마스크로 시험해 살아나는 클립이 없어서 뺐고, 수평선 실패는 완화하지 않음. 차로 폭이 범위 밖이어도 V0처럼 성공(신뢰도 low)으로 둠
- VB 규칙은 영점 신뢰도가 high가 아니면 끔. 지금 규칙에서는 차로 폭이 2.75~3.75 m 밖인 경우와 같음
- VB, VAB는 오프라인으로 재판정했고, 엔진 실행과 같은 결과임을 3클립에서 확인함. VA는 GPU로 30개를 다시 돌림
- 추천(결정 아님): VB. 다만 정상 영상 L1 56프레임은 어느 방식으로도 줄지 않음

## 확신 없는 부분
- LDW 정답 라벨이 없음. VB가 끈 152프레임 중 90프레임은 그 프레임 차로 폭이 영점 폭의 ±30% 밖이라 오측정으로 추정하고, 62프레임은 맞는 측정으로 봄. 근거는 확인 이미지 6장과 폭 비율뿐
- VA로 살아난 클립은 pos_00845 하나(L1 0). pos_01019는 근거 20/필요 21로 한 프레임 차이로 실패
- "그 프레임 폭이 영점 폭 ±30% 밖이면 그 프레임만 측정 불가"로 하는 프레임 단위 검산은 돌려 보지 않은 가설
- 7861 웹 서버가 수정 전 engine을 메모리에 들고 있는지는 확인하지 않음

## 관리자 판단 필요
- 어느 방식을 쓸지 (민옥 결정)
- 정상 L1 증가분(neg_01211 차로 변경, neg_02117 겹선 밟음)을 주의로 볼지. 어느 방식으로도 줄지 않음
- meta의 `car.ldw_variant` 키를 `.orch/contracts/frame_schema.md`에 반영할지 (지금은 반영 안 함)

## 다른 실행자에게
- `engine.py`, `geom_car.py`의 해시가 바뀜(3029ff1b3c49 / 26eaaf539bc0). 기본 실행 출력은 t40과 같음: 2클립 전 프레임 동일, meta는 시간 값만 다름. 오토바이 mixkit은 기본, VAB 모두 105/105 프레임 동일
- 엔진은 결정적이라, 같은 설정으로 다시 돌리면 frames.jsonl이 같음
- 이 환경에서는 명령 앞에 환경변수를 붙이거나 PowerShell에서 `$env`를 설정하는 것이 막힘. HF_HUB_OFFLINE 등은 subprocess env로 넘겼음
- 단위 테스트: `python -m unittest tools.test_ldw_variants tools.test_eval_nexar web.test_server` (40개 통과)