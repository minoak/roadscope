# 개발 기록 (정리본)

프로젝트를 진행하며 남긴 작업 관리 기록을 공개할 수 있도록 정리했다.

민옥이 목표와 판정 원칙을 정했다. 관리자(Cowork)는 이를 바탕으로 작업 지시서를 작성해 실행자(Claude Code, Codex)에게 전달했다. 실행자는 작업이 끝나면 정해진 형식의 보고를 남겼다.

| 파일 | 내용 |
|---|---|
| [DECISIONS.md](DECISIONS.md) | 민옥이 내린 결정과 이유 |
| [CONTEXT.md](CONTEXT.md) | 관리자의 진행 메모 |
| [tasks/](tasks/) | 작업 지시서 55개 |
| [reports/](reports/) | 실행 보고 48개 (실행 하나당 파일 하나) |

## 읽을 때 주의

- 경로와 시각, 수치는 당시 기록을 그대로 남겼다. 수치는 각 시점의 엔진으로 얻은 결과다. 최종 평가 수치는 리포 README의 표를 기준으로 한다.
- 당시 파일과 이 리포의 파일 위치는 아래 표로 대조할 수 있다.

| 당시 위치 | 이 리포 |
|---|---|
| `engine.py`, `geom_car.py`, `car_models.py`, `viz.py` | `src/roadscope/` 같은 이름 |
| `mvp.py` | `src/roadscope/rules.py` |
| `src/simple_tracker.py` | `src/roadscope/tracker.py` |
| `tools/eval_nexar.py`, `tools/eval_distance.py`, `tools/rejudge_ldw_vb.py` | `tools/evaluate/` |
| `tools/get_nexar_*.py`, `tools/add_nexar_from_cache.py` | `tools/data/` |
| `tools/frames_to_video.py`, `tools/debug_lanes.py` | `tools/render/` |
| `tools/test_*.py`, `web/test_server.py` | `tests/` |
| `.orch/contracts/frame_schema.md` | `docs/frame_schema.md` |
| `app.py`(Gradio), 노트북, `tools/check_viz.py`, `tools/mock_frames.py`, `tools/probe_car_models.py`, `tools/rejudge_lane_baseline.py`, `web/verify_*` | 이 리포에 없음 |

## 정리하며 바꾼 것

이 정리본에는 t52b까지의 기록을 담았다.

- 개인 절대 경로는 다음 자리표시로 바꿨다. `<ARCHIVE>`(과거 실험 폴더, lane_focus 가중치 위치), `<MODELS_DIR>`(모델 가중치 폴더), `<ENVS>`(가상환경과 모델을 둔 상위 폴더), `<VENV>`(가상환경), `<HF_CACHE>`(Hugging Face 캐시), `<시스템 Python>`(가상환경 밖 Python 실행 파일), `<HOME>`(사용자 폴더)이다. 프로젝트 폴더 경로는 지워서 상대 경로로 남겼다.
- 로컬 파일은 링크를 없애고 이름만 남겼다. 비공개 문서 링크는 지웠다.
- CONTEXT.md에서 내부 메모 한 줄, 작업 관리 도구 자체의 운영 메모(실행기 설정, 게시판 문제, 도구 코드의 커밋 상태)를 뺐다.
- 개인 일정이 담긴 표현을 지웠다. GPU 사용 시간 규칙은 "사용자가 정한 시간에만"으로 바꿨다. 결정의 기술 내용은 그대로 두었다.
- 문장 부호를 정리하면서 나열에 쓴 가운뎃점은 쉼표로, 줄표는 쌍점이나 쉼표로 바꿨다. 수식의 곱셈 점과 코드 표기 안의 문자는 그대로 두었다. 뜻은 바꾸지 않았다.
- DECISIONS.md 머리말의 호칭을 정리했다.
- 위 항목 외의 내용은 고치지 않았다.

실행 프롬프트 사본과 표준 출력 로그, 실행 메타데이터(세션 ID, 토큰 사용량 등)는 넣지 않았다. 실행자 간 게시판과 화면 디자인 시안, 자동차 블랙박스 실측 원자료도 제외했다.

시험 세트 관련 지시서(t37, t45, t46)도 남겼다. 최종 엔진으로 시험 세트를 한 번만 실행하고 그 결과로 규칙을 고치지 않는다는 절차를 확인할 수 있다.

일부 작업에는 보고가 없다. t22, t26, t27은 실행이 중단됐다. t37, t38은 실행하지 않고 t45, t46으로 다시 지시했다.

t51은 실행 기록에 보고 파일이 없다. t52, t52b는 이 정리본을 만드는 작업이라 보고가 아직 없다.

Codex 보고에 쓰인 어미(~사와요)는 당시 실행자에게 설정했던 말투다.

## 작업 목록

| 지시서 | 제목 | 보고 |
|---|---|---|
| [t01-structure](tasks/t01-structure.md) | 지시서 t01: 구조와 흐름 파악 (읽기) | [20261007-231230](reports/20261007-231230-claude-t01-structure.md) |
| [t02-runnable](tasks/t02-runnable.md) | 지시서 t02: 실행 가능성 점검 (읽기, 명령 실행 가능) | [20261007-231230](reports/20261007-231230-codex-t02-runnable.md) |
| [t03-mvp](tasks/t03-mvp.md) | 지시서 t03: mvp.py 구현 (쓰기) | [20261007-232415](reports/20261007-232415-claude-t03-mvp.md) |
| [t04-viz-plan](tasks/t04-viz-plan.md) | 지시서 t04: 시각화 설계 (읽기) | [20261007-232415](reports/20261007-232415-codex-t04-viz-plan.md) |
| [t05-engine](tasks/t05-engine.md) | 지시서 t05: 영상 엔진 구현 (쓰기) | [20261007-233703](reports/20261007-233703-claude-t05-engine.md) |
| [t06-viz](tasks/t06-viz.md) | 지시서 t06: 시각화와 화면 구현 (쓰기) | [20261007-233703](reports/20261007-233703-codex-t06-viz.md) |
| [t06b-viz-verify](tasks/t06b-viz-verify.md) | 지시서 t06b: 시각화 검증 마무리 (쓰기) | [20261007-234646](reports/20261007-234646-codex-t06b-viz-verify.md) |
| [t07-lane-sanity](tasks/t07-lane-sanity.md) | 지시서 t07: 차선 검출 검증 + 정리 (쓰기) | [20261007-235724](reports/20261007-235724-claude-t07-lane-sanity.md) |
| [t08-viz-polish](tasks/t08-viz-polish.md) | 지시서 t08: 실제 엔진 결과로 시각화 다듬기 (쓰기) | [20261007-235725](reports/20261007-235725-codex-t08-viz-polish.md) |
| [t09-review-lanes](tasks/t09-review-lanes.md) | 지시서 t09: 차선 검증 변경 리뷰 (읽기) | [20261008-004445](reports/20261008-004445-codex-t09-review-lanes.md) |
| [t10-lane-straddle](tasks/t10-lane-straddle.md) | 지시서 t10: 차선 밟음 감지 + 시간 축 보정 (쓰기) | [20261008-183845](reports/20261008-183845-claude-t10-lane-straddle.md) |
| [t11-app-robust](tasks/t11-app-robust.md) | 지시서 t11: 화면 안정화 (쓰기) | [20261008-183845](reports/20261008-183845-codex-t11-app-robust.md) |
| [t11b-app-robust-retry](tasks/t11b-app-robust-retry.md) | 지시서 t11b: t11 재시도 (쓰기) | [20261008-190530](reports/20261008-190530-codex-t11b-app-robust-retry.md), [20261008-192309](reports/20261008-192309-codex-t11b-app-robust-retry.md) |
| [t12-ldw-bsw-rules](tasks/t12-ldw-bsw-rules.md) | 지시서 t12: LDW/BSW 판정 기준 변경 (쓰기) | [20261008-193522](reports/20261008-193522-claude-t12-ldw-bsw-rules.md) |
| [t13-distance-eval](tasks/t13-distance-eval.md) | 지시서 t13: 거리 추정 방식 비교 실험 (쓰기, 판정 로직 변경 금지) | [20261008-203852](reports/20261008-203852-claude-t13-distance-eval.md) |
| [t14-distance-gated-ffix](tasks/t14-distance-gated-ffix.md) | 지시서 t14: 거리 추정 개선 적용 (쓰기) | [20261008-212120](reports/20261008-212120-claude-t14-distance-gated-ffix.md) |
| [t15-box-rescue](tasks/t15-box-rescue.md) | 지시서 t15: 놓친 박스 되살리기 (쓰기) | [20261008-215605](reports/20261008-215605-claude-t15-box-rescue.md) |
| [t16-car-mode](tasks/t16-car-mode.md) | 지시서 t16: 자동차 블랙박스 모드 (쓰기, 작게) | [20261008-225344](reports/20261008-225344-claude-t16-car-mode.md) |
| [t17-nexar-run](tasks/t17-nexar-run.md) | 지시서 t17: Nexar 자동차 영상 10개 실행 (쓰기, 작게) | [20261008-230333](reports/20261008-230333-claude-t17-nexar-run.md) |
| [t18-nexar-eval](tasks/t18-nexar-eval.md) | 지시서 t18: 충돌 경고 평가 도구 (쓰기) | [20261008-230338](reports/20261008-230338-codex-t18-nexar-eval.md) |
| [t19-car-distance-diag](tasks/t19-car-distance-diag.md) | 지시서 t19: 자동차 모드 거리 붕괴 원인 진단 (읽기, 명령 실행 가능) | [20261008-234125](reports/20261008-234125-codex-t19-car-distance-diag.md) |
| [t20-car-geometry](tasks/t20-car-geometry.md) | 지시서 t20: 자동차 모드 기하 재설계 (쓰기) | [20261009-075815](reports/20261009-075815-claude-t20-car-geometry.md) |
| [t21-nexar30-eval-app](tasks/t21-nexar30-eval-app.md) | 지시서 t21: Nexar 30개 평가 준비와 앱 차량 선택 (쓰기) | [20261009-075815](reports/20261009-075815-codex-t21-nexar30-eval-app.md) |
| [t22-scene-only-run30](tasks/t22-scene-only-run30.md) | 지시서 t22: 한 장면 기준 판정으로 정리하고 Nexar 30개 실행 (쓰기) | 없음 |
| [t23-review-car-geometry](tasks/t23-review-car-geometry.md) | 지시서 t23: 자동차 모드 기하 코드 리뷰 (읽기) | [20261009-102720](reports/20261009-102720-codex-t23-review-car-geometry.md) |
| [t24-fix-miss-bugs-run30](tasks/t24-fix-miss-bugs-run30.md) | 지시서 t24: 리뷰가 찾은 미탐 쪽 버그 4개 수정 후 Nexar 30개 실행 (쓰기) | [20261009-105019](reports/20261009-105019-claude-t24-fix-miss-bugs-run30.md) |
| [t25-eval-nexar30](tasks/t25-eval-nexar30.md) | 지시서 t25: Nexar 30개 평가 (쓰기, 평가 결과 파일만) | [20261009-122503](reports/20261009-122503-codex-t25-eval-nexar30.md) |
| [t26-calib-guard-intrusion](tasks/t26-calib-guard-intrusion.md) | 지시서 t26: 보정 안전장치와 침범 기준 조정 후 30개 재실행 (쓰기) | 없음 |
| [t27-interior-calib-guard](tasks/t27-interior-calib-guard.md) | 지시서 t27: 차 실내 영역 마스크 + 보정 안전장치 + 침범 기준, 30개 재실행 (쓰기) | 없음 |
| [t28-car-model-probe](tasks/t28-car-model-probe.md) | 지시서 t28: 자동차용 사전학습 모델 두 개 시험 (쓰기, 실험 결과 파일만) | [20261009-124617](reports/20261009-124617-codex-t28-car-model-probe.md) |
| [t29-car-models-integrate](tasks/t29-car-models-integrate.md) | 지시서 t29: 자동차 모드에 사전학습 모델 통합 + t27 마무리, 30개 실행 (쓰기) | [20261009-133427](reports/20261009-133427-claude-t29-car-models-integrate.md) |
| [t30-eval-t29](tasks/t30-eval-t29.md) | 지시서 t30: t29 결과 평가 (쓰기, 평가 결과 파일만) | [20261009-152306](reports/20261009-152306-codex-t30-eval-t29.md) |
| [t31-lane-baseline-rejudge](tasks/t31-lane-baseline-rejudge.md) | 지시서 t31: 차로 안 평소 위치 기준 실험, 저장 결과로 다시 판정 (쓰기, GPU 금지) | [20261009-160828](reports/20261009-160828-codex-t31-lane-baseline-rejudge.md) |
| [t32-ldw-lane-baseline](tasks/t32-ldw-lane-baseline.md) | 지시서 t32: LDW에 차로 안 평소 위치 기준 넣기 (쓰기, GPU 금지) | [20261009-162357](reports/20261009-162357-claude-t32-ldw-lane-baseline.md) |
| [t33-web-mvp](tasks/t33-web-mvp.md) | 지시서 t33: 서비스용 웹 화면 구현 (FastAPI + HTML 한 장) (쓰기) | [20261009-170505](reports/20261009-170505-codex-t33-web-mvp.md) |
| [t34-run30-ldw](tasks/t34-run30-ldw.md) | 지시서 t34: t32 반영 엔진으로 Nexar 30개 재실행 (쓰기, GPU 사용 가능) | [20261009-180920](reports/20261009-180920-claude-t34-run30-ldw.md) |
| [t35-web-polish](tasks/t35-web-polish.md) | 지시서 t35: 웹 화면 다듬기 (쓰기, GPU 금지) | [20261009-180921](reports/20261009-180921-codex-t35-web-polish.md) |
| [t36-ego-distill](tasks/t36-ego-distill.md) | 지시서 t36: "내 차" 영역 증류 학습 (쓰기, GPU 사용) | [20261010-013124](reports/20261010-013124-claude-t36-ego-distill.md) |
| [t37-testset-run](tasks/t37-testset-run.md) | 지시서 t37: 시험 세트 99개 실행 (쓰기, 실행만, GPU 사용) | 없음 |
| [t38-eval-dev-test](tasks/t38-eval-dev-test.md) | 지시서 t38: 개발 세트와 시험 세트 평가 (쓰기, 평가 파일만, GPU 금지) | 없음 |
| [t39-web-e2e](tasks/t39-web-e2e.md) | 지시서 t39: 웹 실제 업로드 흐름 끝까지 확인 (쓰기, GPU 사용) | [20261010-030934](reports/20261010-030934-codex-t39-web-e2e.md) |
| [t40-lane-zero](tasks/t40-lane-zero.md) | 지시서 t40: 보정 단계에 차선 영점 추가 + 개발 세트 30개 재실행 (쓰기, GPU 사용) | [20261010-001115](reports/20261010-001115-claude-t40-lane-zero.md) |
| [t41_ldw_variants](tasks/t41_ldw_variants.md) | t41: LDW 두 방식 비교 (자동차 모드, dev 30개) | [20261010-074745](reports/20261010-074745-claude-t41_ldw_variants.md) |
| [t42_review_t41](tasks/t42_review_t41.md) | t42: t41(LDW 방식 비교) 리뷰 (read) | [20261010-092005](reports/20261010-092005-codex-t42_review_t41.md) |
| [t43_lane_influence](tasks/t43_lane_influence.md) | t43: 차선 검출이 소실점, 수평선, 카메라 높이, 거리에 주는 영향 (read) | [20261010-111046](reports/20261010-111046-codex-t43_lane_influence.md) |
| [t44_lane_trust](tasks/t44_lane_trust.md) | t44: 차선 신뢰 범위 2.0~4.0 m: 범위 밖이면 차선을 LDW, 수평선, 카메라 높이에서 모두 뺀다 (write) | [20261010-113754](reports/20261010-113754-claude-t44_lane_trust.md) |
| [t45-testset-run](tasks/t45-testset-run.md) | 지시서 t45: 시험 세트 99개 실행 (쓰기, 실행만, GPU 사용), t37 대체 | [20261010-152058](reports/20261010-152058-claude-t45-testset-run.md) |
| [t46-eval-dev-test](tasks/t46-eval-dev-test.md) | 지시서 t46: 개발 세트와 시험 세트 평가 (쓰기, 평가 파일만, GPU 금지), t38 대체 | [20261010-195740](reports/20261010-195740-codex-t46-eval-dev-test.md) |
| [t47-review-t44](tasks/t47-review-t44.md) | t47: t44(차선 신뢰 2.0~4.0 m) 리뷰 (read) | [20261010-135706](reports/20261010-135706-codex-t47-review-t44.md) |
| [t48-lane-trust-fix](tasks/t48-lane-trust-fix.md) | t48: t44 수정: 영점 실패는 LDW만 끄기, 근접 보험 누락 수정 (write, t44 이어서) | [20261010-140056](reports/20261010-140056-claude-t48-lane-trust-fix.md) |
| [t49-review-t48](tasks/t49-review-t48.md) | t49: t48 리뷰 (read) | [20261010-151820](reports/20261010-151820-codex-t49-review-t48.md) |
| [t50-repo-prep](tasks/t50-repo-prep.md) | t50: 새 리포 준비: 조사와 초안만 (write, .orch/repo_prep/ 에만 쓰기) | [20261010-145238](reports/20261010-145238-codex-t50-repo-prep.md) |
| [t51-screenshots](tasks/t51-screenshots.md) | t51: README용 스크린샷 만들기 (write, .orch/repo_prep/screenshots/ 에만 쓰기) | 없음 |
| [t52-repo-assemble](tasks/t52-repo-assemble.md) | t52: 새 리포 roadscope 조립 (write, .orch/repo_build/roadscope/ 에만 쓰기) | 없음 |
| [t52b-repo-assemble-resume](tasks/t52b-repo-assemble-resume.md) | t52b: t52 이어서 (컴퓨터 재부팅으로 중단됨) | 없음 |
