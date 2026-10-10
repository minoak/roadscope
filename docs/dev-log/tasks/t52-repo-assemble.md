# t52: 새 리포 roadscope 조립 (write, .orch/repo_build/roadscope/ 에만 쓰기)

## 배경
`.orch/repo_build/roadscope/`는 https://github.com/minoak/roadscope 를 clone한 것(커밋 cfb0d51, README.md 한 줄만 있음). 여기에 새 리포 내용을 조립한다. 민옥 결정(DECISIONS.md 2026-10-10, 새 리포 방침 줄): AGPL-3.0, 가중치 전부 제외(받는 방법만), Nexar 출처 표기와 번호판, 얼굴 흐림, .orch는 개인 경로 지운 정리본만 docs/dev-log/, 웹(FastAPI)만 남기고 Gradio app.py, 과거 노트북 제외, 새 이력(이 clone 위에 쌓음), 기존 리포와는 링크로만 연결(시작 커밋 b6aad74).
참고 자료: `.orch/repo_prep/` 의 inventory.md, licenses.md, structure.md, README_draft.md, screenshots/(index.md 포함).

## 할 일
1. structure.md 구조대로 엔진, 웹, tools, 테스트 파일을 Quest02에서 복사해 넣는다. 원본(Quest02)은 절대 수정하지 않는다. 복사본에서 하드코딩 절대 경로(<HOME> 등)를 설정값(환경변수 또는 config 파일 + 예시 파일)으로 바꾸고, 바꾼 목록을 적는다.
2. 넣지 않는 것: 가중치(.pt .pth .onnx .safetensors), 영상(mp4 등), runs_in/, runs_out/, web/.data/, web/tmp*, __pycache__, app.py(Gradio), 노트북, archive, .orch 원본, t45_work, 증류 가중치.
3. requirements.txt (structure.md 초안 기준, 실제 import 확인), .gitignore, LICENSE(AGPL-3.0 전문: 로컬 ultralytics 패키지 LICENSE 파일에서 복사), THIRD_PARTY_NOTICES.md(licenses.md 기준: 각 모델, 데이터 출처와 조건, 가중치는 리포에 없음을 명시).
4. docs/models.md: 가중치 받는 방법(모델 ID, revision, 출처 링크, 라이선스, 놓을 경로). 다운로드 스크립트가 있으면 tools/에, 없으면 문서만.
5. docs/images/: `.orch/repo_prep/screenshots/`의 web_01~03.png, hud_01~06.png만 복사(_raw/_old/_browser 제외). 용량이 크면 품질을 유지하는 선에서 PNG 최적화나 1280 폭 유지 JPG 변환.
6. docs/dev-log/: `.orch/`의 CONTEXT.md, DECISIONS.md, tasks/*.md, 보고(runs의 report.md 요약 수준)를 정리본으로. 개인 경로, 사용자 이름 외 개인 정보, 토큰, 내부 수정 후보 메모(CONTEXT.md의 "수정 후보(내부" 줄)는 뺀다. 시험 세트 관련 지시서는 남겨도 된다(절차 투명성).
7. README.md (한국어): README_draft.md를 바탕으로, 평가 표는 `runs_out/eval_t46/readme_table.md` 숫자(시험 99 / 개발 30, 이상치 제외, Wilson, 부트스트랩 95% 구간)로 채운다. 이미지 넣기. 한계 절에 정상 영상 주의 비율(시험 47.1%), 보정 실패 15/99, 밤 영상 3개뿐, 미탐 4건은 모두 거리 5 m 이상으로 기록돼 L2 미도달, 실제 정답 거리 미검증을 적는다. 출발점 절: https://github.com/minoak/AIFFEL_quest_eng/tree/b6aad7411ffdfc9d4518c56f9d253523b8751ba8/Main_Quest/Quest02 . 개발 방식 절은 .orch 공개 버전 문단 사용. 문체: 담백하게, 줄표(—) 남발, "이것이 ~의 핵심", "~인 셈", 가운뎃점 나열, 문학적 도입, 수사적 마무리 금지. 확인 안 된 숫자 금지.
8. 새 위치에서 단위 테스트가 도는지 확인(가중치 없이 도는 테스트만; 가중치가 필요한 테스트는 skip 처리 방식과 함께 보고). 엔진 실제 실행은 하지 않는다.
9. 커밋하지 마라. git add도 하지 마라(상태만 보고). 푸시 금지.

## 금지
`.orch/repo_build/roadscope/` 밖 쓰기 금지(원본 Quest02 수정 금지). GPU 금지. 패키지 설치 금지. 네트워크 다운로드 금지. runs_in/testset 내용 읽기 금지.

## 보고
파일 트리(2단계), 총 용량, 경로 치환 목록, 빠진 것과 이유, 테스트 결과, 민옥이 결정할 것.
