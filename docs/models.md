# 모델 가중치 받기

이 리포에는 모델 가중치가 하나도 없다. 아래 표의 출처에서 각자 받아서 정해진 위치에 둔다. 각 가중치의 이용 조건은 공급자가 정한 것을 따르며, 이 리포의 AGPL-3.0-only는 가중치에 적용되지 않는다.

## 모드별로 필요한 모델

| 모델 | 자동차 모드 | 오토바이 모드 |
|---|---|---|
| YOLO11m | 필요 | 필요 |
| YOLOPv2 | 필요 | - |
| Mask2Former (Mapillary Vistas) | 필요 (사전 점검, `--no-m2f`로 끌 수 있음) | - |
| Depth Pro | 필요 (`--no-depth-guard`로 끌 수 있음) | 필요 |
| lane_focus U-Net R34 5-fold | - | 필요 |

## 모델별 출처와 위치

체크섬은 개발에 쓴 로컬 파일의 SHA-256이다. Hugging Face 모델은 개발 때 캐시에 있던 스냅샷 revision을 적었다.

| 모델 | 출처 | 버전 | 라이선스와 확인 상태 | 놓을 위치 |
|---|---|---|---|---|
| YOLO11m `yolo11m.pt` | Ultralytics GitHub release `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt` | release v8.4.0 (ultralytics 8.4.174의 기본 다운로드 태그). SHA-256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` | Ultralytics는 AGPL-3.0 또는 Enterprise 라이선스를 안내한다. 이 체크포인트만 따로 다룬 조항은 확인하지 못했다 | `<MODELS>/yolo11m.pt` |
| YOLOPv2 `yolopv2.pt` (TorchScript) | CAIC-AD/YOLOPv2 GitHub release `https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt` | release V0.0.1. SHA-256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c` | 코드는 MIT. 가중치는 BDD100K로 학습했다고 README에 적혀 있고, BDD100K의 조건은 확인하지 못했다 | `<MODELS>/yolopv2.pt` |
| Mask2Former | Hugging Face `facebook/mask2former-swin-large-mapillary-vistas-semantic` | revision `4772b6bf101d91f2534c106dc524d906aeb3c68a` | 모델 카드 기준 `license: other`. Mapillary Vistas 학습 모델이며 구체적인 이용 조건은 확인하지 못했다 | Hugging Face 캐시 |
| Depth Pro | Hugging Face `apple/DepthPro-hf` | revision `de816c8ce7168afcb231f96d501d72b869d0beda` | 모델 카드 기준 `license: apple-amlr`. 라이선스 원문의 조건은 확인하지 못했다 | Hugging Face 캐시 |
| lane_focus U-Net R34 5-fold `lane_focus_fold{0..4}_best.pth` | Hugging Face `minoak/motorcycle-night-lane-focus` | fold0 `ef82f911c722638df9be6b4eccc5fec78ff5298a47ab488000649fa1ef4215f3`<br>fold1 `a1702832bdd4e835dcc1390b65eda4e427397e4d5995296ebb1b8ffefdb80713`<br>fold2 `043a33c92e8395f2e1442560dfca1a1401282ac2934205b05e4e5b0d909e3ec4`<br>fold3 `77dbf6486c7ee455d2f9e4bb0645429ec1daf470e5973f4ec617e88397a302a5`<br>fold4 `b087aafe630e8fe38347e2cd0ff1257e5519216e2dc4c52a8dae830054ef8e4e` | 이 프로젝트에서 Kaggle Motorcycle Night Ride 데이터로 학습한 모델. 모델 카드에 라이선스 항목이 없고, 학습 데이터 조건도 확인하지 못했다 | `<LANE>/lane_focus_fold0_best.pth` ~ `fold4` |

`<MODELS>`와 `<LANE>`은 환경변수로 정한다.

| 환경변수 | 뜻 | 기본값 |
|---|---|---|
| `ROADSCOPE_MODELS_DIR` | `yolo11m.pt`, `yolopv2.pt` 폴더 | `<리포>/models` |
| `ROADSCOPE_LANE_DIR` | lane_focus 5개 파일 폴더 | `<ROADSCOPE_MODELS_DIR>/lane_focus` |
| `HF_HOME` | Hugging Face 캐시 위치 (Hugging Face 기본 변수) | 사용자 폴더의 `.cache/huggingface` |

기본값을 쓰면 아래처럼 놓는다. `models/`는 `.gitignore`에 들어 있다.

```text
models/
  yolo11m.pt
  yolopv2.pt
  lane_focus/
    lane_focus_fold0_best.pth ... lane_focus_fold4_best.pth
```

## 받는 명령 예시

이 리포를 정리하는 작업에서는 아래 명령을 실행하지 않았다. 네트워크가 되는 환경에서 리포 루트를 작업 폴더로 두고 실행한다. Hugging Face 모델을 받으려면 각 모델 페이지에서 조건 동의가 필요할 수 있다.

```powershell
New-Item -ItemType Directory -Force models/lane_focus | Out-Null

Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt -OutFile models/yolo11m.pt
Invoke-WebRequest https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt -OutFile models/yolopv2.pt

hf download facebook/mask2former-swin-large-mapillary-vistas-semantic
hf download apple/DepthPro-hf
hf download minoak/motorcycle-night-lane-focus --include "lane_focus_fold*_best.pth" --local-dir models/lane_focus

Get-FileHash models/yolo11m.pt, models/yolopv2.pt, models/lane_focus/*.pth -Algorithm SHA256
```

- Mask2Former는 `local_files_only=True`로 읽는다. 캐시에 없으면 엔진이 내려받지 않고 오류를 낸다.
- Depth Pro는 캐시에 없으면 실행 중 내려받으려 한다. 미리 받은 뒤 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`로 실행하면 네트워크를 쓰지 않는다.
- `yolo11m.pt`가 없으면 ultralytics가 실행 중 같은 경로로 내려받으려 한다.
- 엔진은 Hugging Face 모델을 기본 브랜치(main) 이름으로 읽는다. `hf download`에 `--revision`으로 커밋을 지정하면 main 참조가 캐시에 생기지 않아 오프라인 실행에서 찾지 못한다. 그래서 위 예시는 main을 받는다. 위 표의 revision과 지금 main이 같은지는 확인하지 않았다. 다르면 결과가 평가 때와 다를 수 있다.

## 학습 도구에만 필요한 것

| 모델 | 쓰는 곳 | 출처 | 버전 | 조건 |
|---|---|---|---|---|
| ResNet34 ImageNet encoder | `tools/distill/train_ego.py` (사전학습 encoder) | Hugging Face `smp-hub/resnet34.imagenet` | revision `7a57b34f723329ff020b3f8bc41771163c519d0c` | 확인하지 못했다. 엔진의 lane_focus 로드는 이 가중치를 쓰지 않는다 (`encoder_weights=None`) |

## 공개하지 않는 가중치

내 차 영역 학생 모델(`ego_unet_r34.pt`, U-Net ResNet34)은 공개하지 않는다. 교사 출력(Mask2Former 등), Nexar 영상, 사전학습 encoder의 조건이 학생 가중치에 어떻게 적용되는지 정리되지 않았기 때문이다. 기본 엔진은 이 모델을 쓰지 않는다. 같은 절차로 다시 만들려면 `tools/distill/`의 `make_labels.py`, `train_ego.py`, `eval_ego.py`를 순서대로 쓴다. Nexar 영상과 위 교사 모델이 필요하다.
