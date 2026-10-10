# 제3자 고지

RoadScope 소스 코드의 라이선스는 AGPL-3.0-only다 ([LICENSE](LICENSE)). RoadScope에서 사용하는 외부 코드와 모델, 데이터의 조건을 이 문서에 모았다. 조건은 2026-10-10에 개발 환경의 설치 패키지 메타데이터, 로컬에 있던 문서, Hugging Face 모델 카드와 데이터셋 카드로 확인했다.

"확인하지 못함"으로 적은 항목은 이용을 허락받았다는 뜻이 아니다. 사용 전에 각 공급자의 원문을 확인해야 한다.

## 1. 모델 가중치

가중치 파일은 리포에 포함하지 않았다. 다운로드 방법은 [docs/models.md](docs/models.md)에 정리했다.

| 모델 | 출처 | 확인한 조건 |
|---|---|---|
| YOLO11m | Ultralytics | Ultralytics 패키지는 AGPL-3.0 또는 Enterprise 라이선스를 안내한다. 이 체크포인트만 다룬 조항은 확인하지 못함 |
| YOLOPv2 | CAIC-AD/YOLOPv2 | 코드 MIT. 가중치는 BDD100K 학습. 가중치와 BDD100K의 조건은 확인하지 못함 |
| Mask2Former Swin-L, Mapillary Vistas semantic | facebook (Hugging Face `facebook/mask2former-swin-large-mapillary-vistas-semantic`) | 모델 카드 기준 `license: other`. 구체 조건은 확인하지 못함 |
| Depth Pro | apple (Hugging Face `apple/DepthPro-hf`) | 모델 카드 기준 `license: apple-amlr`. 라이선스 원문의 조건은 확인하지 못함 |
| lane_focus U-Net R34 5-fold | 이 프로젝트에서 학습 (Hugging Face `minoak/motorcycle-night-lane-focus`) | 모델 카드에 라이선스 항목 없음. 학습 데이터 조건은 확인하지 못함 |
| ResNet34 ImageNet encoder (학습 도구만) | Hugging Face `smp-hub/resnet34.imagenet` | 확인하지 못함 |

## 2. 데이터

| 데이터 | 쓰임 | 조건 |
|---|---|---|
| Nexar collision prediction (`nexar-ai/nexar_collision_prediction`) | 개발 30개, 시험 99개, 증류 학습 입력. README 화면 캡처 | Hugging Face 데이터셋 카드(2026-10-10 확인) 기준 Nexar Open Data License. 출처를 표기하면 무료로 이용할 수 있고, 재판매, 사람이나 차량의 재식별, 무기 개발, 해를 끼치는 시스템에 쓰는 것은 금지된다. 데이터에 접근하려면 연락처 공유에 동의해야 한다. 라이선스 원문(LICENSE 파일)은 직접 확인하지 못했다. 리포에는 영상, 라벨, 프레임을 넣지 않았다. README 캡처는 개발 세트 영상에서 만들었고 번호판과 얼굴이 보이는 부분은 흐리게 처리했다 |
| Mapillary Vistas | Mask2Former 학습 데이터 | 확인하지 못함 |
| BDD100K | YOLOPv2 학습 데이터 | 확인하지 못함 |
| Motorcycle Night Ride Segmentation (Kaggle) | lane_focus 학습 데이터 | 확인하지 못함. 리포에 넣지 않음 |
| ImageNet | ResNet34 encoder 사전학습 | 확인하지 못함 |

출처 표기: README의 화면 캡처와 HUD 이미지는 Nexar collision prediction 데이터셋 영상을 RoadScope로 분석한 결과다.

## 3. Python 패키지

Python 패키지는 리포에 포함하지 않고 `pip`로 설치한다. 아래 버전은 개발 환경에서 사용한 버전이다. 표는 패키지 메타데이터의 주 라이선스이며, 휠에 함께 들어 있는 구성품의 고지를 모두 대신하지 않는다.

| 패키지 | 버전 | 라이선스 |
|---|---|---|
| torch | 2.6.0+cu124 | BSD-3-Clause |
| torchvision | 0.21.0+cu124 | BSD-3-Clause |
| numpy | 2.1.0 | BSD-3-Clause |
| opencv-python | 4.13.0.92 (개발 환경은 headless 판) | Apache-2.0 |
| pillow | 11.2.1 | MIT-CMU |
| scipy | 1.18.1 | BSD-3-Clause |
| matplotlib | 3.10.3 | Matplotlib License (PSF 기반) |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause (Python 래퍼) |
| ultralytics | 8.4.174 | AGPL-3.0 |
| segmentation-models-pytorch | 0.5.0 | MIT |
| timm | 1.0.26 | Apache-2.0 |
| transformers | 5.8.1 | Apache-2.0 (Mask2Former, Depth Pro 구현 포함) |
| huggingface-hub | 1.15.0 | Apache-2.0 |
| fastapi | 0.142.2 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| python-multipart | 0.0.32 | Apache-2.0 |
| httpx | 0.28.1 | BSD-3-Clause |
| albumentations (학습 도구만) | 2.0.8 | MIT |

ultralytics 설치 시 함께 설치되는 ultralytics-thop(AGPL-3.0), ultralytics-platform(AGPL-3.0-only)도 있다.

## 4. 실행 파일과 글꼴

- FFmpeg: imageio-ffmpeg 휠에 들어 있는 FFmpeg 7.1 실행 파일은 `-L` 출력에서 GPL-3.0-or-later를 표시한다. RoadScope는 이 실행 파일을 별도 프로세스로 호출하며 리포에 포함하지 않는다.
- 한글 글꼴: HUD는 시스템에 설치된 한글 글꼴(또는 `HUD_FONT`로 지정한 파일)을 쓴다. 글꼴 파일은 리포에 포함하지 않는다.

## 5. 참고한 코드

`src/roadscope/car_models.py`의 YOLOPv2 입력 전처리와 출력 후처리는 YOLOPv2 공식 demo를 따라 작성했다. 아래는 해당 프로젝트의 라이선스 고지 원문이다.

```text
MIT License

Copyright (c) 2022 CAIC-AD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
