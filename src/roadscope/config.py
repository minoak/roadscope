# SPDX-License-Identifier: AGPL-3.0-only
"""경로 설정. 개인 절대 경로 대신 환경변수와 리포 기준 기본값을 쓴다.

    ROADSCOPE_MODELS_DIR  yolo11m.pt, yolopv2.pt 가 있는 폴더       (기본: <리포>/models)
    ROADSCOPE_LANE_DIR    lane_focus_fold0~4_best.pth 가 있는 폴더  (기본: <ROADSCOPE_MODELS_DIR>/lane_focus)

Hugging Face 모델 (Depth Pro, Mask2Former) 은 HF 캐시에서 읽는다. 캐시 위치는 HF_HOME 으로 바꾼다.
받는 방법과 놓을 위치는 docs/models.md.

REPO_ROOT 는 이 파일 위치 (src/roadscope/config.py) 에서 구한다. 리포를 받아 그 자리에서
`pip install -e .` 하거나 PYTHONPATH=src 로 실행하는 경우를 전제로 한다.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get('ROADSCOPE_MODELS_DIR') or REPO_ROOT / 'models')
LANE_DIR = Path(os.environ.get('ROADSCOPE_LANE_DIR') or MODELS_DIR / 'lane_focus')
RUNS_DIR = REPO_ROOT / 'runs_out'   # 엔진 기본 출력 루트 (Git 제외)
