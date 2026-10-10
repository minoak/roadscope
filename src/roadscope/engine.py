# SPDX-License-Identifier: AGPL-3.0-only
"""engine.py — 블랙박스 영상 → 프레임별 위험도 판정 → frames.jsonl

    python -m roadscope.engine <영상경로> --fps 5
    → runs_out/<영상이름>/meta.json, frames.jsonl  (형식: docs/frame_schema.md)

프레임 1장 처리 (다른 코드에서 직접 호출):
    from roadscope.engine import Engine
    eng = Engine()                       # 모델 로드 (한 번)
    record = eng.assess_frame(bgr)       # 스키마 레코드 dict
    # 또는 모듈 함수 assess_frame(bgr) — 처음 호출 때 기본 Engine 을 만든다

구성 (DECISIONS.md):
    차선/ego 마스크 — lane_focus U-Net R34 5-fold 다수결 (노트북 §8 과 같은 전처리)
    객체           — YOLO (ultralytics). mvp.extract_objects (마스크 덩어리) 대신 검출 박스 사용
    거리           — ground / size / Depth Pro 융합에서 쓸 수 없는 성분을 뺀다 (fused_gated).
                     영상은 두 번 훑어 Depth Pro 초점거리를 장면 구간별로 고정한다 (run_video)
    판정           — mvp.assess_v5 와 같은 순서·규칙. 객체 추출 단계만 YOLO 박스로 교체
    트래킹         — tracker.py IoU 트래커로 track_id 부여 (판정 bin 에는 관여하지 않음)
    자동차 모드    — --vehicle car (t20, DECISIONS 2026-10-09). 판정 전에 보정 단계 (calibrate_car) 에서
                     구간별 수평선·소실점·보닛·카메라 높이를 고정하고, 거리·차로·TTC 를 기하로 계산한다
                     (geom_car.py). Depth Pro 는 박스 없는 근접 물체 보험에만 쓴다. 오토바이 모드는 그대로
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from roadscope import config
from roadscope import rules as mvp            # 판정 로직 (노트북 추출본, 기존 mvp.py)
from roadscope.tracker import SimpleIouTracker
from roadscope import geom_car as gc          # 자동차 모드 기하 (t20)
from roadscope import car_models              # 자동차 모드 사전학습 모델 (t29)


# ============================================================
# 설정
# ============================================================
# 가중치 위치는 config (환경변수 ROADSCOPE_MODELS_DIR, ROADSCOPE_LANE_DIR). 받는 방법은 docs/models.md
MODELS_DIR = config.MODELS_DIR
LANE_CKPTS = [config.LANE_DIR / f'lane_focus_fold{i}_best.pth' for i in range(5)]
DET_WEIGHTS = MODELS_DIR / 'yolo11m.pt'     # 없으면 ultralytics 가 이 경로로 내려받는다
DEPTH_MODEL = 'apple/DepthPro-hf'

# 세그 전처리 (노트북 predict_seg 의 albumentations Resize + Normalize 와 동일)
NUM_CLASSES = 7
IMG_SIZE = 768
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# YOLO — 도로 위 위협이 될 수 있는 COCO 클래스만. 이름은 mvp.OBJECT_WIDTH_M 의 차종 prior 키와 같다
DET_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
DET_CONF = 0.25          # ultralytics 기본값

# 차선 검증 (find_lanes 참고)
LANE_MAX_DYDX = 3.0      # 기울기 상한 |dx/dy| — 가로 노면표시 제외
VP_TOL = 0.20            # 선이 VP 를 지나는 허용 거리 (W 비율). 실측 (mixkit): 커브 바깥쪽 실제 차선
                         #   0.05~0.16, 미러·바람막이 상단 테두리 0.35 이상
U_CENTER = 0.25          # 바이크 정중앙 수직선 기준 |u|. u = (선의 화면 하단 x − 바이크 x) / (H − vp_y)
                         #   ≈ 횡거리 / 카메라 높이. 바람막이 테두리처럼 차체에 붙은 선이 여기 걸린다

# 차선 위 주행 (judge_lane 참고) — 근거리 = 화면 하단 30%
STRADDLE_NEAR_Y = 0.70   # 근거리 시작 행 (H 비율)
STRADDLE_TOL = 0.03      # 선 주변 허용 폭 (W 비율). 곡선 도로에서 먼 쪽 직선 근사와 어긋나는 몫
STRADDLE_MIN_ROWS = 0.06 # 근거리에서 선을 따라 lane 픽셀이 보여야 하는 행 수 (H 비율)
# (t10 의 차선 시간 축 보정 — 측정이 끊긴 프레임에서 직전 '차선 위 주행' 을 0.8초 이어받기 — 은 t22 에서 뺐다.
#  DECISIONS 2026-10-09: 위험도는 그 순간의 장면으로 평가한다. LDW 는 그 프레임에서 보이는 것만으로 정한다)

# BSW (judge 참고, DECISIONS 2026-10-08)
LANE_WIDTH_M = mvp.LANE_WIDTH_M   # 3.5m (노트북 상수)
BSW_SIDE_MAX_M = 8.0     # "측면 나란히" 종방향 거리 한도 (임시값 — 민옥 확인 전)
BSW_SIDE_LAT_MIN = 0.5 * LANE_WIDTH_M   # 이보다 가까우면 내 차로 안
BSW_SIDE_LAT_MAX = 1.5 * LANE_WIDTH_M   # 차선 1폭 + 옆차선 절반

# 거리 융합 / 초점거리 고정 (t14, DECISIONS 2026-10-08)
FUSION_GATED = True      # 쓸 수 없는 성분을 빼는 융합 (estimate_distance)
EDGE_PX = 3              # 박스가 화면 가장자리에서 이 안이면 잘린 박스
FOCAL_LOCK = True        # run_video 두 번 훑기로 구간별 초점거리 고정 (--no-focal-lock 으로 끔)
CUT_NCC = 0.5            # 장면 전환: 연속 원본 프레임 썸네일 정규화 상관이 이 미만 (run_video 참고)

# 차종 (t16) — moto: 오토바이 (기본, 지금까지와 같음) | car: 자동차 블랙박스 (--vehicle car)
#   car 는 lane_focus 의 My bike(5)·Rider(6) 를 ego 로 쓰지 않는다 (보닛을 바이크로 잡아도 무시).
#   ego 대신 화면 아래 HOOD_RATIO 띠를 보닛으로 보고 검출·차선에서 뺀다. 내 위치는 화면 가로 중앙.
VEHICLE = 'moto'
HOOD_RATIO = 0.12            # car 기본값. 보닛은 보통 하단 0~20% — 너무 크면 바로 앞 차의 하단·차선이 잘린다
HOOD_BOX_FRAC = 0.5          # 박스 높이의 이 비율 이상이 보닛 띠 안이면 내 차 (보닛·반사) 로 보고 버린다
CAR_CAMERA_HEIGHT_M = 1.3    # (t17 값. t20 부터 car 모드는 보정 단계의 카메라 높이를 쓴다 — 미사용)

# 자동차 모드 기하 (t20, DECISIONS 2026-10-09) — 수식과 기준값은 geom_car.py
HOOD_MANUAL = False          # --hood-ratio 를 직접 줬으면 보정 대신 그 값 (수동 지정)
CAR_FPS_DEFAULT = 10.0       # car 기본 평가 fps. TTC 0.5초 창에 관측 6개 (5fps 면 3개라 직선 맞춤이 불안정)
DEPTH_EVERY_SEC = 0.4        # car 모드 Depth Pro 실행 간격 (박스 없는 근접 물체 보험 전용, 그 사이는 직전 경보 유지)
CAR_TTC = True               # TTC 계산·기록 (판정에는 안 씀, DECISIONS 2026-10-09). --no-ttc 로 끔
CAR_DEPTH_GUARD = True       # --no-depth-guard 로 끔 (끄면 Depth Pro 를 돌리지 않는다)
CAL_CACHE_MAX = 240          # 보정 단계에서 돌린 차선 세그·YOLO 결과를 본 판정에 다시 쓰는 프레임 수 상한 (메모리)
CAR_M2F = True               # 사전 점검 Mask2Former 내 차 영역 (t29). --no-m2f 로 끔
M2F_SAMPLE_SEC = (0.5, 2.0, 3.5, 5.0, 6.5)   # 구간 시작 기준 Mask2Former 를 돌릴 시각 (보정 구간 8초 안, 과반 투표)
CAR_CAL = None               # 지금 판정 중인 구간의 보정값 (gc.CarCalib). Engine.set_car_calib 이 바꾼다
# [car] LDW 방식 (t41 비교 실험 — 어느 쪽을 쓸지는 민옥이 정한다). --ldw-variant 로 고른다. 기본 V0 = t40 그대로
#   VA : 영점 실패 판정 완화 — 최소 근거 프레임 max(30, 본 프레임 10%) → max(15, 5%) (gc.LANE_ZERO_RELAX_*).
#        y_ref·차로 폭·b 를 정하는 규칙과 차로 폭 검산 (범위 밖이어도 성공, 신뢰도 low) 은 V0 그대로
#   VB : 영점 신뢰도가 high 가 아니면 (지금 규칙에서는 y_ref 차로 폭이 gc.LANE_ZERO_W_RANGE_M 2.75~3.75 m 밖)
#        그 구간 LDW 는 전부 측정 불가 (경고를 내지 않는다)
#   VAB: 둘 다
LDW_VARIANTS = ('V0', 'VA', 'VB', 'VAB')
LDW_VARIANT = 'V0'
# [car] 차선 신뢰 (t44·t48, DECISIONS 2026-10-10, gc.lane_trust). 불신 구간은 LDW 측정 불가 ("차선 신뢰 낮음").
# 그중 영점 차로 폭 (또는 차선 없는 수평선으로 다시 잰 검산 폭) 이 gc.LANE_TRUST_W_RANGE_M 밖인 구간만 보정값을
# 차선을 뺀 것 (FOE·차량 박스 회귀·차량 높이) 으로 바꾸고 근접 보험 축척에서도 차선 픽셀을 뺀다. 영점 실패·검산 불가는
# LDW 만 끄고 보정값은 t40 그대로. --no-lane-trust 로 끄면 t40 동작 (위 LDW 방식은 그대로 고를 수 있다)
LANE_TRUST = True

# 놓친 박스 되살리기 (t15) — 겹마다 끌 수 있다 (CLI --no-lowconf / --no-mask-boxes / --no-ghost)
RESCUE_LOWCONF = True
RESCUE_CONF_MIN = 0.10
RESCUE_AREA = 0.10           # 화면 면적 비율. 차(1.8×1.5m) 면적 ≈ 2.0~3.9/d² (초점 0.65~0.9W) → d ≈ 4.5~6m
RESCUE_BOTTOM = 0.90         # 박스 하단이 화면 높이 90% 아래 = 지면 접점이 바로 앞·옆
RESCUE_AREA_BOTTOM = 0.03
RESCUE_EGO_BOX = 0.30        # 박스 중 ego 비율 이상이면 버림
RESCUE_EGO_COVER = 0.15      # ego 마스크 중 박스가 덮는 비율 이상이면 버림
RESCUE_DUP = 0.70            # 이미 받은 박스와 겹치는 비율 이상이면 같은 대상
MASK_BOXES = True
MASK_MIN_AREA = 0.005        # 덩어리 면적 비율. 차 면적 ≈ 2.0/d² (초점 0.65W) → d ≈ 20m
MASK_COVERED = 0.20
MASK_EGO = 0.20
MASK_BELOW_VP = 0.02         # 덩어리 아래끝이 VP 높이 + 2%H 보다 아래여야 (지면 접점)
SLIDESHOW_CUT_RATIO = 0.5    # 평가 프레임의 이 비율 이상이 장면 전환이면 사진 이어 붙인 영상으로 본다
GHOST_BOXES = True           # 유지 시간은 트래커 ghost 설정 (TRACKER_GHOST_SEC) 그대로
NEAR_OF_TWO = True           # 거리 성분 두 개만 남고 2배 이상 차이면 가까운 쪽 (estimate_distance)
NEAR_OF_TWO_RATIO = 2.0

# ego 박스 기준 제거 (judge 3단계 참고)
EGO_LIKE_CLS = {'motorcycle', 'person', 'bicycle'}
EGO_COVER_MIN = 0.45

# 트래커 — v6 영상 파이프라인과 같은 IoU 0.3, ghost 유지 0.5초 (평가 fps 기준 프레임 수로 환산)
TRACKER_IOU = 0.3
TRACKER_GHOST_SEC = 0.5

DIRECTION_EN = {'좌측': 'left', '우측': 'right', 'left': 'left', 'right': 'right'}


def _r1(v):
    """거리 m — 소수 1자리, 측정 불가는 None."""
    return None if v is None else round(float(v), 1)


# ============================================================
# 모델 로더 / 추론
# ============================================================
def load_lane_models(ckpt_paths, device):
    """노트북 load_seg_model 과 같은 구조 (smp U-Net R34, 7-class)."""
    import segmentation_models_pytorch as smp
    models = []
    for p in ckpt_paths:
        model = smp.Unet('resnet34', encoder_weights=None,
                         in_channels=3, classes=NUM_CLASSES)
        # {state_dict, epoch, miou, lane_iou, fold} — 텐서·기본 타입뿐이라 weights_only=True 로 열린다
        ckpt = torch.load(str(p), map_location=device, weights_only=True)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        models.append(model.to(device).eval())
    return models


@torch.inference_mode()
def predict_lane_mask(models, rgb, device):
    """5-fold 픽셀 다수결 → (H, W) uint8 7-class mask.

    노트북은 fold 마다 원본 크기로 NEAREST 복원 후 투표하지만, NEAREST 는 픽셀 선택일 뿐이라
    768 에서 투표 후 한 번 복원해도 결과가 같다 (동점은 둘 다 낮은 클래스 번호 우선).
    """
    H, W = rgb.shape[:2]
    x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    votes = torch.zeros((NUM_CLASSES, IMG_SIZE, IMG_SIZE), dtype=torch.int32, device=device)
    for model in models:
        pred = model(t).argmax(1)[0]
        votes += torch.nn.functional.one_hot(pred, NUM_CLASSES).permute(2, 0, 1).int()
    mask = votes.argmax(0).to(torch.uint8).cpu().numpy()
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def load_depth_pro(device):
    """노트북 load_depth_pro 와 같음 (cuda 면 fp16)."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL, dtype=dtype).to(device).eval()
    return proc, model


@torch.inference_mode()
def predict_depth(proc, model, rgb, device, info=None, floor=0.5):
    """노트북 predict_depth (depth_pro 분기) 와 같음. Returns (depth_map m, focal_px).

    info (dict) 를 주면 진단용으로 모델이 낸 초점거리 원값과 대체값 사용 여부를 채운다.
    floor: 깊이 하한 (노트북 0.5m). car 모드 근접 보험은 원값이 0.5m 아래로 무너지는 영상이 있어 낮춰 쓴다 (t19)
    """
    H, W = rgb.shape[:2]
    inputs = proc(images=Image.fromarray(rgb), return_tensors='pt')
    if device == 'cuda':
        inputs = {k: (v.to(device).half() if v.dtype == torch.float32 else v.to(device))
                  for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    post = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])
    depth = post[0]['predicted_depth'].float().cpu().numpy().astype(np.float32)
    depth = np.nan_to_num(depth, nan=100.0, posinf=100.0, neginf=floor)
    depth = np.clip(depth, floor, 100.0)
    focal_px = float(post[0].get('focal_length', 0.0))
    if info is not None:
        info['focal_raw'] = focal_px
        info['focal_fallback'] = focal_px <= 0
    if focal_px <= 0: focal_px = W * 0.8
    if info is not None:
        info['focal'] = focal_px   # 판정에 쓰는 값 (대체값 포함)
    return depth, focal_px


def detect_objects(det_model, bgr, device, conf=DET_CONF, lowconf=None):
    """YOLO → mvp.extract_objects 와 같은 모양의 dict 리스트 (+ cls, conf, source).

    class_id 는 4 (Moveable) 로 통일 — 판정 로직이 class 4 만 위협 대상으로 다루기 때문.
    클래스 간 중복 박스 (car/truck 동시 검출 등) 는 agnostic NMS 로 하나만 남긴다.
    lowconf 를 주면 conf 미만 ~ lowconf 이상 박스도 source='yolo_lowconf' 로 돌려준다
    (구제 후보 — judge_scene 의 rescue_lowconf 가 거른다). NMS 는 높은 점수부터 남기므로
    conf 이상 박스 집합은 lowconf 를 줘도 같다.
    """
    H, W = bgr.shape[:2]
    res = det_model.predict(
        bgr, conf=min(conf, lowconf) if lowconf else conf, classes=list(DET_CLASSES),
        agnostic_nms=True, device=0 if device == 'cuda' else 'cpu', verbose=False)[0]
    objs = []
    if res.boxes is None or len(res.boxes) == 0:
        return objs
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    for (x0, y0, x1, y1), c, p in zip(xyxy, cls_ids, confs):
        x0 = int(np.clip(round(x0), 0, W - 1)); x1 = int(np.clip(round(x1), 0, W - 1))
        y0 = int(np.clip(round(y0), 0, H - 1)); y1 = int(np.clip(round(y1), 0, H - 1))
        if x1 <= x0 or y1 <= y0: continue
        objs.append({
            'class_id': 4,
            'cls': DET_CLASSES.get(int(c), str(c)),
            'conf': float(p),
            'bbox': (x0, y0, x1, y1),
            'cx': int((x0 + x1) / 2),
            'cy': int((y0 + y1) / 2),
            'source': 'yolo' if p >= conf else 'yolo_lowconf',
        })
    return objs


# ============================================================
# 놓친 박스 되살리기 (t15, DECISIONS 2026-10-08) — 1겹 저확신 구제, 2겹 마스크 보조 박스.
# 3겹 (트래커 유령) 은 Engine._ghost_objects. 되살린 객체는 aux=True:
# 거리 기반 근접 판정 (FCW, BSW 근접·나란히) 에만 쓰고 차선 침범·역주행 판정에는 넣지 않는다
# ============================================================
def ego_mask_for(mask):
    """moto: lane_focus My bike(5)+Rider(6) 로 만든 ego 마스크 (mvp.get_ego_mask).
    car: 빈 마스크 — 자동차 보닛을 바이크로 잡아도 무시한다 (ego 역할은 보닛 띠가 대신)."""
    if VEHICLE == 'car':
        return np.zeros(mask.shape, np.uint8)
    return mvp.get_ego_mask(mask)


def hood_top(H):
    """car 모드 보닛 띠가 시작하는 행. moto 거나 HOOD_RATIO 0 이면 H (보닛 없음).
    car 모드는 보정 단계에서 정한 값 (CAR_CAL, 수동 --hood-ratio 포함), 보정 전에는 HOOD_RATIO."""
    if VEHICLE == 'car' and CAR_CAL is not None and CAR_CAL.H == H:
        return int(CAR_CAL.hood_row)
    if VEHICLE == 'car' and HOOD_RATIO > 0:
        return int(round(H * (1 - HOOD_RATIO)))
    return H


def _manual_hood_row(H):
    return int(round(H * (1 - HOOD_RATIO))) if HOOD_RATIO > 0 else H


_DEFAULT_CALS = {}


def _car_cal(W, H):
    """car 모드에서 지금 쓰는 보정값. 보정 단계를 거치지 않은 호출 (단일 프레임 assess_frame) 은 기본값
    (수평선 실패 취급 — 바닥 거리·X 를 쓰지 않는다)."""
    if CAR_CAL is not None and CAR_CAL.W == W and CAR_CAL.H == H:
        return CAR_CAL
    key = (W, H, HOOD_RATIO, HOOD_MANUAL)
    if key not in _DEFAULT_CALS:
        _DEFAULT_CALS[key] = gc.default_calib(W, H, _manual_hood_row(H),
                                              'manual' if HOOD_MANUAL else 'default')
    return _DEFAULT_CALS[key]


def _car_corridor(cal):
    """내 차로 (|X| ≤ 2.5m) corridor — 경계선은 소실점을 지나고, 구역 높이는 FCW 거리 기준 (3/5/10m).
    구간마다 한 번 만든다 (보정값에 붙여 둔다)."""
    if not cal.horizon_ok:
        return None
    if getattr(cal, '_corridor', None) is None:
        lx, rx = gc.corridor_bottom_x(cal)
        alphas = gc.corridor_alphas(cal, mvp.FCW_DIST_CRITICAL, mvp.FCW_DIST_WARNING, mvp.FCW_DIST_CAUTION)
        cal._corridor = mvp.build_corridor(cal.vp_x, cal.horizon_y, lx, rx, cal.H, cal.W,
                                           vp_conf=1.0, alphas=alphas) or False
    return cal._corridor or None


def _removal_hood_top(H):
    """[car] 박스를 내 차로 보고 지우는 판단 (in_hood, hood_cover) 에 쓰는 보닛 행. moto 는 H.
    보닛 추정이 실패해 기본 비율 (HOOD_RATIO) 을 쓰는 중이면 H — 가정한 띠로 박스를 지우지 않는다
    (t24, 미탐 우선: 오탐이 늘더라도 실제 근접 차를 지우는 것보다 낫다). 보정 전·단일 프레임 호출도 같다.
    차선 픽셀 제거 (lane_input_mask) 는 박스를 지우는 판단이 아니라 그대로 hood_top 을 쓴다."""
    if VEHICLE != 'car':
        return H
    if CAR_CAL is not None and CAR_CAL.H == H:
        return int(CAR_CAL.hood_row) if CAR_CAL.hood_known else H
    return hood_top(H) if HOOD_MANUAL else H


def in_hood(bbox, H, frac=HOOD_BOX_FRAC):
    """박스 높이 중 보닛 띠 안에 든 비율이 frac 이상이면 True (내 차로 본다). moto 는 항상 False.
    되살린 박스 (저확신·마스크) 는 moto 의 ego 가드처럼 더 엄격한 비율을 준다 (t20): 앞의 실제 물체는
    대시보드에 가려 박스가 보닛 선 근처에서 끝나야 하는데, 띠 안으로 깊이 내려온 박스는 내 차를 잡은 것.
    보닛 추정이 실패했으면 항상 False (_removal_hood_top)."""
    hy = _removal_hood_top(H)
    x0, y0, x1, y1 = bbox
    if y1 <= hy:
        return False
    return (y1 - max(y0, hy)) / max(y1 - y0, 1) >= frac


def hood_cover(bbox, W, H):
    """[car] 보닛 띠 (hood_top ~ 화면 아래, 화면 전체 폭) 면적 중 박스가 덮는 비율. moto 는 0.
    moto 의 'ego 마스크를 덮는 박스 = 내 바이크' 규칙에 해당한다: 앞의 실제 물체는 대시보드에 가려 보닛 선
    위에서 끝나므로 띠를 넓게 덮을 수 없다. YOLO 가 대시보드째 화면 전체를 차로 잡는 경우 (pos_00077, 00905).
    보닛 추정이 실패했으면 0 (_removal_hood_top)."""
    hy = _removal_hood_top(H)
    if hy >= H:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(x1 - x0, 0) * max(y1 - max(y0, hy), 0) / (W * (H - hy))


def interior_frac(bbox, W, H):
    """[car] 박스 면적 중 차 실내 마스크 (gc.solve_interior: 보닛 띠 ∪ 테두리 정지 영역 ∪ 고정 박스) 비율.
    moto 거나 마스크가 없으면 (보정 전·실내 추정 실패) 0 — 지우지 않는다."""
    if VEHICLE != 'car' or CAR_CAL is None or CAR_CAL.W != W or CAR_CAL.H != H:
        return 0.0
    return gc.interior_frac(bbox, CAR_CAL)


def _box_area(b):
    return max(b[2] - b[0], 1) * max(b[3] - b[1], 1)


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def rescue_lowconf(cands, kept, ego_mask, W, H, removed=None):
    """1겹 — conf 0.10~0.25 박스 중 '크고 가까운' 것만 cls='unknown_near' 로 받는다.

    크고 가까움: 화면 면적 ≥ RESCUE_AREA (차 기준 대략 5m 안 — 판정이 반응하는 거리),
                 또는 박스 하단이 화면 RESCUE_BOTTOM 아래이고 면적 ≥ RESCUE_AREA_BOTTOM (바로 옆·앞).
    ego 가드 (기존 ego 필터보다 엄격): 박스의 RESCUE_EGO_BOX 이상이 ego 이거나 ego 마스크의
    RESCUE_EGO_COVER 이상을 덮으면 버린다. 실측 (night·mixkit): 저확신 큰 박스는 거의 다 내 바이크
    (박스 중 ego 0.36~0.94) 와 라이더 팔다리 (0.59~0.77) 였다.
    이미 받은 박스와 같은 대상 (구제 박스의 RESCUE_DUP 이상이 겹침) 도 버린다.
    car 모드: 보닛 띠·실내 마스크 기준으로 내 차인 박스는 removed (리스트를 주면) 에 넣는다 (기록용).
    """
    ego_area = int(ego_mask.sum())
    out = []
    for o in cands:
        b = o['bbox']
        area = _box_area(b) / (W * H)
        near = area >= RESCUE_AREA or (b[3] >= RESCUE_BOTTOM * H and area >= RESCUE_AREA_BOTTOM)
        if not near:
            continue
        if (in_hood(b, H, RESCUE_EGO_BOX)          # car 모드: 보닛 띠에 걸친 큰 박스 = 내 차
                or hood_cover(b, W, H) >= RESCUE_EGO_COVER
                or interior_frac(b, W, H) >= gc.INT_BOX_FRAC_AUX):
            if removed is not None:
                removed.append({**o, 'cls': 'unknown_near', 'det_cls': o['cls'], 'source': 'yolo_lowconf'})
            continue
        ov = int(ego_mask[b[1]:b[3] + 1, b[0]:b[2] + 1].sum())
        if ov / _box_area(b) >= RESCUE_EGO_BOX or (ego_area and ov / ego_area >= RESCUE_EGO_COVER):
            continue
        if any(_inter(b, k['bbox']) / _box_area(b) >= RESCUE_DUP for k in kept + out):
            continue
        out.append({**o, 'cls': 'unknown_near', 'det_cls': o['cls'],
                    'source': 'yolo_lowconf', 'aux': True})
    return out


def mask_boxes(mask, ego_mask, kept, W, H, vp_y):
    """2겹 — lane_focus Moveable (class 4) 연결 성분 중 박스로 잡히지 않은 큰 덩어리를 보조 박스로.

    덩어리 면적 ≥ MASK_MIN_AREA (차 기준 대략 20m 안), 이미 받은 박스 안에 MASK_COVERED 이상 들어가
    있으면 같은 대상으로 보고 버림, 덩어리 박스의 MASK_EGO 이상이 ego (class 5·6) 면 버림.
    덩어리 아래끝이 지평선 (vp_y + MASK_BELOW_VP·H) 보다 위면 버림 — 도로 위 물체는 지평선 아래에서
    땅에 닿는다. 실측 (mixkit): 바람막이 윗부분이 class 4 로 잡혀 2.1m 치명 FCW 오탐이 났다.
    """
    n, lbl, st, _ = cv2.connectedComponentsWithStats((mask == 4).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < MASK_MIN_AREA * W * H:
            continue
        b = (int(x), int(y), int(x + w - 1), int(y + h - 1))
        if (b[3] < vp_y + MASK_BELOW_VP * H or in_hood(b, H, MASK_EGO)
                or interior_frac(b, W, H) >= gc.INT_BOX_FRAC_AUX):   # car: 실내 마스크 위 덩어리
            continue
        comp = lbl[b[1]:b[3] + 1, b[0]:b[2] + 1] == k
        covered = 0
        for o in kept:
            ob = o['bbox']
            ix0, iy0 = max(ob[0], b[0]), max(ob[1], b[1])
            ix1, iy1 = min(ob[2], b[2]), min(ob[3], b[3])
            if ix1 > ix0 and iy1 > iy0:
                covered = max(covered, int(comp[iy0 - b[1]:iy1 - b[1] + 1,
                                               ix0 - b[0]:ix1 - b[0] + 1].sum()) / a)
        if covered >= MASK_COVERED:
            continue
        if ego_mask[b[1]:b[3] + 1, b[0]:b[2] + 1].sum() / _box_area(b) >= MASK_EGO:
            continue
        out.append({'class_id': 4, 'cls': 'mask_object', 'conf': 0.0, 'bbox': b,
                    'cx': int((b[0] + b[2]) / 2), 'cy': int((b[1] + b[3]) / 2),
                    'source': 'mask', 'aux': True})
    return out


# ============================================================
# 차선 검증 — mvp.detect_lane_lines 를 감싼다 (원본 로직은 그대로)
# ============================================================
def lane_input_mask(mask, ego_mask):
    """차선 검출용 mask: 바이크 영역 안의 lane 픽셀(class 3) 을 지운다.

    바이크 영역 = 열마다 ego 마스크 최상단보다 아래. 핸들·계기판 사이 틈으로 보이는
    lane 조각이 짧은 가짜 선분이 되어 ego 차선으로 뽑히는 것을 막는다. 다른 판정
    (객체, ego 위치) 은 원래 mask 를 그대로 쓴다.
    car 모드는 ego 마스크가 비어 있고, 대신 보닛 띠 (hood_top 아래) 와 내 차 영역 (t29, 대시보드·앞유리 반사) 의
    lane 픽셀을 지운다.
    """
    H = mask.shape[0]
    hy = hood_top(H)
    if VEHICLE == 'car':
        im = getattr(CAR_CAL, 'interior_mask', None) if CAR_CAL is not None and CAR_CAL.H == H else None
        if hy >= H and im is None:
            return mask
        out = mask.copy()
        band = out[hy:]
        band[band == 3] = 0
        if im is not None and im.shape == out.shape:
            out[(out == 3) & im] = 0
        return out
    if hy < H:
        out = mask.copy()
        band = out[hy:]
        band[band == 3] = 0
        return out
    if ego_mask.sum() == 0:
        return mask
    has_ego = ego_mask.any(axis=0)
    ego_top = np.where(has_ego, ego_mask.argmax(axis=0), H)
    bike_region = np.arange(H)[:, None] >= ego_top[None, :]
    out = mask.copy()
    out[(out == 3) & bike_region] = 0
    return out


def line_u(line, bike_x, vp_y, H):
    """선의 바이크 기준 횡위치 u = (화면 하단 x − 바이크 x) / (H − vp_y) ≈ 횡거리 / 카메라 높이."""
    return (mvp.line_x_at(line, H - 1) - bike_x) / max(H - 1 - vp_y, 1.0)


def find_lanes(mask, ego_mask, bike_x=None, vp_fixed=None):
    """검증된 차선 → {'lines', 'vp', 'ego', 'ok', 'reason', 'n_raw'}.

    vp_fixed (x, y): [car] 보정 단계에서 고정한 소실점. 주면 프레임 차선 교점 대신 이 점으로 3)·4) 를
    판단한다 (선이 2개 미만이면 수렴 안 됨으로 본다 — 한 줄을 화면 중앙 대칭으로 복제한 차선을 막는다).

    1) 바이크 영역 lane 픽셀 제거 (lane_input_mask)
    2) 기울기 상한: |dx/dy| > LANE_MAX_DYDX 인 선 제외 — 횡단보도·정지선 같은 가로 노면표시
    3) 수렴 조건: 남은 선들의 VP 신뢰도 < 0.3 이면 차선을 쓰지 않는다
       (노트북은 이때 VP 만 기본값으로 바꾸고 ego 차선·LDW 는 그대로 썼다)
    4) VP 를 지나지 않는 선 제외 (|x(vp_y) − vp_x| > VP_TOL·W) — 미러·핸들 테두리
    5) 바이크 정중앙 수직선 제외 (|u| < U_CENTER) — 바람막이 테두리처럼 차체에 붙은 선.
       실제 차선이 바이크 정중앙 아래에 있는 경우도 함께 빠진다 (한 프레임으로는 구분 불가)
    ok=False 면 ego 차선을 비워서 LDW·차선 침범·corridor 모두 차선 없이 판정한다.
    """
    H, W = mask.shape
    if bike_x is None:
        bike_x = W / 2
    raw = mvp.detect_lane_lines(lane_input_mask(mask, ego_mask))
    lines = [l for l in raw if abs(l['dydx']) <= LANE_MAX_DYDX]
    if vp_fixed is not None:
        vp_x, vp_y = int(round(vp_fixed[0])), int(round(vp_fixed[1]))
        vp_conf = 1.0 if len(lines) >= 2 else 0.0
    else:
        vp_x, vp_y, vp_conf = mvp.vp_from_lanes(lines, W, H)
    if vp_conf >= 0.3:
        lines = [l for l in lines
                 if abs(mvp.line_x_at(l, vp_y) - vp_x) <= VP_TOL * W
                 and abs(line_u(l, bike_x, vp_y, H)) >= U_CENTER]
    ego = mvp.identify_ego_lane(lines, W, H)
    if ego['left'] is None:
        reason = '차선 없음'
    elif vp_conf < 0.3:
        reason = '차선 수렴 안 됨'
    else:
        reason = ''
    if reason:
        ego = {'left': None, 'right': None, 'conf': 0.0}
    return {'lines': lines, 'n_raw': len(raw), 'vp': (vp_x, vp_y, vp_conf),
            'ego': ego, 'ok': not reason, 'reason': reason}


# ============================================================
# 판정 — mvp.assess_v5 와 같은 순서. 바뀐 곳은 [YOLO] / [차선] 표시
# ============================================================
def box_truncated(bbox, W, H):
    """박스가 화면 가장자리 EDGE_PX 안에 닿으면 잘린 것으로 본다."""
    x0, y0, x1, y1 = bbox
    return x0 <= EDGE_PX or y0 <= EDGE_PX or x1 >= W - 1 - EDGE_PX or y1 >= H - 1 - EDGE_PX


def estimate_distance(est, obj, d_depth, vp_y, vp_default, W, H):
    """[YOLO] 차종 prior 를 bbox 가로세로비 추정(classify_object) 대신 검출 클래스로 정한다.
    ground / size / depth 3-estimator fusion 은 DistanceEstimator.estimate 와 같다.

    [t14] 쓸 수 없는 성분을 빼고 융합 (fused_gated, DECISIONS 2026-10-08):
      - 화면 끝에 잘린 박스 → size 제외 (보이는 폭이 줄어 멀게 나온다)
      - 기본 VP (차선으로 VP 를 못 잡음) → ground 제외 (지평선 근거가 없다)
      남은 성분이 하나면 그 값 (fuse_distances 가 그대로 돌려준다). 제외 사유는 'gated' 에.
    d_depth 는 judge_scene 에서 뽑아 둔 박스 중앙값 (초점 고정 시 배율 적용 후).
    """
    obj_class = obj['cls'] if obj['cls'] in mvp.OBJECT_WIDTH_M else 'moveable'
    bbox = obj['bbox']
    d_ground = mvp.distance_from_ground(bbox, vp_y, est.f_px, est.camera_h)
    d_size = mvp.distance_from_size(bbox, obj_class, est.f_px)
    use = {'ground': d_ground, 'size': d_size, 'depth': d_depth}
    gated = []
    if FUSION_GATED:
        if vp_default and d_ground is not None:
            use['ground'] = None
            gated.append('ground:기본 VP')
        if box_truncated(bbox, W, H) and d_size is not None:
            use['size'] = None
            gated.append('size:박스 잘림')
    fused = mvp.fuse_distances(use, depth_is_metric=est.depth_is_metric)
    # [t15] 성분이 두 개만 남으면 이상치를 거를 수 없다 (중앙값 = 두 값의 중간). 2배 이상 차이면
    # 가까운 쪽을 쓴다 (미탐 우선, DECISIONS 2026-10-08). 예: night 12.2s ground 18.5 vs depth 3.8
    valid = [v for v in use.values() if v is not None]
    if NEAR_OF_TWO and len(valid) == 2 and max(valid) >= NEAR_OF_TWO_RATIO * min(valid):
        d = min(valid)
        fused['d_fused_m'] = round(d, 2)
        fused['d_bin_idx'] = next((i for i, (lo, hi) in enumerate(mvp.DISTANCE_BINS) if lo <= d < hi), 3)
        fused['d_bin'] = mvp.DISTANCE_BIN_NAMES[fused['d_bin_idx']]
        gated.append('near_of_two')
    # 진단용: 제외된 성분도 원래 값을 남긴다 (판정에는 fused 만 쓴다)
    fused['estimates'] = {k: (round(v, 2) if v else None)
                          for k, v in (('ground', d_ground), ('size', d_size), ('depth', d_depth))}
    fused['gated'] = gated
    fused['obj_class'] = obj_class
    return fused


def near_support_rows(mask, line, H, W):
    """원본 mask 에서 line 을 따라 (±STRADDLE_TOL·W) lane 픽셀이 보이는 근거리 행 수."""
    tol = STRADDLE_TOL * W
    rows = 0
    for y in range(int(STRADDLE_NEAR_Y * H), H):
        x = mvp.line_x_at(line, y)
        x0, x1 = max(0, int(x - tol)), min(W, int(x + tol) + 1)
        if x1 > x0 and (mask[y, x0:x1] == 3).any():
            rows += 1
    return rows


def judge_lane(mask, ego_mask):
    """[차선] 차선 검증 + LDW (한 프레임). LDW 는 이 프레임에서 보이는 것만으로 정한다.

    LDW 는 '차선 위 주행' 만 L1 이다 (DECISIONS 2026-10-08). 노트북의 편향 15~30% L1 은 끄고
    비율만 기록한다. 노트북 detect_ldw 는 편향 30% 초과를 '측정 불가 (이탈 의심 / 감지 오류 의심)'
    L0 로 두는데, 이때 이탈 쪽 ego 차선이 원본 mask (바이크 영역 포함) 에서 화면 하단 근거리까지
    실제로 이어져 보이면 (STRADDLE_MIN_ROWS 이상) '차선 위 주행' L1.
    - L1 인 이유: 노트북은 한 프레임 이탈 확정이 어렵다는 이유로 이탈 L2 를 뺐다. 근거리 픽셀이
      그 확신을 일부 채우지만 한 프레임 판정이라 L1 에 둔다
    - 핸들·계기판 경계 오탐 방지: 이탈 쪽 선 자체가 검증 (VP 수렴, 기울기, 바이크 영역 밖에서
      검출) 을 통과한 선이어야 하고, 대칭 합성선이면 판정하지 않는다
    [car] LDW 는 _ldw_car (t40: 차선 영점 행 y_ref 에서만 잰다). 위 설명은 오토바이 모드
    """
    H, W = mask.shape
    vp_fixed = None
    if VEHICLE == 'car':
        # [car] 내 차 중심선 = 소실점 열을 지나는 수직선 (카메라가 차 가운데 달렸다고 본다). 소실점도 보정값으로 고정
        cal = _car_cal(W, H)
        ego_cx = float(cal.vp_x)
        if cal.horizon_ok:
            vp_fixed = (cal.vp_x, cal.horizon_y)
    elif ego_mask.sum() > 200:
        ego_cx = float(np.where(ego_mask > 0)[1].mean())
    else:
        ego_cx = None
    lane = find_lanes(mask, ego_mask, bike_x=ego_cx, vp_fixed=vp_fixed)
    if VEHICLE == 'car':
        return {'lane': lane, 'ego_cx': ego_cx, 'ldw': _ldw_car(mask, ego_mask, lane, cal)}
    ego = lane['ego']
    ldw_x = ego_cx
    vp_y = lane['vp'][1] if lane['vp'][2] >= 0.3 else int(H * 0.4)

    if not lane['ok']:   # 검증 탈락 → LDW 판정 안 함 (lane_measurable=false)
        ldw = mvp.WarningResult('LDW', 0, f'[측정 불가] {lane["reason"]}',
                                extra={'unreliable': True, 'vp_conf': round(lane['vp'][2], 2)})
        return {'lane': lane, 'ego_cx': ego_cx, 'ldw': ldw}

    ldw = mvp.detect_ldw(ego, W, H, vp_y, ego_x=ldw_x)
    ratio = ldw.extra.get('lateral_ratio')
    if ldw.level == 1:
        # 편향 비율만으로 띄우는 L1 (15~30%) 은 끈다 (DECISIONS 2026-10-08: 확실한 경우만).
        # 비율은 그대로 남겨 offset_ratio·lane_measurable 에 쓴다
        ldw = mvp.WarningResult('LDW', 0, '', extra={**ldw.extra, 'offset_only': True})
    elif ldw.level == 0 and ldw.extra.get('unreliable') and ratio is not None:
        y_check = H * 0.95
        lane_center = (mvp.line_x_at(ego['left'], y_check) + mvp.line_x_at(ego['right'], y_check)) / 2
        ex = ldw_x if ldw_x is not None else W / 2
        side = 'left' if ex < lane_center else 'right'
        side_line = ego[side]
        detected = any(side_line is l for l in lane['lines'])   # 대칭 합성선 제외
        # 근거리 근거는 원래 mask (바이크 아래로 이어지는 차선)
        support = mask if hood_top(H) >= H else lane_input_mask(mask, ego_mask)
        rows = near_support_rows(support, side_line, H, W) if detected else 0
        if rows >= STRADDLE_MIN_ROWS * H:
            direction = '좌측' if side == 'left' else '우측'
            ldw = mvp.WarningResult('LDW', 1, f'{direction} 차선 위 주행',
                                    extra={'lateral_ratio': ratio, 'straddle': True,
                                           'near_rows': rows})
    return {'lane': lane, 'ego_cx': ego_cx, 'ldw': ldw}


def _lane_trust_text(lt):
    """[car] 차선 불신 구간의 LDW 측정 불가 사유 (t44). lt = gc.lane_trust 결과."""
    r = lt['reason']
    if r == 'lane_width_out_of_range':          # 차선을 쓴 보정값으로 잰 폭 (영점 값)
        why = f'차로 폭 {lt["lane_w_m"]:.1f}m'
    elif r == 'lane_width_out_of_range_check':  # 차선 없는 수평선 (+독립 높이) 으로 다시 잰 폭
        why = f'검산 차로 폭 {lt["lane_w_m_check"]:.1f}m'
    elif r.startswith('lane_zero_failed'):
        why = '영점 실패'
    else:   # no_lane_free_horizon | y_ref_near_lane_free_horizon — 차선 없는 수평선으로 폭을 잴 수 없다
        why = '검산 불가'
    return f'차선 신뢰 낮음 ({why})'


def _ldw_car(mask, ego_mask, lane, cal):
    """[car] LDW (t40, DECISIONS 2026-10-09) — 구간의 차선 영점 행 y_ref 에서만 잰다.

    - 영점 실패 구간 (cal.lane_baseline.ldw_measurable False) 은 전부 측정 불가
    - [t44] 차선 신뢰 판정 (cal.lane_trust) 이 불신인 구간도 전부 측정 불가 ("차선 신뢰 낮음"). 영점 실패도 불신이다
    - y_ref 에서 내 차 기준 열 (소실점 열 + 평소 위치 이동 = vp_x − b/k) 좌우로 가장 가까운 검증된 선을 고르고,
      두 선이 그 행에서 실제로 보여야 잰다 (gc.line_x_measured: y_ref ± 2행의 차선 픽셀 평균 x). 안 보이면 측정 불가
      (보이지 않는 아래쪽으로 선을 연장하지 않는다)
    - 편향 비율 = |기준 열 − 차로 중심| / 차로 폭. 차선 위 주행 기준은 지금 값 그대로 (mvp.LDW_LATERAL_WARNING 30% 초과,
      두 선 모두 그 행에서 보인 실제 선이라 한쪽 근거리 픽셀 조건은 이것으로 대신한다). 15~30% 는 비율만 기록 (offset_only)
    - 내 차 폭 gc.EGO_CAR_W_M 을 y_ref 에서 px 로 바꿔 차체 옆면에서 각 선까지 간격 (m, 음수 = 차체가 선 위) 을
      extra['ref'] 에 남긴다. 판정에는 아직 안 쓴다 (기준은 민옥이 정한다)
    """
    H, W = mask.shape
    lz = getattr(cal, 'lane_baseline', None) or {}

    def unmeasurable(why, **extra):
        return mvp.WarningResult('LDW', 0, f'[측정 불가] {why}', extra={'unreliable': True, 'ref': None, **extra})

    lt = getattr(cal, 'lane_trust', None)
    if lt is not None and not lt['trusted']:
        return unmeasurable(_lane_trust_text(lt), lane_trust=lt['reason'])
    if not lz.get('ldw_measurable'):
        return unmeasurable('차선 영점 실패')
    if LDW_VARIANT in ('VB', 'VAB') and lz.get('conf') != 'high':   # [t41 VB] 영점 신뢰 낮은 구간은 경고를 내지 않는다
        return unmeasurable(f'차선 영점 신뢰 낮음 (차로 폭 {lz["lane_w_m"]:.2f}m)')
    if not lane['ok']:
        return unmeasurable(lane['reason'], vp_conf=round(lane['vp'][2], 2))
    y, k = lz['y_ref'], lz['m_per_px']
    ego_x = float(cal.vp_x + lz['ldw_shift_px'])
    left, right = gc.pick_pair_at(lane['lines'], y, ego_x)
    if left is None or right is None:
        return unmeasurable('기준 행 차선 한쪽 없음')
    tol, band, _ = gc.lane_zero_params(W, H)
    lane_bin = lane_input_mask(mask, ego_mask) == 3
    xl, xr = gc.line_x_measured(lane_bin, left, y, band, tol), gc.line_x_measured(lane_bin, right, y, band, tol)
    if xl is None or xr is None:
        return unmeasurable('기준 행에서 차선 안 보임')
    lane_w = xr - xl
    if lane_w <= 20 * W / gc.REF_W:
        return unmeasurable('차선 폭 이상')
    off = ego_x - (xl + xr) / 2
    ratio = round(abs(off) / lane_w, 3)
    half = gc.EGO_CAR_W_M / 2
    ref = {'y': int(y), 'x_left': round(xl, 1), 'x_right': round(xr, 1), 'ego_x': round(ego_x, 1),
           'ego_w_px': lz['ego_w_px'], 'lane_w_m': round(lane_w * k, 2),
           'gap_left_m': round((ego_x - xl) * k - half, 2), 'gap_right_m': round((xr - ego_x) * k - half, 2)}
    if ratio > mvp.LDW_LATERAL_WARNING:
        return mvp.WarningResult('LDW', 1, f'{"좌측" if off < 0 else "우측"} 차선 위 주행',
                                 extra={'lateral_ratio': ratio, 'straddle': True, 'ref': ref})
    return mvp.WarningResult('LDW', 0, '', extra={'lateral_ratio': ratio, 'ref': ref,
                                                  'offset_only': ratio > mvp.LDW_LATERAL_CAUTION})


def bsw_l1(objs, violators, lat_min=BSW_SIDE_LAT_MIN):
    """[BSW] L1 경로 (DECISIONS 2026-10-08). 해당 객체가 없으면 None.

    1) 차선 침범 차량 (check_lane_viol_strict / x1) — 거리와 상관없이 L1 이상
    2) 측면 나란히: lat_min ≤ |옆 간격| ≤ 1.5 × LANE_WIDTH 이고 종방향 ≤ BSW_SIDE_MAX_M
       lat_min = '내 차로' 경계 — moto 0.5 × LANE_WIDTH, car 는 X 기준 내 차로 2.5m (geom_car)
    둘 다면 침범 우선, 같은 종류 안에서는 가장 가까운 객체.
    """
    def dist(o):
        return o['dist_info']['d_fused_m']

    def side_of(o):
        lat = o.get('lateral_m')
        return '좌측' if (lat < 0 if lat is not None else o['cx_rel'] < 0.5) else '우측'

    def result(o, what, extra):
        d = dist(o)
        d_txt = f' {d:.1f}m' if d is not None else ''
        lat = o.get('lateral_m')
        return mvp.WarningResult(
            'BSW', 1, f'{side_of(o)} {what}{d_txt}', bbox=o['bbox'],
            extra={'side': side_of(o), 'distance_m': round(d, 2) if d is not None else None,
                   'lateral_m': round(lat, 2) if lat is not None else None, **extra})

    if violators:
        o = min(violators, key=lambda o: dist(o) if dist(o) is not None else 999)
        return result(o, '차선 침범', {'violating': True})
    side = [o for o in objs
            if o.get('lateral_m') is not None and dist(o) is not None
            and lat_min <= abs(o['lateral_m']) <= BSW_SIDE_LAT_MAX
            and dist(o) <= BSW_SIDE_MAX_M]
    if side:
        return result(min(side, key=dist), '측면 나란히', {'side_by_side': True})
    return None


def depth_summary(depth_map, bbox):
    """객체 박스의 Depth Pro 요약값. depth = 판정에 쓰는 박스 중앙값 (mvp.distance_from_depth 와 같음),
    나머지는 진단용 (eval_distance.py): p10/p25 = 하위 퍼센타일, center = 가운데 절반, bottom = 아래 20%."""
    if depth_map is None:
        return {'depth': None}
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    out = {'depth': mvp.distance_from_depth(depth_map, bbox, is_metric=True)}
    patch = depth_map[y0:y1 + 1, x0:x1 + 1]
    valid = patch[(patch > 0.3) & (patch < 100)]
    if valid.size >= 5:
        out['depth_p10'] = round(float(np.percentile(valid, 10)), 2)
        out['depth_p25'] = round(float(np.percentile(valid, 25)), 2)
    out['depth_center'] = _median_valid(
        depth_map[y0 + bh // 4:y1 - bh // 4 + 1, x0 + bw // 4:x1 - bw // 4 + 1])
    out['depth_bottom'] = _median_valid(
        depth_map[y1 - max(bh // 5, 1):y1 + 1, x0 + bw // 5:x1 - bw // 5 + 1])
    return out


def _scale_depth(d, k):
    """초점 고정 배율 적용 후 metric 유효 범위 (mvp.distance_from_depth 와 같은 0.3~100m) 밖이면 None."""
    if d is None:
        return None
    d = d * k
    return d if 0.3 < d < 100 else None


def _car_dist_info(g, cls, d_depth):
    """[car] geom_car.object_geom 결과 → estimate_distance 와 같은 모양의 dist_info.
    d_fused_m = 경보 거리 (바닥·차폭 중 가까운 쪽, 보닛 가림이면 보닛 경계 상한). size = 차폭 거리.
    depth 는 진단용으로만 남긴다 (판정에 안 씀, DECISIONS 2026-10-09)."""
    d = g['d']
    r2 = lambda v: round(v, 2) if v is not None else None
    idx = (next((i for i, (lo, hi) in enumerate(mvp.DISTANCE_BINS) if lo <= d < hi), 3)
           if d is not None else -1)
    return {'d_fused_m': r2(d), 'd_bin_idx': idx,
            'd_bin': mvp.DISTANCE_BIN_NAMES[idx] if idx >= 0 else 'unknown',
            'confidence': g['used'] or 'none',
            'n_valid': sum(v is not None for v in (g['ground'], g['width_used'])),
            'estimates': {'ground': r2(g['ground']), 'size': r2(g['width']), 'depth': r2(d_depth)},
            'gated': list(g['gated']), 'obj_class': cls}


def _car_objs_with_dist(scene, cal, extra_objs):
    """[car] 객체별 거리·X. 유령 (트래커가 놓친 track) 은 예측 박스 (직전 움직임으로 옮긴 박스) 의 기하로
    거리·X 를 다시 구한다 — 거리 외삽은 박스가 한 번 튀면 먼 작은 박스도 1m 로 만든다 (neg_02117).
    기하로 못 구하면 외삽 거리로 X = (cx − x_vp)·d / f. 유령 dict 의 dist_info 도 바꿔
    _hold_approaching 이 같은 거리를 쓰게 한다."""
    out = []
    for s in scene['objs']:
        raw = s['depth_raw']
        g = s['geom']
        o = {k: v for k, v in s.items() if k != 'depth_raw'}
        o.update(dist_info=_car_dist_info(g, s['cls'], raw.get('depth')), lateral_m=g['X'],
                 depth_stats={k: v for k, v in raw.items() if k != 'depth'})
        out.append(o)
    for gh in extra_objs:
        g = gc.object_geom(gh['bbox'], gh['cls'], cal)
        if g['d'] is not None:
            gh['dist_info'] = {**_car_dist_info(g, gh['cls'], None),
                               'gated': g['gated'] + ['ghost'], 'confidence': 'ghost'}
            lateral = g['X']
        else:
            g = None
            d = gh['dist_info']['d_fused_m']
            lateral = (gh['cx'] - cal.vp_x) * d / cal.f_px if d is not None else None
        out.append({**gh, 'cx_rel': 0.5 + (gh['cx'] - cal.vp_x) / cal.W, 'lateral_m': lateral,
                    'depth_stats': {}, 'geom': g})
    return out, False, cal.f_px


def _car_fcw_no_distance(fcw, front_objs, W, H):
    """[car] 거리를 못 구한 정면 객체도 화면 면적 조건 (mvp.FCW_BBOX_RATIO_CRITICAL) 으로 치명을 판단한다.

    mvp.detect_fcw 는 가장 가까운 객체 하나만 보고, 그 거리가 없으면 면적 조건까지 가지 않고 L0 으로 끝난다.
    수평선 보정 실패 (바닥 거리 없음) 와 좌우 잘림 (차폭 거리 없음) 이 겹치면 화면 절반을 덮는 박스도 L0 이
    되던 문제 (t24, 리뷰 t23). 거리 있는 객체의 판정은 그대로 두고, 거리 없는 객체 중 면적 조건을 넘는
    것이 있으면 치명으로 올린다 (내리지는 않는다). extra 에 distance_m 을 넣지 않는다 (HEAD 억제 비교용)."""
    if fcw.level >= 3:
        return fcw
    best = None
    for o in front_objs:
        if o['dist_info']['d_fused_m'] is not None:
            continue
        x0, y0, x1, y1 = o['bbox']
        ar = (x1 - x0) * (y1 - y0) / (W * H)
        if ar > mvp.FCW_BBOX_RATIO_CRITICAL and (best is None or ar > best[0]):
            best = (ar, o)
    if best is None:
        return fcw
    ar, o = best
    return mvp.WarningResult('FCW', 3, f'정면 거리 측정 불가 (화면 {ar * 100:.0f}%)', bbox=o['bbox'],
                             extra={'area_ratio': round(ar, 3), 'cause': 'area_no_distance'})


def _car_direction(o, cal):
    """[car] |X| ≤ 2.5m 면 정면 (내 차로), 아니면 X 부호로 좌우. X 를 못 구하면 소실점 열 ± 15%W 띠."""
    X = o.get('lateral_m')
    if X is not None:
        return 'front' if abs(X) <= gc.EGO_LANE_HALF_M else ('left' if X < 0 else 'right')
    off = o['cx'] - cal.vp_x
    return 'front' if abs(off) <= 0.15 * cal.W else ('left' if off < 0 else 'right')


def judge(mask, det_objs, depth_map, focal_px, depth_is_metric=True):
    """mask (차선·ego) + 검출 박스 + depth → 판정. 반환 dict 는 assess_v5 와 같은 키를 쓴다.
    = judge_scene (초점과 무관한 부분) + judge_with_focal (거리·경고). depth 는 metric (Depth Pro) 전제."""
    return judge_with_focal(judge_scene(mask, det_objs, depth_map), focal_px)


def judge_scene(mask, det_objs, depth_map):
    """1단계 — 초점거리·거리와 무관한 판정 재료. 두 번 훑기의 2회차는 이 결과만으로 다시 판정한다.

    차선/VP/LDW(한 프레임), ego 제거, corridor·zone, 차선 침범 여부, 객체별 Depth Pro 요약값.
    mask·depth 지도처럼 큰 배열은 남기지 않는다 (corridor 도 다각형만).
    """
    H, W = mask.shape
    ego_mask = ego_mask_for(mask)   # car 모드는 빈 마스크 (ego 역할은 보닛 띠)

    # 1. [차선] Lane → VP → Ego lane, 검증 통과한 차선만 사용 (judge_lane / find_lanes)
    jl = judge_lane(mask, ego_mask)
    lane = jl['lane']
    lines = lane['lines']
    vp_x, vp_y, vp_conf = lane['vp']
    if vp_conf < 0.3:
        vp_x, vp_y = W // 2, int(H * 0.4)
    ego = lane['ego']
    cal = None
    if VEHICLE == 'car':
        # [car] 구간 보정값 (수평선·소실점 열) 으로 고정. 보정 실패면 신뢰도 0 → 레코드 vp=null
        cal = _car_cal(W, H)
        vp_x, vp_y, vp_conf = cal.vp_x, cal.horizon_y, cal.conf_num

    # 3. [YOLO] 객체 = 검출 박스. ego (내 바이크·라이더) 제거 규칙 2개 중 하나라도 맞으면 제거
    #    (a) 마스크 기준 (노트북 그대로): 박스 면적의 55% 초과 (작은 박스 40%) 가 ego 마스크
    #        → 사이드미러에 비친 차, 라이더 무릎·손 같은 조각
    #    (b) 박스 기준 (YOLO 추가): motorcycle/person/bicycle 박스가 ego 마스크의 45% 이상을 덮음
    #        → YOLO 가 내 바이크 전체를 하나의 큰 박스로 잡은 경우. (a) 로는 박스가 마스크보다
    #          훨씬 커서 안 걸린다. 임시 영상 실측: 내 바이크 박스 0.48~0.98, 다른 객체 ≤ 0.32.
    #          car/truck/bus 는 제외 — 바로 앞 근접 차량을 지우는 위험을 피한다
    raw_moveables = [o for o in det_objs if o.get('source', 'yolo') == 'yolo']
    lowconf_cands = [o for o in det_objs if o.get('source') == 'yolo_lowconf']
    ego_area = int(ego_mask.sum())

    def is_ego_fragment(o):
        if in_hood(o['bbox'], H):   # car 모드: 보닛 띠에 주로 걸친 박스 = 내 차 보닛·반사
            return True
        if hood_cover(o['bbox'], W, H) >= EGO_COVER_MIN:   # car 모드: 보닛 띠를 넓게 덮는 박스 = 내 차 (t20)
            return True
        if interior_frac(o['bbox'], W, H) >= gc.INT_BOX_FRAC:   # car 모드: 박스 절반 이상이 차 실내 (t27)
            return True
        if ego_area == 0: return False
        x0, y0, x1, y1 = o['bbox']
        patch = ego_mask[y0:y1+1, x0:x1+1]
        overlap = int(patch.sum())
        bbox_px = max((x1 - x0) * (y1 - y0), 1)
        thr = 0.55 if (x1 - x0) * (y1 - y0) >= 3500 else 0.40
        if overlap / bbox_px > thr:
            return True
        return o['cls'] in EGO_LIKE_CLS and overlap / ego_area >= EGO_COVER_MIN

    ego_flags = [is_ego_fragment(o) for o in raw_moveables]
    moveables = [o for o, e in zip(raw_moveables, ego_flags) if not e]
    n_filtered = len(raw_moveables) - len(moveables)
    # [car] 실내 (보닛 띠·실내 마스크) 로 뺀 검출 박스는 objects 에 interior=true 로 기록만 한다 (t27)
    interior_objs = [o for o, e in zip(raw_moveables, ego_flags) if e] if cal is not None else None
    # (DA V2 용 calibrate_depth_alpha 는 metric depth (Depth Pro) 에서 아무 일도 하지 않아 뺐다)

    # [t15] 놓친 박스 되살리기 1·2겹 (3겹 유령은 Engine.finish_frame)
    rescued = (rescue_lowconf(lowconf_cands, moveables, ego_mask, W, H, removed=interior_objs)
               if RESCUE_LOWCONF else [])
    masked = mask_boxes(mask, ego_mask, moveables + rescued, W, H, vp_y) if MASK_BOXES else []
    moveables = moveables + rescued + masked

    # 4. Zone corridor — [car] 픽셀 차선 대신 X 기준 내 차로 (|X| ≤ 2.5m)
    corridor = None
    if cal is not None:
        corridor = _car_corridor(cal)
    elif ego['left'] is not None and ego['right'] is not None and vp_conf >= 0.3:
        y_bot = H - 1
        Lx_bot = mvp.line_x_at(ego['left'], y_bot)
        Rx_bot = mvp.line_x_at(ego['right'], y_bot)
        corridor = mvp.build_corridor(vp_x, vp_y, Lx_bot, Rx_bot, H, W,
                                      vp_conf=vp_conf)

    # 5-6 중 초점과 무관한 부분: 객체별 zone, 차선 침범 여부, Depth Pro 요약값
    #      되살린 객체 (aux) 는 차선 침범 판정에서 뺀다 — 거리 기반 근접 판정에만 쓴다
    #      [car] 기하 (거리 두 가지, X) 를 여기서 계산하고, 차로 침범은 픽셀 차선 대신 X 기준 (geom_car.cutin_side).
    #            cx_rel 은 소실점 열 기준 (0.5 = 소실점 열) — BSW 좌우, HEAD 정면 판단이 소실점을 축으로 쓴다
    objs = []
    for o in moveables:
        zone_sev, zone_info = mvp.classify_zone(o['bbox'], corridor)
        aux = o.get('aux', False)
        if cal is not None:
            g = gc.object_geom(o['bbox'], o['cls'], cal)
            strict = bool(not aux and gc.cutin_side(g))
            x1 = False
            cx_rel = 0.5 + (o['cx'] - cal.vp_x) / W
        else:
            g = None
            strict = bool(not aux and ego['left'] and ego['right']
                          and mvp.check_lane_viol_strict(o['bbox'], ego, H))
            x1 = bool(not aux and mvp.check_lane_viol_x1(o['bbox'], W))
            cx_rel = o['cx'] / W
        objs.append({
            'bbox': o['bbox'], 'class_id': o['class_id'],
            'cls': o['cls'], 'conf': o['conf'],
            'cx': o['cx'], 'cy': o['cy'], 'cx_rel': cx_rel,
            'zone_sev': zone_sev, 'zone_info': zone_info,
            'strict': strict,
            'x1': x1,
            'depth_raw': depth_summary(depth_map, o['bbox']),
            'source': o.get('source', 'yolo'), 'aux': aux,
            **({'geom': g} if cal is not None else {}),
        })

    return {
        'H': H, 'W': W, 'ldw': jl['ldw'], 'lane': lane, 'lines': lines, 'ego': ego,
        'vp': (vp_x, vp_y, vp_conf), 'vp_default': vp_conf < 0.3,
        'corridor': None if corridor is None else {'polygons': corridor['polygons']},
        'objs': objs, 'n_filtered': n_filtered,
        **({'car_cal': cal, 'interior_objs': interior_objs} if cal is not None else {}),
    }


def judge_with_focal(scene, focal_px, depth_scale=1.0, extra_objs=()):
    """2단계 — 거리와 경고. focal_px 로 ground/size 를 계산하고 depth 는 depth_scale 배로 쓴다.

    초점 고정 (run_video 2회차): focal_px = 구간 f_fixed, depth_scale = f_fixed / 그 프레임 초점거리.
    Depth Pro 의 미터 깊이는 자기가 추정한 초점거리에 비례하므로 같은 배율로 보정된다.
    extra_objs: [t15] 트래커 유령 객체 (Engine._ghost_objects) — 예측 박스와 예측 거리
    (dist_info) 를 이미 가지고 있어 거리 추정을 건너뛴다.
    """
    H, W = scene['H'], scene['W']
    vp_x, vp_y, vp_conf = scene['vp']
    ego, lines, corridor = scene['ego'], scene['lines'], scene['corridor']
    n_filtered = scene['n_filtered']
    car_cal = scene.get('car_cal')

    if car_cal is not None:
        # [car] 거리·X 는 judge_scene 의 기하 (geom_car.object_geom). Depth Pro 원값과 초점거리는 쓰지 않는다
        objs_with_dist, f_ok, f_px_used = _car_objs_with_dist(scene, car_cal, extra_objs)
        front_objs, left_objs, right_objs = [], [], []
        for o in objs_with_dist:
            {'front': front_objs, 'left': left_objs, 'right': right_objs}[_car_direction(o, car_cal)].append(o)
        bsw_lat_min = gc.EGO_LANE_HALF_M
    else:
        # 2. Distance estimator init
        est = mvp.DistanceEstimator(depth_is_metric=True)
        if focal_px is not None and focal_px > 0:
            est.f_px = float(focal_px)
            f_ok = True
        else:
            f_ok = est.calibrate(lines, vp_y, W, H)
        f_px_used = est.f_px

        # 5. Per-object distance + [BSW] 옆 간격
        #    lateral ≈ (박스 하단 중심 x − vp_x) × Z / f  (왼쪽 음수). VP 가 없으면 기본 VP (W/2) 로
        #    계산하고, 그 사실은 레코드의 vp=null 로 남는다
        objs_with_dist = []
        for s in scene['objs']:
            raw = s['depth_raw']
            info = estimate_distance(est, s, _scale_depth(raw.get('depth'), depth_scale),
                                     vp_y, scene['vp_default'], W, H)
            d = info['d_fused_m']
            lateral = (s['cx'] - vp_x) * d / est.f_px if (d is not None and est.f_px) else None
            o = {k: v for k, v in s.items() if k != 'depth_raw'}
            o.update(dist_info=info, lateral_m=lateral,
                     depth_stats={k: (round(v * depth_scale, 2) if v is not None else None)
                                  for k, v in raw.items() if k != 'depth'})
            objs_with_dist.append(o)
        for g in extra_objs:
            d = g['dist_info']['d_fused_m']
            lateral = (g['cx'] - vp_x) * d / est.f_px if (d is not None and est.f_px) else None
            objs_with_dist.append({**g, 'cx_rel': g['cx'] / W, 'lateral_m': lateral,
                                   'depth_stats': {}})

        # 6. 방향별 분류
        front_objs = [o for o in objs_with_dist if 0.35 <= o['cx_rel'] <= 0.65]
        left_objs = [o for o in objs_with_dist if o['cx_rel'] < 0.35]
        right_objs = [o for o in objs_with_dist if o['cx_rel'] > 0.65]
        bsw_lat_min = BSW_SIDE_LAT_MIN
    # violation (침범 여부는 judge_scene 에서 계산. car 는 X 기준 침범만, x1 없음)
    strict_violators = [o for o in objs_with_dist if o['strict']]
    x1_violators = [o for o in objs_with_dist if o['x1']]

    # 7-8. 4 warnings — LDW 는 judge_lane 에서 (ego_x = ego_mask 무게중심, 차선 위 주행 포함)
    fcw = mvp.detect_fcw(front_objs, W, H)
    if car_cal is not None:
        fcw = _car_fcw_no_distance(fcw, front_objs, W, H)
    ldw = scene['ldw']
    # [BSW] 노트북 detect_bsw (측면 띠 객체: <3m+침범 L3, <3m L2, <5m L1) 에 L1 경로 2개를 더한다
    violators = strict_violators + x1_violators
    bsw = mvp.detect_bsw(left_objs, right_objs, strict_violators, x1_violators)
    if bsw.level == 0:
        bsw = bsw_l1(objs_with_dist, violators, bsw_lat_min) or bsw
    head = mvp.detect_head(front_objs, strict_violators, x1_violators, W, H)

    # 9. Zone clamp patches
    zone_adjustments = []
    if corridor is not None:
        cut_in_scenario = (bsw.level >= 2 or head.level >= 2)
        if (fcw.level == 3 and fcw.extra.get('area_ratio', 0) > 0.25
                and fcw.bbox is not None and not cut_in_scenario):
            matching = next(
                (o for o in objs_with_dist if tuple(o['bbox']) == tuple(fcw.bbox)),
                None)
            if matching is not None and matching['zone_sev'] < 2:
                fcw = mvp.WarningResult(
                    'FCW', 2, fcw.reason + ' [zone clamp→L2]',
                    bbox=fcw.bbox,
                    extra={**fcw.extra, 'zone_clamped': True,
                           'zone_sev': matching['zone_sev']})
                zone_adjustments.append('FCW L3→L2 (zone clamp, no cut-in)')
        # (노트북의 BSW sidelobe boost — 측면 구역 겹침만으로 L0→L2 — 는 폐지. DECISIONS 2026-10-08)

    # 10. Distance summary
    def closest_d(lst):
        ds = [o['dist_info']['d_fused_m'] for o in lst
              if o['dist_info']['d_fused_m'] is not None]
        return min(ds) if ds else None

    state = mvp.V5State(
        fcw=fcw, ldw=ldw, bsw=bsw, head=head,
        d_front_m=closest_d(front_objs),
        d_left_m=closest_d(left_objs),
        d_right_m=closest_d(right_objs),
        f_calibrated=f_ok,
        f_px=round(f_px_used, 1) if f_px_used else None,
    )

    # 2m 이내 억제는 "내 바이크·라이더·미러를 객체로 잘못 잡았다" 는 가정. car 모드는 그 근거가 없고
    # 보닛 띠 박스를 이미 뺐으므로 적용하지 않는다 (2m 안 실제 객체를 지우지 않게 — 미탐 우선)
    if VEHICLE != 'car':
        mvp.suppress_short_range_warnings(state)
    if state.head.level > 0:
        fcw_ok = state.fcw.level > 0 and (
            state.fcw.extra and state.fcw.extra.get('distance_m', 999) <= 4.5)
        if not fcw_ok:
            state.head.level = 0
            state.head.reason = '[억제] FCW 동시 활성 조건 미충족 (역주행 확신 부족)'
            state.head.bbox = None
    # [BSW] 차선 침범 차량은 무조건 L1 이상 — 2m 억제 등으로 L0 이 됐어도 다시 올린다
    if state.bsw.level == 0 and violators:
        state.bsw = bsw_l1(objs_with_dist, violators, bsw_lat_min)

    return {
        'vp': (int(vp_x), int(vp_y), float(vp_conf)),
        'n_ego_fragments_filtered': n_filtered,
        'zone_adjustments': zone_adjustments,
        '_state': state,
        '_ego': ego,
        '_lines': lines,
        '_objs': objs_with_dist,
        '_interior_objs': scene.get('interior_objs') or [],
        '_corridor': corridor,
        '_fronts': {id(o) for o in front_objs},
        '_lefts': {id(o) for o in left_objs},
        # 진단 기록용 (--dump-estimates). 판정에는 쓰지 않는다
        '_strict_ids': {id(o) for o in strict_violators},
        '_x1_ids': {id(o) for o in x1_violators},
    }


# ============================================================
# 판정 결과 → 스키마 레코드
# ============================================================
_DEBUG_TAG_RE = re.compile(r'\s*\[zone [^\]]*\]')
_REASON_KO = [
    ('bbox ', '화면 '),
    ('ego_lane 식별 실패', '[측정 불가] 차선 없음'),
    ('ego_lane 너비 이상', '[측정 불가] 차선 폭 이상'),
]


def user_reason(w):
    """화면에 보일 사유 (한국어). 판정 내부 문구 (영어 태그, 억제·생략 사유) 는 _state 에만 남는다.

    - FCW/BSW/HEAD 가 L0 이면 빈 문자열 (cut-in 가능성, [억제] … 같은 디버그 문구)
    - [zone clamp→L2] 같은 태그 제거, bbox 같은 영어 단어 한국어로
    - HEAD 의 (aspect …) 같은 수치 꼬리 제거
    """
    if w.name != 'LDW' and w.level == 0:
        return ''
    r = _DEBUG_TAG_RE.sub('', w.reason or '')
    for en, ko in _REASON_KO:
        r = r.replace(en, ko)
    if w.name == 'HEAD':
        r = re.sub(r'\s*\(aspect [^)]*\)', '', r)
    return r.strip()


def to_record(res, H, track_ids, frame_idx, t_sec):
    state = res['_state']
    objs = res['_objs']
    vp_x, vp_y, vp_conf = res['vp']

    def tid_of(bbox):
        if bbox is None: return None
        return track_ids.get(tuple(bbox))

    # 객체별 경고 이름 — 같은 박스에 여러 경고면 높은 레벨, 동률이면 HEAD > FCW > BSW
    warn_by_bbox = {}
    for w in (state.head, state.fcw, state.bsw):
        if w.level > 0 and w.bbox is not None:
            key = tuple(w.bbox)
            if key not in warn_by_bbox or w.level > warn_by_bbox[key].level:
                warn_by_bbox[key] = w

    fcw, ldw, bsw, head = state.fcw, state.ldw, state.bsw, state.head
    ldw_ratio = ldw.extra.get('lateral_ratio') if ldw.extra else None
    side = DIRECTION_EN.get(bsw.extra.get('side')) if (bsw.level > 0 and bsw.extra) else None
    warnings = {
        'FCW': {'level': fcw.level, 'reason': user_reason(fcw),
                'track_id': tid_of(fcw.bbox) if fcw.level > 0 else None,
                'distance_m': _r1(fcw.extra.get('distance_m')) if fcw.level > 0 else None},
        'LDW': {'level': ldw.level, 'reason': user_reason(ldw),
                'offset_ratio': ldw_ratio},
        'BSW': {'level': bsw.level, 'reason': user_reason(bsw),
                'track_id': tid_of(bsw.bbox) if bsw.level > 0 else None,
                'side': side},
        'HEAD': {'level': head.level, 'reason': user_reason(head),
                 'track_id': tid_of(head.bbox) if head.level > 0 else None},
    }

    ego = res['_ego']
    lanes = None
    if ego['left'] is not None and ego['right'] is not None:
        # 선분은 VP 높이 ~ 화면 하단. 두 선이 VP 보다 아래에서 만나면 교차점부터 (X 자 방지)
        y_top, y_bot = max(vp_y, 0), H - 1
        dd = ego['left']['dydx'] - ego['right']['dydx']
        if abs(dd) > 1e-6:
            y_cross = (ego['right']['c'] - ego['left']['c']) / dd
            if y_top < y_cross < y_bot:
                y_top = y_cross
        lanes = {side_: [int(round(mvp.line_x_at(ego[side_], y_top))), int(y_top),
                         int(round(mvp.line_x_at(ego[side_], y_bot))), int(y_bot)]
                 for side_ in ('left', 'right')}

    corridor = None
    if res['_corridor'] is not None:
        polys = res['_corridor']['polygons']
        corridor = {k: polys[k].tolist() for k in ('critical', 'danger', 'caution')}

    car = res.get('_car')   # [car] 자동차 모드 추가 필드 (frame_schema.md 의 자동차 모드 절)
    r2 = lambda v: round(float(v), 2) if v is not None else None
    if car is not None:
        warnings['FCW']['cause'] = ((fcw.extra or {}).get('cause')
                                    or ('ghost_hold' if (fcw.extra or {}).get('ghost_hold') else 'distance')
                                    if fcw.level > 0 else None)
        # FCW 대상 객체의 TTC (기록만 — 판정에 안 씀). 두 방법이 모두 있을 때 큰 값
        tgt = (next((o for o in objs if fcw.bbox is not None and tuple(o['bbox']) == tuple(fcw.bbox)), None)
               if fcw.level > 0 else None)
        t = (tgt or {}).get('ttc') or {}
        warnings['FCW']['ttc_s'] = (r2(max(t['scale_s'], t['ground_s']))
                                    if t.get('scale_s') is not None and t.get('ground_s') is not None else None)
        # LDW 를 잰 위치 (t40: 차선 영점 행 y_ref). 측정 불가면 null
        warnings['LDW']['ref'] = (ldw.extra or {}).get('ref')

    objects = []
    for o in objs:
        if id(o) in res['_fronts']: direction = 'front'
        elif id(o) in res['_lefts']: direction = 'left'
        else: direction = 'right'
        w = warn_by_bbox.get(tuple(o['bbox']))
        item = {
            'track_id': tid_of(o['bbox']),
            'cls': o['cls'],
            'conf': round(o['conf'], 2),
            'bbox': [int(v) for v in o['bbox']],
            'distance_m': _r1(o['dist_info']['d_fused_m']),
            'lateral_m': _r1(o['lateral_m']),
            'zone_sev': int(o['zone_sev']),
            'direction': direction,
            'warning': w.name if w is not None else None,
            'source': o.get('source', 'yolo'),   # yolo | yolo_lowconf | mask | ghost
            'ghost': bool(o.get('ghost', False)),
        }
        if car is not None:
            g, t = o.get('geom'), o.get('ttc')
            item['in_lane'] = None if o['lateral_m'] is None else bool(abs(o['lateral_m']) <= gc.EGO_LANE_HALF_M)
            item['dist_geo'] = None if g is None else {
                'ground_m': _r1(g['ground']), 'width_m': _r1(g['width']),
                'width_implied_m': r2(g['width_implied_m']), 'used': g['used'], 'gated': g['gated']}
            item['ttc'] = None if not t else {'scale_s': r2(t.get('scale_s')), 'ground_s': r2(t.get('ground_s'))}
            item['interior'] = False
        objects.append(item)
    if car is not None:
        # 차 실내로 판정에서 뺀 검출 박스 — 기록만 (거리·경고 없음, 트래커에도 안 넣음)
        for o in res.get('_interior_objs', []):
            b = [int(v) for v in o['bbox']]
            off = (b[0] + b[2]) / 2 - vp_x
            objects.append({
                'track_id': None, 'cls': o['cls'], 'conf': round(o['conf'], 2), 'bbox': b,
                'distance_m': None, 'lateral_m': None, 'zone_sev': 0,
                'direction': 'front' if abs(off) <= 0.15 * car['W'] else ('left' if off < 0 else 'right'),
                'warning': None, 'source': o.get('source', 'yolo'), 'ghost': False,
                'in_lane': None, 'dist_geo': None, 'ttc': None, 'interior': True,
            })

    top = state.top_warning
    extra_fields = {}
    if car is not None:
        extra_fields = {'calib_seg': car.get('calib_seg'), 'depth_guard': car.get('depth_guard')}
    return {
        'frame_idx': frame_idx,
        't_sec': t_sec,
        'bin': state.max_severity,
        'bin_label': state.bin,
        'top_warning': top.name if top else None,
        'top_reason': user_reason(top) if top else '',
        'warnings': warnings,
        # LDW 가 편향 비율을 계산했고 '측정 불가' 판정이 아닐 때만 True
        'lane_measurable': bool(ldw_ratio is not None and not ldw.extra.get('unreliable')),
        'lanes': lanes,
        # VP 신뢰도 0.3 미만이면 판정은 기본 VP (W/2, 0.4H) 로 진행하지만 레코드에는 null
        'vp': [vp_x, vp_y, round(vp_conf, 3)] if vp_conf >= 0.3 else None,
        'corridor': corridor,
        'objects': objects,
        'd_front_m': _r1(state.d_front_m),
        'd_left_m': _r1(state.d_left_m),
        'd_right_m': _r1(state.d_right_m),
        **extra_fields,
    }


# ============================================================
# 진단 기록 — 거리 추정 방식 비교용 (--dump-estimates). 판정·frames.jsonl 과 무관
# ============================================================
def _median_valid(a):
    v = a[(a > 0.3) & (a < 100)]
    return round(float(np.median(v)), 2) if v.size >= 5 else None


def estimates_record(res, record, track_ids, focal_info, W, H, lock=None):
    """프레임 1장의 객체별 거리 추정값 (estimates.jsonl 한 줄).

    ground / size / depth 는 성분 원래 값 (제외됐어도 남긴다, 'gated' 에 제외 사유),
    fused 는 판정에 실제 쓰인 값. depth 는 박스 전체 중앙값 (distance_from_depth).
    비교용 Depth Pro 요약값 (모두 초점 고정 배율 적용 후):
      depth_p10 / depth_p25 — 박스 전체 하위 10·25 퍼센타일
      depth_center — 박스 가운데 절반 (가로·세로 25~75%) 중앙값
      depth_bottom — 박스 아래 20% 행 × 가로 20~80% 중앙값 (차 범퍼·바퀴 쪽)
    lock: 초점 고정 정보 {'segment', 'f_fixed', 'f_frame', 'depth_scale'} (고정 안 하면 None)
    """
    state = res['_state']
    vp_x, vp_y, vp_conf = res['vp']
    if res.get('_car') is not None:
        f_src = 'car_geom_prior'          # [car] 초점거리 prior (geom_car.F_REF_PX), Depth Pro 초점 안 씀
    elif focal_info.get('focal_fallback'):
        f_src = 'depth_pro_fallback_0.8W'
    elif focal_info:
        f_src = 'depth_pro_locked' if lock else 'depth_pro'
    else:
        f_src = 'lane' if state.f_calibrated else 'default_1.2W'
    objs = []
    for o in res['_objs']:
        x0, y0, x1, y1 = o['bbox']
        bw, bh = x1 - x0, y1 - y0
        est = o['dist_info']['estimates']
        objs.append({
            'track_id': track_ids.get(tuple(o['bbox'])),
            'cls': o['cls'], 'conf': round(o['conf'], 3), 'bbox': [int(v) for v in o['bbox']],
            'cx': int(o['cx']), 'cx_rel': round(o['cx_rel'], 4),
            'area_ratio': round(bw * bh / (W * H), 4),
            'zone_sev': int(o['zone_sev']),
            'source': o.get('source', 'yolo'), 'ghost': bool(o.get('ghost', False)),
            'strict': o['strict'], 'x1': o['x1'],
            'obj_class': o['dist_info'].get('obj_class'),
            'ground': est.get('ground'), 'size': est.get('size'), 'depth': est.get('depth'),
            'gated': o['dist_info'].get('gated', []),
            'fused': o['dist_info']['d_fused_m'],
            'confidence': o['dist_info'].get('confidence'),
            'lateral_m': round(o['lateral_m'], 2) if o['lateral_m'] is not None else None,
            **o['depth_stats'],
            **({'geom_used': (o.get('geom') or {}).get('used'),
                'width_implied_m': (round(o['geom']['width_implied_m'], 3)
                                    if o.get('geom') and o['geom']['width_implied_m'] is not None else None),
                'ttc': o.get('ttc')} if res.get('_car') is not None else {}),
        })
    return {
        'frame_idx': record['frame_idx'], 't_sec': record['t_sec'], 'W': W, 'H': H,
        'f_px': state.f_px, 'f_src': f_src, 'focal_raw': focal_info.get('focal_raw'),
        'f_frame': focal_info.get('focal'),
        'lock': lock,
        'vp_used': [int(vp_x), int(vp_y), round(float(vp_conf), 3)],
        'vp_default': bool(vp_conf < 0.3),
        'corridor_active': res['_corridor'] is not None,
        'levels': {k: record['warnings'][k]['level'] for k in ('FCW', 'LDW', 'BSW', 'HEAD')},
        'bin': record['bin'],
        'objects': objs,
    }


# ============================================================
# Engine
# ============================================================
class Engine:
    """모델 3종을 한 번 로드하고 프레임을 반복 판정한다. 트래커 상태를 가진다 (영상 1개당 reset)."""

    def __init__(self, lane_ckpts=LANE_CKPTS, det_weights=DET_WEIGHTS,
                 device='cuda', fps_eval=5.0, n_folds=None, det_conf=DET_CONF,
                 dump_estimates=False, log=print):
        from ultralytics import YOLO
        self.dump_estimates = dump_estimates   # True 면 assess_frame 마다 last_estimates 를 채운다
        self.last_estimates = None
        self.device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        if device == 'cuda' and self.device != 'cuda':
            log('[engine] CUDA 미감지 → CPU 사용 (느림)')
        lane_ckpts = list(lane_ckpts)[:n_folds] if n_folds else list(lane_ckpts)
        self.det_conf = det_conf

        t0 = time.time()
        self.lane_models = self.yolop = None
        if VEHICLE == 'car':
            # [car] 차선·주행 가능 영역은 YOLOPv2 (t29, DECISIONS 2026-10-09). lane_focus 는 오토바이 전용
            self.yolop = car_models.load_yolopv2(self.device)
            log(f'[engine] YOLOPv2 로드 ({time.time() - t0:.1f}s)')
        else:
            self.lane_models = load_lane_models(lane_ckpts, self.device)
            log(f'[engine] 차선 세그 {len(self.lane_models)} fold 로드 ({time.time() - t0:.1f}s)')
        t0 = time.time()
        self.det_model = YOLO(str(det_weights))
        log(f'[engine] YOLO 로드: {Path(det_weights).name} ({time.time() - t0:.1f}s)')
        t0 = time.time()
        self.depth_proc, self.depth_model = load_depth_pro(self.device)
        log(f'[engine] Depth Pro 로드 ({time.time() - t0:.1f}s)')

        self.models_info = {
            'lane': (f'YOLOPv2 ({car_models.YOLOP_WEIGHTS.name}, lane + drivable area only)' if self.yolop is not None
                     else f'lane_focus U-Net R34 {len(self.lane_models)}-fold majority vote '
                          f'({", ".join(Path(p).name for p in lane_ckpts)})'),
            'det': f'{Path(det_weights).name} (ultralytics, conf {det_conf}, '
                   f'classes {"/".join(DET_CLASSES.values())})',
            'depth': f'{DEPTH_MODEL} ({"fp16" if self.device == "cuda" else "fp32"})',
        }
        self.tracker = SimpleIouTracker(
            iou_threshold=TRACKER_IOU,
            max_ghost_frames=max(1, round(TRACKER_GHOST_SEC * fps_eval)))
        # [t15] 트래커 유령용 track 이력: track_id → 최근 관측 2개 (박스·거리·면적·경고)
        self.track_hist = {}
        self._n_calls = 0
        self.last_timing = {}
        # [car] TTC 이력 (기록용), 근접 보험 축척 이력·직전 결과, 지금 구간 번호 (set_car_calib)
        self.ttc = gc.TtcTracker()
        self.dg_scales = []
        self.dg_last = None
        self.car_seg = None

    def reset(self):
        """새 영상 시작 시 트래커·track 이력 초기화."""
        self.tracker.reset()
        self.track_hist = {}
        self._n_calls = 0
        self.set_car_calib(None, None)

    def set_car_calib(self, cal, seg):
        """[car] 판정에 쓸 구간 보정값을 바꾼다. 구간이 바뀌면 기하가 달라지므로 TTC·근접 보험 이력을 비운다."""
        global CAR_CAL
        CAR_CAL = cal
        self.car_seg = seg
        self.ttc.reset()
        self.dg_scales = []
        self.dg_last = None

    def _sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    def assess_frame(self, bgr_image, frame_idx=None, t_sec=None):
        """BGR 프레임 1장 → frame_schema 레코드 dict.

        frame_idx / t_sec 를 안 주면 frame_idx 는 호출 순번, t_sec 는 None.
        단일 프레임 함수라 초점거리 고정이 없다 — 그 프레임의 Depth Pro 초점거리를 그대로 쓴다.
        영상 단위 고정 (두 번 훑기) 은 run_video 에서만 한다.
        """
        scene, focal_info, tm = self.analyze_frame(bgr_image)
        t = time.perf_counter()
        record = self.finish_frame(scene, focal_info, frame_idx, t_sec)
        tm['logic'] += time.perf_counter() - t
        tm['total'] = sum(v for k, v in tm.items() if k != 'total')
        self.last_timing = tm
        return record

    def seg_mask(self, bgr_image, rgb=None):
        """차선 세그 마스크 (lane_focus 7-class 형식). car 는 YOLOPv2 → 2 주행 가능 영역, 3 차선, 0 나머지
        (Moveable·내 바이크 클래스가 없어 2겹 마스크 보조 박스는 car 모드에서 생기지 않는다)."""
        if self.yolop is not None:
            return car_models.yolop_mask(self.yolop, bgr_image, self.device)
        if rgb is None:
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return predict_lane_mask(self.lane_models, rgb, self.device)

    def models_lane_det(self, bgr_image, rgb=None):
        """차선 세그 + YOLO (보정 단계와 본 판정이 같이 쓴다). Returns (mask, det_objs)."""
        mask = self.seg_mask(bgr_image, rgb)
        det_objs = detect_objects(self.det_model, bgr_image, self.device, self.det_conf,
                                  lowconf=RESCUE_CONF_MIN if RESCUE_LOWCONF else None)
        return mask, det_objs

    def analyze_frame(self, bgr_image, cached=None, run_depth=True):
        """모델 3종 + judge_scene (초점과 무관한 판정 재료). Returns (scene, focal_info, timings).
        focal_info = {'focal': 판정용 초점거리, 'focal_raw': 모델 원값, 'focal_fallback': bool}

        [car] cached = 보정 단계에서 이미 돌린 (mask, det_objs) — 차선 세그·YOLO 를 다시 돌리지 않는다.
              run_depth=False 면 Depth Pro 를 건너뛴다 (depth=None). Depth Pro 를 돌린 프레임은
              박스 없는 근접 물체 보험 (gc.depth_guard) 결과를 scene['depth_guard'] 에 둔다.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tm = {}

        if cached is not None:
            mask, det_objs = cached
            tm['lane'] = tm['det'] = 0.0
        else:
            t = time.perf_counter()
            mask = self.seg_mask(bgr_image, rgb)
            self._sync(); tm['lane'] = time.perf_counter() - t

            t = time.perf_counter()
            det_objs = detect_objects(self.det_model, bgr_image, self.device, self.det_conf,
                                      lowconf=RESCUE_CONF_MIN if RESCUE_LOWCONF else None)
            self._sync(); tm['det'] = time.perf_counter() - t

        t = time.perf_counter()
        focal_info = {}
        depth = None
        if run_depth:
            depth, _ = predict_depth(self.depth_proc, self.depth_model, rgb, self.device,
                                     info=focal_info,
                                     **({'floor': gc.DG_FLOOR} if VEHICLE == 'car' else {}))
            self._sync()
        tm['depth'] = time.perf_counter() - t

        t = time.perf_counter()
        scene = judge_scene(mask, det_objs, depth)
        if (VEHICLE == 'car' and depth is not None and CAR_DEPTH_GUARD
                and scene['car_cal'].horizon_conf in gc.DG_CONFS):
            scene['depth_guard'] = gc.depth_guard(depth, mask, scene['car_cal'],
                                                  [o['bbox'] for o in scene['objs']], self.dg_scales)
        tm['logic'] = time.perf_counter() - t
        return scene, focal_info, tm

    def finish_frame(self, scene, focal_info, frame_idx=None, t_sec=None, lock=None,
                     new_scene=False):
        """judge_with_focal + 트래커 → 레코드. 모델은 돌리지 않는다.

        lock = {'segment', 'f_fixed', ...} 이면 구간 고정 초점거리로 판정한다
        (focal_px = f_fixed, depth 배율 = f_fixed / 그 프레임 초점거리). None 이면 그 프레임 값.
        new_scene: 이전 프레임과 다른 장면 (사진 이어 붙인 영상의 사진 교체) — 트래커·track 이력을
        비운다. 이전 사진의 객체가 유령으로 넘어오지 않게. track_id 번호는 이어 간다.
        """
        if frame_idx is None:
            frame_idx = self._n_calls
        W, H = scene['W'], scene['H']
        if new_scene:
            self.tracker.tracks.clear()
            self.track_hist = {}

        # 트래커: 이번 프레임 검출 객체 (bbox, zone_sev) → track_id. 입력이 거리와 무관해 판정 전에 돌린다.
        # 유령은 트래커 입력에 넣지 않는다 (넣으면 스스로를 계속 이어 붙인다)
        dets = [(tuple(o['bbox']), int(o['zone_sev'])) for o in scene['objs']]
        tracks = self.tracker.update(self._n_calls, dets)
        track_ids = {tuple(tr.bbox): tr.track_id for tr in tracks if tr.ghost_count == 0}
        ghosts = self._ghost_objects(tracks, W, H) if GHOST_BOXES else []

        f_frame = focal_info.get('focal')
        if lock is not None and f_frame:
            lock = {**lock, 'f_frame': round(f_frame, 1),
                    'depth_scale': round(lock['f_fixed'] / f_frame, 4)}
            res = judge_with_focal(scene, lock['f_fixed'], lock['f_fixed'] / f_frame,
                                   extra_objs=ghosts)
        else:
            lock = None
            res = judge_with_focal(scene, f_frame, extra_objs=ghosts)
        state = res['_state']
        if VEHICLE == 'car':
            self._car_frame(res, scene, track_ids, tracks, t_sec)
        self._hold_approaching(state, ghosts)

        track_ids.update({tuple(g['bbox']): g['track_id'] for g in ghosts})
        record = to_record(res, H, track_ids, frame_idx, t_sec)
        if self.dump_estimates:
            self.last_estimates = estimates_record(res, record, track_ids, focal_info,
                                                   W, H, lock=lock)
        self._update_track_hist(res, track_ids, tracks, W, H)
        self._n_calls += 1
        return record

    # ---------------- [car] TTC 기록, 박스 없는 근접 물체 보험 → FCW ----------------
    def _car_frame(self, res, scene, track_ids, tracks, t_sec):
        """정면 TTC 를 계산해 객체에 기록하고 (판정에는 안 씀), Depth Pro 근접 보험으로 FCW 를 올린다.

        TTC (기록만, DECISIONS 2026-10-09 — 위험도는 그 순간의 장면으로 평가, 연속 프레임 TTC 는 판정에서 뺌):
             track 별 0.5초 창에서 박스 폭 증가율·바닥 위치 변화율 (gc.TtcTracker). YOLO 박스
             (conf ≥ 0.4) 만 쓰고, 화면 끝에 잘린 박스는 폭, 바닥이 가려진 박스는 바닥 방법에서 뺀다.
        근접 보험: Depth Pro 를 돌린 프레임의 gc.depth_guard 경보 → 위험 (L2). 다음 Depth Pro 프레임까지
             (DEPTH_EVERY_SEC × 1.5 안) 경보를 유지한다.
        """
        state = res['_state']
        W = scene['W']
        if CAR_TTC and t_sec is not None:
            usable = []
            for o in res['_objs']:
                if o.get('aux') or o.get('ghost') or o['conf'] < gc.TTC_MIN_CONF:
                    continue
                tid = track_ids.get(tuple(o['bbox']))
                if tid is None:
                    continue
                g = o['geom']
                w_ok = not g['side_cut'] and g['w_px'] >= gc.TTC_MIN_W_PX * W / gc.REF_W
                self.ttc.add(tid, t_sec, g['w_px'] if w_ok else None, g['dy'] if g['ground_ok'] else None)
                usable.append((o, tid))
            self.ttc.prune({tr.track_id for tr in tracks})
            for o, tid in usable:
                r = self.ttc.ttc(tid, t_sec)
                if r.get('scale_s') is not None or r.get('ground_s') is not None:
                    o['ttc'] = r
        dg = scene.get('depth_guard')
        if dg is not None:
            self.dg_last = (t_sec, dg)
        elif (self.dg_last is not None and t_sec is not None and self.dg_last[0] is not None
              and self.dg_last[1].get('status') == 'alarm'
              and t_sec - self.dg_last[0] <= DEPTH_EVERY_SEC * 1.5 + 1e-6):
            dg = {**self.dg_last[1], 'held_from_t': self.dg_last[0]}
        if dg is not None and dg.get('status') == 'alarm' and state.fcw.level < 2:
            d = dg['distance_m']
            state.fcw = mvp.WarningResult('FCW', 2, f'정면 근접 물체 {d:.1f}m (박스 없음)',
                                          extra={'distance_m': d, 'cause': 'depth_guard'})
        res['_car'] = {'calib_seg': self.car_seg, 'depth_guard': dg, 'W': W}

    # ---------------- [t15] 3겹: 트래커 유령 ----------------
    def _update_track_hist(self, res, track_ids, tracks, W, H):
        """이번 프레임에 실제로 보인 객체의 박스·거리·경고를 track 별로 최근 2개까지 남긴다."""
        state = res['_state']
        warns = [w for w in (state.fcw, state.bsw, state.head) if w.level > 0 and w.bbox is not None]
        for o in res['_objs']:
            if o.get('ghost'):
                continue
            tid = track_ids.get(tuple(o['bbox']))
            if tid is None:
                continue
            w = max((w for w in warns if tuple(w.bbox) == tuple(o['bbox'])),
                    key=lambda w: w.level, default=None)
            entry = {'n': self._n_calls, 'bbox': tuple(o['bbox']), 'd': o['dist_info']['d_fused_m'],
                     'area': _box_area(o['bbox']), 'cls': o['cls'], 'zone_sev': o['zone_sev'],
                     'warn': None if w is None else {'name': w.name, 'level': w.level,
                                                     'reason': w.reason,
                                                     'side': (w.extra or {}).get('side')}}
            self.track_hist[tid] = (self.track_hist.get(tid, []) + [entry])[-2:]
        alive = {tr.track_id for tr in tracks}
        self.track_hist = {k: v for k, v in self.track_hist.items() if k in alive}

    def _ghost_objects(self, tracks, W, H):
        """트래커가 이번 프레임에 놓친 track (ghost, 유지 시간 = TRACKER_GHOST_SEC) 을 직전 움직임으로
        예측한 박스로 만든다. 박스 = 마지막 박스 + (마지막 − 그 전) × 놓친 프레임 수.
        거리: 다가오던 track (박스가 커지거나 거리가 줄던) 은 같은 비율로 더 가깝게 외삽 (0.5m 하한),
        아니면 마지막 거리 그대로 (멀어지는 쪽으로는 외삽하지 않는다)."""
        out = []
        for tr in tracks:
            if tr.ghost_count == 0:
                continue
            h = self.track_hist.get(tr.track_id)
            if not h:
                continue
            last = h[-1]
            g = self._n_calls - last['n']
            prev = h[-2] if len(h) >= 2 and last['n'] - h[-2]['n'] == 1 else None
            vel = [a - b for a, b in zip(last['bbox'], prev['bbox'])] if prev else [0, 0, 0, 0]
            b = [int(round(last['bbox'][i] + vel[i] * g)) for i in range(4)]
            b = [int(np.clip(b[0], 0, W - 1)), int(np.clip(b[1], 0, H - 1)),
                 int(np.clip(b[2], 0, W - 1)), int(np.clip(b[3], 0, H - 1))]
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                b = list(last['bbox'])
            approaching = prev is not None and (
                last['area'] > prev['area']
                or (last['d'] is not None and prev['d'] is not None and last['d'] < prev['d']))
            d = last['d']
            if d is not None and approaching and prev['d'] is not None:
                d = max(min(d, d + (d - prev['d']) * g), 0.5)
            out.append({
                'bbox': tuple(b), 'class_id': 4, 'cls': last['cls'], 'conf': 0.0,
                'cx': int((b[0] + b[2]) / 2), 'cy': int((b[1] + b[3]) / 2),
                'zone_sev': last['zone_sev'], 'zone_info': {},
                'strict': False, 'x1': False, 'aux': True,
                'source': 'ghost', 'ghost': True, 'track_id': tr.track_id,
                'approaching': approaching, 'last_warn': last['warn'],
                'dist_info': {'d_fused_m': round(d, 2) if d is not None else None,
                              'estimates': {'ground': None, 'size': None, 'depth': None},
                              'gated': ['ghost'], 'obj_class': last['cls'],
                              'confidence': 'ghost'},
            })
        return out

    def _hold_approaching(self, state, ghosts):
        """다가오던 track 이 사라졌으면 그 track 의 직전 경고 레벨 이상을 유지한다 (미탐 우선).
        예측 거리가 2m 안쪽이라 근거리 억제 (ego 오분류 가정) 에 걸린 경우도 여기서 되살아난다."""
        for g in ghosts:
            lw = g['last_warn']
            if not (g['approaching'] and lw and lw['level'] > 0):
                continue
            attr = lw['name'].lower()
            if getattr(state, attr).level >= lw['level']:
                continue
            extra = {'distance_m': g['dist_info']['d_fused_m'], 'ghost_hold': True}
            if lw.get('side'):
                extra['side'] = lw['side']
            setattr(state, attr, mvp.WarningResult(
                lw['name'], lw['level'], f'놓친 객체 유지 — 직전 {lw["reason"]}',
                bbox=g['bbox'], extra=extra))


_DEFAULT_ENGINE = None


def assess_frame(bgr_image, frame_idx=None, t_sec=None):
    """모듈 함수 버전 — 처음 호출 때 기본 설정 Engine 을 만든다."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = Engine()
    return _DEFAULT_ENGINE.assess_frame(bgr_image, frame_idx, t_sec)


# ============================================================
# CLI — 영상 처리
# ============================================================
def _gpu_used_mb():
    """nvidia-smi 기준 GPU 전체 사용량 (MB). 실패하면 None."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (int(v) for v in out.split(','))
        return used, total
    except Exception:
        return None


def _cut_thumb(bgr):
    """장면 전환 감지용 썸네일: 64×36 회색, 평균 0 · 노름 1 (밝기·대비 변화에 둔감)."""
    g = cv2.cvtColor(cv2.resize(bgr, (64, 36), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def focal_segments(eval_items, cut_frames):
    """평가 프레임을 장면 전환 기준으로 구간으로 나누고 구간별 고정 초점거리 (중앙값) 를 정한다.

    eval_items: [(frame_idx, focal)], cut_frames: 직전 원본 프레임과 이어지지 않는 원본 프레임 번호.
    두 평가 프레임 사이 (a, b] 에 전환이 하나라도 있으면 b 부터 새 구간.
    Returns [{'start_frame', 'end_frame', 'n_eval', 'f_fixed', 'members': [k ...]}]
    """
    cuts = sorted(cut_frames)
    segs, cur = [], [0]
    for k in range(1, len(eval_items)):
        a, b = eval_items[k - 1][0], eval_items[k][0]
        lo = np.searchsorted(cuts, a, side='right')   # a < cut <= b ?
        if lo < len(cuts) and cuts[lo] <= b:
            segs.append(cur)
            cur = [k]
        else:
            cur.append(k)
    if eval_items:
        segs.append(cur)
    out = []
    for m in segs:
        fs = [eval_items[k][1] for k in m if eval_items[k][1]]
        out.append({'start_frame': eval_items[m[0]][0], 'end_frame': eval_items[m[-1]][0],
                    'n_eval': len(m), 'f_fixed': round(float(np.median(fs)), 1) if fs else None,
                    'members': m})
    return out


# ============================================================
# [car] 보정 단계 (t20, DECISIONS 2026-10-09)
# ============================================================
def calibrate_car(engine, video_path, fps_src, fps_eval, W, H, log=print):
    """판정 전에 영상을 한 번 훑는다. 장면 전환 (CUT_NCC) 으로 구간을 나누고, 구간마다 앞부분
    (gc.CAL_SEC, 광류 소실점이 부족하면 gc.CAL_MAX_SEC 까지) 에서 표본을 모아 보정값을 정한다.

    - 광류: 0.1초 간격 프레임 쌍 → 회전 보정 FOE, 보닛 띠 추적점 (gc.FlowCollector)
    - 모델: 평가 프레임 (fps_eval 간격) 마다 차선 세그 + YOLO → 차선 소실점·차선 높이, 차량 박스.
            결과는 본 판정에서 다시 쓰도록 cache 에 둔다 (CAL_CACHE_MAX 프레임까지)
    - 차 실내 (t27): 구간 전체의 0.1초 쌍 (보정 구간 뒤는 gc.INT_EXTRA_EVERY 쌍에 하나) 추적점 → 정지 칸,
            구간 전체 YOLO (보정 구간 뒤는 gc.INT_DET_EVERY_SEC 간격) → 반복 검출 박스
    - 내 차 영역 (t29): Mask2Former 를 이 함수 동안만 올려 구간 앞 M2F_SAMPLE_SEC 시각 프레임에 돌린다
            (과반 투표 → 테두리에 닿는 덩어리). 보닛 대체 추정, 차선 픽셀 제거, 실내 마스크에 쓴다.
            로드 실패면 쓰지 않는다 (박스를 지우지 않는 쪽). 구간이 끝나면 gc.solve_interior 가 마스크를 붙인다
    Returns (cuts, calibs, cache, n_eval, segs_info)
    """
    cap = cv2.VideoCapture(str(video_path))
    step = fps_src / fps_eval if 0 < fps_eval < fps_src else 1.0
    m2f, m2f_status, t_m2f = None, 'off', 0.0
    if CAR_M2F:
        a = time.perf_counter()
        try:
            m2f = car_models.load_mask2former(engine.device)
            m2f_status = 'ok'
        except Exception as e:   # 모델·캐시 문제 → 내 차 영역 없이 진행 (지우지 않는 쪽)
            m2f_status = f'load_failed: {type(e).__name__}: {e}'[:200]
            log(f'[engine] Mask2Former 로드 실패 — 내 차 영역 없이 진행: {m2f_status}')
        t_m2f += time.perf_counter() - a

    def new_seg(i):
        fc = gc.FlowCollector(W, H, fps_src)
        return {'start': i, 'end': i, 't0': i / fps_src, 't_end': None, 'done': False, 'cal': None,
                'flow': fc, 'fstep': fc.step, 'prev_gray': None, 'frames': [],
                'ic': gc.InteriorCollector(W, H), 'gprev': None, 'int_k': 0, 'win_dets': [],
                'last_det_t': None, 'm2f_masks': [], 'm2f_next': 0, 'm2f_ego': None,
                'lane_pngs': []}

    def solve(sg):
        """표본을 다 모은 구간을 바로 풀고 마스크·광류 표본을 놓는다 (cache 에 든 것은 본 판정이 비운다)."""
        if sg['t_end'] is None:
            sg['t_end'] = sg['end'] / fps_src
        sg['n_flow_pairs'], sg['n_model_frames'] = len(sg['flow'].pairs), len(sg['frames'])
        sg['m2f_ego'] = car_models.vote_ego(sg['m2f_masks'])
        sg['m2f_info'] = {'status': m2f_status, 'n_frames': len(sg['m2f_masks'])}
        sg['m2f_masks'] = []
        sg['cal'] = _solve_car_calib(sg, W, H)
        sg['frames'] = sg['flow'] = sg['prev_gray'] = None

    def finish(sg):
        """구간 끝: 보정값이 없으면 풀고, 구간 전체 증거로 실내 마스크와 차선 영점을 붙인다.
        [t44·t48] 그다음 차선 신뢰 판정 — 폭이 범위 밖이면 차선을 뺀 보정값으로 바꾼다 (_apply_lane_trust)."""
        if sg['cal'] is None:
            solve(sg)

        def interior(cal):
            gc.solve_interior(sg['ic'], cal, sg['win_dets'], SimpleIouTracker,
                              m2f_ego=sg['m2f_ego'], m2f_info=sg['m2f_info'])
        interior(sg['cal'])
        _lane_zero(sg['cal'], sg['lane_pngs'], W, H)
        if LANE_TRUST:
            sg['cal'] = _apply_lane_trust(sg['cal'], interior)
        sg['ic'] = sg['gprev'] = sg['win_dets'] = sg['m2f_ego'] = sg['lane_pngs'] = None

    segs = [new_seg(0)]
    cache, cuts = {}, []
    prev_thumb, next_t, i, n_eval = None, 0.0, 0, 0
    t_flow = t_model = t_int = t_lane = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        th = _cut_thumb(frame)
        if prev_thumb is not None and float((th * prev_thumb).sum()) < CUT_NCC:
            cuts.append(i)
            finish(segs[-1])
            segs.append(new_seg(i))
        prev_thumb = th
        cur = segs[-1]
        cur['end'] = i
        t = i / fps_src
        is_eval = i + 1e-6 >= next_t
        if is_eval:
            next_t += step
            n_eval += 1
        on_step = (i - cur['start']) % cur['fstep'] == 0
        g = None
        if on_step:
            a = time.perf_counter()
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cur['gprev'] is not None:
                k = cur['int_k']
                cur['int_k'] += 1
                if not cur['done'] or k % gc.INT_EXTRA_EVERY == 0:
                    cur['ic'].add_pair(cur['gprev'], g, t)
            cur['gprev'] = g
            t_int += time.perf_counter() - a
        if (m2f is not None and not cur['done'] and cur['m2f_next'] < len(M2F_SAMPLE_SEC)
                and t - cur['t0'] >= M2F_SAMPLE_SEC[cur['m2f_next']]):
            a = time.perf_counter()
            cur['m2f_masks'].append(car_models.m2f_ego_mask(m2f, frame, engine.device))
            cur['m2f_next'] += 1
            t_m2f += time.perf_counter() - a
        if not cur['done']:
            el = t - cur['t0']
            fc = cur['flow']
            if el >= gc.CAL_SEC and (fc.n_prov_ok >= gc.FOE_TARGET_PROV or el >= gc.CAL_MAX_SEC):
                cur['done'], cur['t_end'] = True, t
                solve(cur)
            else:
                if on_step:
                    a = time.perf_counter()
                    if cur['prev_gray'] is not None:
                        fc.add_pair(cur['prev_gray'], g, t)
                    cur['prev_gray'] = g
                    t_flow += time.perf_counter() - a
                if is_eval:
                    a = time.perf_counter()
                    mask, dets = engine.models_lane_det(frame)
                    if len(cache) < CAL_CACHE_MAX:
                        cache[i] = (mask, dets)
                    cur['frames'].append((i, t, mask, dets))
                    cur['win_dets'].append((t, dets))
                    cur['lane_pngs'].append(cv2.imencode('.png', mask)[1])
                    t_model += time.perf_counter() - a
        elif is_eval:
            # 보정 구간 뒤 — 차로 안 평소 위치용 차선 마스크 (평가 프레임마다, PNG 로 압축해 구간 끝까지 둔다)
            if engine.yolop is not None:
                a = time.perf_counter()
                cur['lane_pngs'].append(cv2.imencode('.png', engine.seg_mask(frame))[1])
                t_lane += time.perf_counter() - a
            # 실내 반복 검출 판단용 YOLO 만 (0.5초 간격)
            if cur['last_det_t'] is None or t - cur['last_det_t'] >= gc.INT_DET_EVERY_SEC - 1e-6:
                a = time.perf_counter()
                dets = detect_objects(engine.det_model, frame, engine.device, engine.det_conf,
                                      lowconf=RESCUE_CONF_MIN if RESCUE_LOWCONF else None)
                cur['win_dets'].append((t, dets))
                cur['last_det_t'] = t
                t_int += time.perf_counter() - a
        i += 1
    cap.release()
    a = time.perf_counter()
    finish(segs[-1])
    t_int += time.perf_counter() - a
    if m2f is not None:   # 사전 점검 끝 — Mask2Former 를 GPU 에서 내린다
        del m2f
        if engine.device == 'cuda':
            torch.cuda.empty_cache()

    calibs, info = [], []
    for k, sg in enumerate(segs):
        cal = sg['cal']
        calibs.append(cal)
        info.append({'segment': k, 'start_frame': sg['start'], 'end_frame': sg['end'],
                     'n_flow_pairs': sg['n_flow_pairs'], 'n_model_frames': sg['n_model_frames']})
        if k < 5 or k == len(segs) - 1:
            ii = cal.interior_info
            log(f'[engine] 보정 구간 {k} (프레임 {sg["start"]}~{sg["end"]}, 표본 {sg["t0"]:.1f}~{sg["t_end"]:.1f}s): '
                f'수평선 {cal.horizon_y:.0f}px ({cal.horizon_y / H:.3f}H, {cal.horizon_src}/{cal.horizon_conf}), '
                f'소실점 x {cal.vp_x:.0f} ({cal.vp_x_src}), 보닛 {cal.hood_row} ({cal.hood_src}), '
                f'카메라 높이 {cal.cam_h:.2f}m ({cal.cam_h_src}), 내 차 영역 {ii["status"]} {ii["frac"] * 100:.1f}% '
                f'(M2F {ii["m2f"].get("frac", 0) * 100:.1f}% / {ii["m2f"].get("n_frames", 0)}장, 움직임 쌍 '
                f'{ii["n_moving_pairs"]}, 반복 검출 박스 '
                f'{sum(1 for b in ii.get("static_boxes", []) if "rejected" not in b)})')
    for k, cal in enumerate(calibs[:5]):
        lb = cal.lane_baseline
        log(f'[engine] 구간 {k} 차선 영점: '
            + (f'y_ref {lb["y_ref"]} ({lb["y_ref_ratio"]:.3f}H), 차로 폭 {lb["lane_w_px"]:.0f}px = {lb["lane_w_m"]:.2f}m, '
               f'평소 위치 {lb["baseline_m"]:+.3f}m (원값 {lb["raw_median_m"]:+.3f}{", 상한 적용" if lb["capped"] else ""}, '
               f'LDW 기준 열 이동 {lb["ldw_shift_px"]:+.0f}px), 근거 {lb["evidence_frames"]}/{lb["examined_frames"]}, '
               f'신뢰도 {lb["conf"]}' + (f' ({", ".join(lb["conf_reasons"])}, 의심 {lb["suspect"]})' if lb['conf_reasons'] else '')
               if lb['ldw_measurable']
               else f'실패 → LDW 측정 불가 ({lb["reason"]}, 쌍 프레임 {lb["pair_frames"]}, 근거 {lb["evidence_frames"]}/'
                    f'{lb["examined_frames"]}, 필요 {lb["required_frames"]})'))
        lt = getattr(cal, 'lane_trust', None)
        if lt is not None:
            ck = lt['check']
            m = lambda v: f'{v:.2f}m' if v is not None else '-'
            log(f'[engine] 구간 {k} 차선 신뢰: {"신뢰" if lt["trusted"] else "불신"} ({lt["reason"]}), 차로 폭 '
                f'{m(lt["lane_w_m"])}, 검산 폭 {m(lt["lane_w_m_check"])} (범위 {lt["range_m"][0]}~{lt["range_m"][1]}m. '
                f'검산 = 차선 없는 수평선 {ck["horizon_y"]:.0f}px {ck["horizon_src"]}, 높이 {ck["cam_h_m"]:.2f}m '
                f'{ck["cam_h_src"]}{"" if ck["cam_h_independent"] else " — 독립 높이 없음"})'
                + (f' → 차선 뺀 보정값으로 바꿈 (기하 {"바뀜" if lt["calib_changed"] else "같음"})' if lt['applied']
                   else (' → LDW 만 끔, 보정값 유지' if not lt['trusted'] else '')))
    log(f'[engine] 보정 단계: 구간 {len(segs)}개, 광류 {t_flow:.1f}s, 모델 {t_model:.1f}s, 실내 {t_int:.1f}s, '
        f'평소 위치 차선 {t_lane:.1f}s, Mask2Former {t_m2f:.1f}s ({m2f_status})')
    return cuts, calibs, cache, n_eval, info


def _lane_zero(cal, pngs, W, H):
    """[car] 사전 점검 — 차선 영점 (t40, DECISIONS 2026-10-09). 구간의 YOLOPv2 차선 마스크 (보정 구간 + 그 뒤 평가
    프레임마다) 에 판정과 같은 차선 고르기 (find_lanes: 보닛 띠·내 차 영역의 차선 픽셀 제거, 고정 소실점) 를 하고,
    gc.lane_zero 가 y_ref (차선이 실제 보이는 가장 아래 행), 그 행의 차로 폭, 차로 안 평소 위치 b 를 정한다.
    실패하면 (근거 부족·수평선 실패) 이 구간 LDW 는 전부 측정 불가. 결과는 cal.lane_baseline (t32 키 이름 유지)."""
    global CAR_CAL
    saved = CAR_CAL
    CAR_CAL = cal
    try:
        empty = np.zeros((H, W), np.uint8)
        vp_fixed = (cal.vp_x, cal.horizon_y) if cal.horizon_ok else None
        frames = []
        for png in pngs or []:
            lane = find_lanes(cv2.imdecode(png, cv2.IMREAD_UNCHANGED), empty, bike_x=float(cal.vp_x),
                              vp_fixed=vp_fixed)
            frames.append(lane['lines'] if lane['ok'] else [])

        def lane_of(j):
            return lane_input_mask(cv2.imdecode(pngs[j], cv2.IMREAD_UNCHANGED), empty) == 3
        cal.lane_baseline = gc.lane_zero(cal, frames, lane_of, len(frames),
                                         relaxed=LDW_VARIANT in ('VA', 'VAB'))   # [t41 VA] 최소 근거 완화
    finally:
        CAR_CAL = saved


def _solve_car_calib(sg, W, H):
    """구간 하나의 표본 → gc.CarCalib (gc.decide_calib 이 출처·신뢰도를 정한다)."""
    global CAR_CAL
    foe = sg['flow'].solve()
    m2f_ego = sg.get('m2f_ego')
    m2f_hood = car_models.hood_row_from_ego(m2f_ego) if m2f_ego is not None else None
    # 보닛: 수동 (--hood-ratio) > 광류 (움직이는 동안 정지한 띠) > Mask2Former 내 차 영역 (t29) > 기본 HOOD_RATIO
    if HOOD_MANUAL:
        hood_row, hood_src = _manual_hood_row(H), 'manual'
    elif foe['hood_row'] is not None:
        hood_row = int(foe['hood_row'])
        hood_src = {'flow': 'flow', 'motion_edge': 'flow_motion_edge'}.get(foe['hood_status'],
                                                                           'flow_none_visible')
    elif m2f_hood is not None:
        hood_row, hood_src = int(m2f_hood), 'mask2former'
    else:
        hood_row, hood_src = _manual_hood_row(H), 'default'
    # 차선 소실점·차선 높이 (t29): YOLOPv2 차선 픽셀에서 보닛 띠 아래와 Mask2Former 내 차 영역 (반사) 을 지우고
    # 프레임마다 내 차로 좌우 차선 쌍 (car_models.lane_pair_from_mask, t28 probe 규칙) → 교점과 기울기 차
    lane_samples = []
    for fi, t, mask, dets in sg['frames']:
        lane_bin = mask == 3
        lane_bin[hood_row:] = False
        if m2f_ego is not None:
            lane_bin &= ~m2f_ego
        r = car_models.lane_pair_from_mask(lane_bin)
        if r is not None:
            lane_samples.append(r)
    lane_sum = gc.lane_summary(lane_samples, W, H)
    # 차량 박스: 보정 구간 YOLO 박스 (conf ≥ DET_CONF) 로 track 을 다시 만든다
    tr = SimpleIouTracker(iou_threshold=TRACKER_IOU, max_ghost_frames=2)
    veh = []
    for k, (fi, t, mask, dets) in enumerate(sg['frames']):
        yd = [o for o in dets if o.get('source', 'yolo') == 'yolo']
        tracks = tr.update(k, [(tuple(o['bbox']), 0) for o in yd])
        ids = {tuple(x.bbox): x.track_id for x in tracks if x.ghost_count == 0}
        veh += [{'bbox': o['bbox'], 'cls': o['cls'], 'conf': o['conf'], 'tid': ids.get(tuple(o['bbox']))}
                for o in yd]
    reg_fn = lambda xv: gc.box_regression_horizon(veh, xv, hood_row, W, H)
    veh_fn = lambda yh, xv: gc.vehicle_height(veh, yh, xv, hood_row, W, H)
    window = (sg['t0'], sg['t_end'])
    hood_detail = {'status': foe['hood_status'], 'row_flow': foe['hood_row'], 'row_mask2former': m2f_hood}
    cal = gc.decide_calib(W, H, foe, lane_sum, reg_fn=reg_fn, veh_fn=veh_fn,
                          hood_row=hood_row, hood_src=hood_src, window=window)
    cal.detail['hood'] = hood_detail
    if LANE_TRUST:
        # [t44] 차선을 뺀 보정값 (수평선 FOE → 차량 박스 회귀 → 기본값, 높이 차량 박스 → 기본값). 구간 끝 차선 신뢰
        # 판정의 기준이고, 불신이면 판정에 이것을 쓴다 (_apply_lane_trust). 차선 결과는 detail 에 기록만
        lf = gc.decide_calib(W, H, dict(foe), {}, reg_fn=reg_fn, veh_fn=veh_fn,
                             hood_row=hood_row, hood_src=hood_src, window=window)
        lf.detail['hood'] = hood_detail
        lf.detail['lane'] = {**lane_sum, 'excluded': 'lane_untrusted'}
        cal.lane_free = lf
    return cal


def _apply_lane_trust(cal, redo_interior):
    """[car] 구간 끝 차선 신뢰 판정 (t44·t48, DECISIONS 2026-10-10). cal = 차선을 쓴 보정값 (실내 마스크·영점까지 붙은 것),
    cal.lane_free = 차선을 뺀 보정값. 믿거나, 불신이어도 폭이 범위 밖이 아니면 (영점 실패·검산 불가 — LDW 만 끈다)
    cal 그대로 (t40 과 같은 보정값). 폭이 범위 밖 (lt.exclude_lanes) 이면 lane_free 로 바꾸고 실내 마스크를 그
    수평선·소실점으로 다시 만든다 (redo_interior — 보호 영역과 보닛 대체 추정이 수평선을 쓴다). 영점은 차선 픽셀
    측정이라 cal 의 것을 옮겨 기록만 한다 (LDW 는 어차피 측정 불가). Returns 판정에 쓸 보정값 (.lane_trust 붙음,
    applied = 바꿨는지 = gc.lanes_excluded)."""
    lf = cal.lane_free
    cal.lane_free = None
    lt = gc.lane_trust(cal.lane_baseline, cal, lf)
    if not lt['exclude_lanes']:
        lt['applied'] = False
        cal.lane_trust = lt
        return cal
    before, after = gc.calib_geom(cal), gc.calib_geom(lf)
    lt.update(applied=True, calib_changed=before != after, lane_calib=before)
    lf.lane_baseline = cal.lane_baseline
    lf.lane_trust = lt
    redo_interior(lf)
    return lf


def run_video_car(video_path, fps_eval, out_dir, engine, dump_estimates=False, limit=0, log=print):
    """[car] 보정 단계 → 본 판정 한 번 (Depth Pro 초점 고정 2회차 없음 — 거리에 Depth Pro 를 안 쓴다).
    본 판정은 원본 프레임을 처음부터 다시 읽으며 구간이 바뀌면 그 구간 보정값으로 바꾼다."""
    cap = cv2.VideoCapture(str(video_path))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = fps_src / fps_eval if 0 < fps_eval < fps_src else 1.0
    t_start = time.time()
    cuts, calibs, cache, n_eval_all, seg_info = calibrate_car(engine, video_path, fps_src, fps_eval,
                                                              W, H, log=log)
    t_cal = time.time() - t_start
    for k, cal in enumerate(calibs):   # 구간별 실내 마스크 (흰색 = 실내) — 확인용
        name = f'interior_seg{k}.png'
        cv2.imwrite(str(out_dir / name), cal.interior_mask.astype(np.uint8) * 255)
        cal.interior_info['png'] = name
    slideshow = len(calibs) >= SLIDESHOW_CUT_RATIO * max(n_eval_all, 1)
    cut_set = set(cuts)

    engine.reset()
    engine.set_car_calib(calibs[0], 0)
    est_f = open(out_dir / 'estimates.jsonl', 'w', encoding='utf-8') if dump_estimates else None
    timings, gpu_peak = [], None
    n_eval, next_t, i, seg = 0, 0.0, 0, 0
    last_depth_t, new_scene = None, False
    t2 = time.time()
    with open(out_dir / 'frames.jsonl', 'w', encoding='utf-8') as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in cut_set:
                seg += 1
                engine.set_car_calib(calibs[seg], seg)
                last_depth_t = None
                new_scene = slideshow
            if i + 1e-6 >= next_t:
                t_sec = round(i / fps_src, 3)
                # 근접 보험은 수평선 신뢰도 high/medium 구간만 (gc.DG_CONFS). 아니면 Depth Pro 도 안 돌린다
                run_depth = (CAR_DEPTH_GUARD and calibs[seg].horizon_conf in gc.DG_CONFS
                             and (last_depth_t is None or t_sec - last_depth_t >= DEPTH_EVERY_SEC - 1e-6))
                scene, focal_info, tm = engine.analyze_frame(frame, cached=cache.pop(i, None),
                                                             run_depth=run_depth)
                if run_depth:
                    last_depth_t = t_sec
                t = time.perf_counter()
                rec = engine.finish_frame(scene, focal_info, i, t_sec, new_scene=new_scene)
                new_scene = False
                tm['logic'] += time.perf_counter() - t
                tm['total'] = sum(v for k, v in tm.items() if k != 'total')
                engine.last_timing = tm
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                if est_f is not None:
                    est_f.write(json.dumps(engine.last_estimates, ensure_ascii=False) + '\n')
                timings.append(tm)
                n_eval += 1
                next_t += step
                if n_eval == 3:
                    gpu_peak = _gpu_used_mb()
                if n_eval % 50 == 0:
                    log(f'  [{n_eval}] frame {i} → {rec["bin_label"]} {rec["top_warning"] or ""} '
                        f'({tm["total"]:.2f}s)')
                if limit and n_eval >= limit:
                    break
            i += 1
    cap.release()
    if est_f is not None:
        est_f.close()
    t_judge = time.time() - t2
    car_meta = {
        'calib_sec': round(t_cal, 2), 'judge_sec': round(t_judge, 2),
        'fps_default': CAR_FPS_DEFAULT, 'depth_every_sec': DEPTH_EVERY_SEC if CAR_DEPTH_GUARD else None,
        'ttc': 'record_only' if CAR_TTC else False, 'depth_guard': CAR_DEPTH_GUARD, 'slideshow': slideshow,
        'ttc_win_sec': gc.TTC_WIN_SEC,
        'ego_lane_half_m': gc.EGO_LANE_HALF_M,
        'n_cuts': len(cuts),
        # [t41] 비교 실험 방식일 때만 남긴다 (기본 V0 는 t40 과 같은 meta)
        **({'ldw_variant': LDW_VARIANT} if LDW_VARIANT != 'V0' else {}),
        # [t44] 차선 신뢰 범위 (구간별 판정은 car_calib[].lane_trust). --no-lane-trust 면 키 없음
        **({'lane_trust_range_m': list(gc.LANE_TRUST_W_RANGE_M)} if LANE_TRUST else {}),
    }
    calib_meta = [{**si, **cal.to_dict()} for si, cal in zip(seg_info, calibs)]
    return n_eval, timings, gpu_peak, fps_src, W, H, car_meta, calib_meta


def _run_video_car_and_meta(video_path, fps_eval, out_dir, engine, dump_estimates, limit, log):
    """[car] run_video_car + meta.json + 성능 로그."""
    if engine.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_eval, timings, gpu_peak, fps_src, W, H, car_meta, calib_meta = run_video_car(
        video_path, fps_eval, out_dir, engine, dump_estimates=dump_estimates, limit=limit, log=log)
    meta = {
        'source': str(video_path.resolve()),
        'fps_src': round(fps_src, 3),
        'fps_eval': round(min(fps_eval, fps_src), 3),
        'size': [W, H],
        'n_frames': n_eval,
        'models': engine.models_info,
        'focal_lock': {'enabled': False, 'reason': 'car 모드: 거리에 Depth Pro 를 쓰지 않아 초점 고정 없음'},
        'vehicle': VEHICLE,
        # 보닛으로 본 화면 아래 비율 (첫 구간 실제 사용값). 구간별 값과 출처는 car_calib
        'hood_ratio': calib_meta[0]['hood_ratio'] if calib_meta else None,
        'hood_ratio_src': calib_meta[0]['hood_src'] if calib_meta else None,
        'car': car_meta,
        'car_calib': calib_meta,
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    steady = timings[1:] or timings
    perf = {k: float(np.mean([t[k] for t in steady])) for k in steady[0]} if steady else {}
    log(f'[engine] 완료: {n_eval} 프레임 / {time.time() - t0:.1f}s '
        f'(보정 {car_meta["calib_sec"]:.1f}s, 판정 {car_meta["judge_sec"]:.1f}s) → {out_dir}')
    if perf:
        log('[engine] 프레임당 평균 (워밍업 제외, 보정 단계 재사용·Depth Pro 건너뛴 프레임 포함): ' +
            ', '.join(f'{k} {v * 1000:.0f}ms' for k, v in perf.items()))
    if engine.device == 'cuda':
        log(f'[engine] torch GPU peak: allocated {torch.cuda.max_memory_allocated() / 2**30:.2f}GB, '
            f'reserved {torch.cuda.max_memory_reserved() / 2**30:.2f}GB')
        if gpu_peak:
            log(f'[engine] nvidia-smi 사용량 (실행 중): {gpu_peak[0]}MB / {gpu_peak[1]}MB')
    return out_dir


def run_video(video_path, fps_eval=5.0, out_root=None, device='cuda',
              n_folds=None, limit=0, engine=None,
              dump_estimates=False, focal_lock=FOCAL_LOCK, log=print):
    """영상 → frames.jsonl (+ meta.json, estimates.jsonl).

    focal_lock (기본, DECISIONS 2026-10-08) — 두 번 훑기:
      1회차: 프레임마다 모델을 돌려 판정 재료 (judge_scene) 와 그 프레임 초점거리를 메모리에 모은다.
             이때도 그 프레임 초점거리로 판정해 frames.jsonl 에 바로 쓴다 (웹 진행률이 이 줄 수를 센다).
      구간: 연속 원본 프레임 썸네일의 정규화 상관 < CUT_NCC 이면 장면 전환 → 구간 나눔.
            구간별 고정값 = 그 구간 초점거리 중앙값.
      2회차: 모델 없이 고정값으로 다시 판정 (트래커 포함 처음부터) →
             frames.jsonl.tmp 에 쓴 뒤 frames.jsonl 을 교체한다.
    """
    video_path = Path(video_path)
    out_dir = Path(out_root or config.RUNS_DIR) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'영상 열기 실패: {video_path}')
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = fps_src / fps_eval if 0 < fps_eval < fps_src else 1.0
    log(f'[engine] {video_path.name}: {n_src}f @ {fps_src:.2f}fps, {W}x{H} '
        f'→ 평가 {min(fps_eval, fps_src):g}fps (step {step:.2f})')

    if engine is None:
        engine = Engine(device=device, fps_eval=min(fps_eval, fps_src), n_folds=n_folds,
                        dump_estimates=dump_estimates, log=log)
    engine.dump_estimates = dump_estimates
    engine.reset()
    if VEHICLE == 'car':
        cap.release()
        return _run_video_car_and_meta(video_path, fps_eval, out_dir, engine, dump_estimates, limit, log)
    # 초점 고정이면 estimates 는 2회차 (최종 판정) 에서 쓴다
    est_f = (open(out_dir / 'estimates.jsonl', 'w', encoding='utf-8')
             if dump_estimates and not focal_lock else None)
    if engine.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    timings = []
    gpu_peak = None
    n_eval = 0
    next_t = 0.0
    i = 0
    t_start = time.time()
    store = []          # 1회차 판정 재료: (frame_idx, t_sec, scene, focal_info)
    cut_frames = []     # 직전 원본 프레임과 이어지지 않는 원본 프레임 번호
    prev_thumb = None
    with open(out_dir / 'frames.jsonl', 'w', encoding='utf-8') as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if focal_lock:
                th = _cut_thumb(frame)
                if prev_thumb is not None and float((th * prev_thumb).sum()) < CUT_NCC:
                    cut_frames.append(i)
                prev_thumb = th
            if i + 1e-6 >= next_t:
                t_sec = round(i / fps_src, 3)
                scene, focal_info, tm = engine.analyze_frame(frame)
                t = time.perf_counter()
                rec = engine.finish_frame(scene, focal_info, i, t_sec)
                tm['logic'] += time.perf_counter() - t
                tm['total'] = sum(v for k, v in tm.items() if k != 'total')
                engine.last_timing = tm
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                if est_f is not None:
                    est_f.write(json.dumps(engine.last_estimates, ensure_ascii=False) + '\n')
                if focal_lock:
                    store.append((i, t_sec, scene, focal_info))
                timings.append(tm)
                n_eval += 1
                next_t += step
                if n_eval == 3:   # 모델이 모두 올라간 뒤 GPU 사용량 1회 측정
                    gpu_peak = _gpu_used_mb()
                if n_eval % 10 == 0:
                    log(f'  [{n_eval}] frame {i} → {rec["bin_label"]} '
                        f'{rec["top_warning"] or ""} ({tm["total"]:.2f}s)')
                if limit and n_eval >= limit:
                    break
            i += 1
    cap.release()
    t_pass1 = time.time() - t_start

    lock_meta = {'enabled': False}
    t_pass2 = 0.0
    if focal_lock and store:
        # 구간 나누기 + 구간별 고정 초점거리
        segs = focal_segments([(s[0], s[3].get('focal')) for s in store], cut_frames)
        seg_of = {k: si for si, sg in enumerate(segs) for k in sg['members']}
        seg_start = {sg['members'][0] for sg in segs}
        # 사진 이어 붙인 영상: 평가 프레임 절반 이상이 새 구간을 연다 → 구간마다 시간 상태 초기화.
        # 연속 영상의 짧은 전환 (사고로 카메라가 구르는 순간 등) 에서는 유령을 이어 가야 해서 초기화 안 함
        slideshow = len(segs) >= SLIDESHOW_CUT_RATIO * len(store)
        log(f'[engine] 초점거리 고정: 장면 전환 {len(cut_frames)}곳 → 구간 {len(segs)}개'
            f'{" (사진 이어 붙인 영상으로 판단 — 구간마다 트래커 초기화)" if slideshow else ""}, '
            f'2회차 재판정 시작')
        # 2회차 — 모델 없이 처음부터 (트래커 초기화)
        t2 = time.time()
        engine.reset()
        tmp = out_dir / 'frames.jsonl.tmp'
        est_f = open(out_dir / 'estimates.jsonl', 'w', encoding='utf-8') if dump_estimates else None
        with open(tmp, 'w', encoding='utf-8') as f:
            for k, (fi, t_sec, scene, focal_info) in enumerate(store):
                sg = segs[seg_of[k]]
                lock = ({'segment': seg_of[k], 'f_fixed': sg['f_fixed']}
                        if sg['f_fixed'] else None)
                rec = engine.finish_frame(scene, focal_info, fi, t_sec, lock=lock,
                                          new_scene=slideshow and k > 0 and k in seg_start)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                if est_f is not None:
                    est_f.write(json.dumps(engine.last_estimates, ensure_ascii=False) + '\n')
                if (k + 1) % 10 == 0 or k + 1 == len(store):
                    log(f'  [재판정 {k + 1}/{len(store)}] frame {fi} → {rec["bin_label"]} '
                        f'{rec["top_warning"] or ""}')
        if est_f is not None:
            est_f.close()
        # Windows 는 다른 프로세스 (웹 진행률 집계) 가 잠깐 열고 있으면 교체가 실패한다 → 재시도
        for attempt in range(50):
            try:
                os.replace(tmp, out_dir / 'frames.jsonl')
                break
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)
        t_pass2 = time.time() - t2
        lock_meta = {
            'enabled': True,
            'cut_method': f'연속 원본 프레임 64x36 회색 썸네일 정규화 상관 < {CUT_NCC}',
            'n_cuts': len(cut_frames),
            'n_segments': len(segs),
            'slideshow': slideshow,
            'segments': [{k: v for k, v in sg.items() if k != 'members'} for sg in segs],
            'pass2_sec': round(t_pass2, 2),
        }
    elif est_f is not None:
        est_f.close()

    meta = {
        'source': str(video_path.resolve()),
        'fps_src': round(fps_src, 3),
        'fps_eval': round(min(fps_eval, fps_src), 3),
        'size': [W, H],
        'n_frames': n_eval,
        'models': engine.models_info,
        'focal_lock': lock_meta,
        'vehicle': VEHICLE,
        **({'hood_ratio': HOOD_RATIO} if VEHICLE == 'car' else {}),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding='utf-8')

    # 성능 요약 (첫 프레임은 워밍업이라 제외)
    steady = timings[1:] or timings
    perf = {k: float(np.mean([t[k] for t in steady])) for k in steady[0]} if steady else {}
    log(f'[engine] 완료: {n_eval} 프레임 / {time.time() - t_start:.1f}s '
        f'(1회차 {t_pass1:.1f}s, 2회차 {t_pass2:.1f}s) → {out_dir}')
    if perf:
        log('[engine] 프레임당 평균 (워밍업 제외): ' +
            ', '.join(f'{k} {v * 1000:.0f}ms' for k, v in perf.items()))
    if engine.device == 'cuda':
        log(f'[engine] torch GPU peak: allocated {torch.cuda.max_memory_allocated() / 2**30:.2f}GB, '
            f'reserved {torch.cuda.max_memory_reserved() / 2**30:.2f}GB')
        if gpu_peak:
            log(f'[engine] nvidia-smi 사용량 (실행 중): {gpu_peak[0]}MB / {gpu_peak[1]}MB')
    return out_dir


def main(argv=None):
    # CLI 옵션이 바꾸는 모듈 설정 (판정 함수들이 읽는다)
    global VEHICLE, HOOD_RATIO, RESCUE_LOWCONF, MASK_BOXES, GHOST_BOXES, NEAR_OF_TWO
    global HOOD_MANUAL, CAR_TTC, CAR_DEPTH_GUARD, CAR_M2F, LDW_VARIANT, LANE_TRUST
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(description='블랙박스 영상 → frames.jsonl')
    ap.add_argument('video', type=Path)
    ap.add_argument('--fps', type=float, default=None,
                    help=f'평가 fps (기본: moto 5, car {CAR_FPS_DEFAULT:g})')
    ap.add_argument('--out', type=Path, default=None, help='출력 루트 (기본 runs_out/)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--folds', type=int, default=None, help='차선 세그 fold 수 (기본 5 전부)')
    ap.add_argument('--limit', type=int, default=0, help='평가 프레임 수 상한 (0 = 전부)')
    ap.add_argument('--dump-estimates', action='store_true',
                    help='객체별 거리 추정값 진단 기록 → estimates.jsonl (판정에는 영향 없음)')
    ap.add_argument('--no-focal-lock', action='store_true',
                    help='초점거리 영상 단위 고정 (두 번 훑기) 끄기 — 프레임마다 Depth Pro 초점거리')
    # 놓친 박스 되살리기 (t15) — 겹마다 끄기
    ap.add_argument('--no-lowconf', action='store_true', help='1겹 저확신 큰·가까운 박스 구제 끄기')
    ap.add_argument('--no-mask-boxes', action='store_true', help='2겹 Moveable 마스크 보조 박스 끄기')
    ap.add_argument('--no-ghost', action='store_true', help='3겹 트래커 유령 박스 끄기')
    ap.add_argument('--no-near-of-two', action='store_true',
                    help='거리 성분 두 개가 2배 이상 다를 때 가까운 쪽 쓰기 끄기')
    # 차종 (t16)
    ap.add_argument('--vehicle', choices=['moto', 'car'], default='moto',
                    help='moto: 오토바이 (기본) | car: 자동차 블랙박스 (바이크·라이더 클래스 무시)')
    ap.add_argument('--hood-ratio', type=float, default=None,
                    help=f'car 모드 보닛 비율 수동 지정: 화면 아래 이 비율을 보닛으로 보고 검출·차선에서 뺀다 '
                         f'(0 = 보닛 없음). 안 주면 보정 단계가 영상마다 잡고, 못 잡으면 {HOOD_RATIO}. '
                         f'moto 모드에서는 쓰지 않는다')
    ap.add_argument('--no-ttc', action='store_true',
                    help='car 모드 TTC 계산·기록 끄기 (TTC 는 판정에 쓰지 않고 기록만 한다)')
    ap.add_argument('--no-depth-guard', action='store_true',
                    help='car 모드 박스 없는 근접 물체 보험 끄기 (Depth Pro 를 돌리지 않는다)')
    ap.add_argument('--no-m2f', action='store_true',
                    help='car 모드 사전 점검에서 Mask2Former 내 차 영역 끄기 (광류·반복 검출만)')
    ap.add_argument('--ldw-variant', choices=LDW_VARIANTS, default='V0',
                    help='car 모드 LDW 방식 (t41 비교 실험). V0 = 기본 (t40) | VA = 영점 최소 근거 절반 완화 | '
                         'VB = 영점 신뢰도 low (차로 폭 2.75~3.75 m 밖) 구간 LDW 측정 불가 | VAB = 둘 다. moto 모드에서는 쓰지 않는다')
    ap.add_argument('--no-lane-trust', action='store_true',
                    help=f'car 모드 차선 신뢰 판정 끄기 (t40 동작). 켜져 있으면 (기본) 영점 차로 폭이 '
                         f'{gc.LANE_TRUST_W_RANGE_M[0]}~{gc.LANE_TRUST_W_RANGE_M[1]} m 밖이거나 영점 실패인 구간은 LDW 측정 '
                         f'불가이고 차선을 수평선·소실점·카메라 높이에서 뺀다')
    args = ap.parse_args(argv)
    VEHICLE = args.vehicle
    CAR_M2F = not args.no_m2f
    LDW_VARIANT = args.ldw_variant
    LANE_TRUST = not args.no_lane_trust
    if args.hood_ratio is not None:
        HOOD_RATIO = max(0.0, min(args.hood_ratio, 0.5))
        HOOD_MANUAL = True
    RESCUE_LOWCONF = not args.no_lowconf
    MASK_BOXES = not args.no_mask_boxes
    GHOST_BOXES = not args.no_ghost
    NEAR_OF_TWO = not args.no_near_of_two
    CAR_TTC = not args.no_ttc
    CAR_DEPTH_GUARD = not args.no_depth_guard
    fps = args.fps if args.fps is not None else (CAR_FPS_DEFAULT if VEHICLE == 'car' else 5.0)
    run_video(args.video, fps_eval=fps, out_root=args.out, device=args.device,
              n_folds=args.folds, limit=args.limit, dump_estimates=args.dump_estimates,
              focal_lock=not args.no_focal_lock)
    return 0


if __name__ == '__main__':
    sys.exit(main())
