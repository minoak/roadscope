# SPDX-License-Identifier: AGPL-3.0-only
"""rules.py (기존 mvp.py) — Risk Assessor v5 판정 로직 단일 파일 (torch 없이 동작).
엔진은 `from roadscope import rules as mvp` 로 불러 쓴다.

src/RISK_ASSESSOR_V5.ipynb 의 코드 셀에서 판정에 필요한 함수만 그대로 옮겼다.
    차선/VP → 객체·ego → 거리(3-estimator) → 4-zone corridor → 4 경고 → 보정 → bin = max

노트북 대비 뺀 것:
    - torch / transformers / 세그·깊이 모델 로딩 (GT mask 주입 + depth 없음 전제)
    - 시각화 (draw_hud_v5, draw_zones_on_ax, 한글 폰트 설정)
    - promote_lane_violators_to_bsw (노트북에서도 호출부가 주석 처리된 함수)

사용:
    from roadscope.rules import assess, result_to_json
    result = assess('samples/images/X.png', 'samples/masks/X.png')
    print(result['bin'], result['active_warnings'])

CLI (samples/ 폴더는 이 리포에 없다. 이미지와 GT 마스크를 직접 준비한다):
    python -m roadscope.rules samples
    → outputs/<이미지이름>.json + outputs/compare.md (expected_outputs.json 대비)
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from PIL import Image


# ====== Class ======
CLASS_NAMES = ['Background', 'Undrivable', 'Road', 'Lane Mark',
               'Moveable', 'My bike', 'Rider']

# ====== Distance priors ======
LANE_WIDTH_M = 3.5            # 한국 차선 표준
CAMERA_HEIGHT_M = 1.0         # 오토바이 dashcam 높이
DEFAULT_FOCAL_RATIO = 1.2     # f_default = img_w * 1.2 (광각 fallback)

OBJECT_WIDTH_M = {
    'moveable': 1.8, 'car': 1.8, 'truck': 2.5, 'bus': 2.55,
    'motorcycle': 0.8, 'bicycle': 0.6, 'person': 0.5,
}

DISTANCE_BINS = [(0, 5), (5, 15), (15, 30), (30, float('inf'))]
DISTANCE_BIN_NAMES = ['<5m', '5-15m', '15-30m', '>30m']

# Estimator 가중치 (depth 모드에 따라 다름)
ESTIMATOR_WEIGHTS_INVERSE = {'ground': 0.55, 'size': 0.40, 'depth': 0.05}  # DA V2
ESTIMATOR_WEIGHTS_METRIC = {'ground': 0.25, 'size': 0.25, 'depth': 0.50}   # Depth Pro

# ====== 4 Warning thresholds ======
FCW_DIST_CAUTION = 10.0
FCW_DIST_WARNING = 5.0
FCW_DIST_CRITICAL = 3.0
FCW_BBOX_RATIO_CRITICAL = 0.25

LDW_LATERAL_CAUTION = 0.15
LDW_LATERAL_WARNING = 0.3

BSW_DIST_CAUTION = 5.0
BSW_DIST_WARNING = 3.0

HEAD_DIST_WARNING = 5.0
HEAD_DIST_CRITICAL = 3.0
HEAD_ASPECT_MIN = 1.5
HEAD_BBOX_RATIO = 0.10

# ====== Zone alphas ======
DEFAULT_ALPHAS = {'critical': 0.85, 'danger': 0.55, 'caution': 0.15}
SIDE_EXPAND = 0.30

SEVERITY_TO_BIN = {0: '안전', 1: '주의', 2: '위험', 3: '치명'}

# ====== 최소 유효 거리 필터 (ego 오분류 방어) ======
# 2m 이내 거리는 바이크 본체 / 라이더 / 핸들바 / 사이드미러 반사 등으로 간주하여
# 모든 경고 대상에서 제외한다.
MIN_DIST_FCW = 2.0
MIN_DIST_BSW = 2.0
MIN_DIST_HEAD = 2.0


# ============================================================
# Lane Detection + VP (Method L)
# ============================================================
def detect_lane_lines(mask: np.ndarray, min_seg: int = 30) -> list:
    """Lane mark (class 3) → Hough → cluster → unified lines.
    각 line dict: {dydx, c, y_top, y_bot, length}."""
    lane_bin = (mask == 3).astype(np.uint8) * 255
    if lane_bin.sum() < 100:
        return []

    k = np.ones((3, 3), np.uint8)
    lane_clean = cv2.morphologyEx(lane_bin, cv2.MORPH_OPEN, k)

    lines_raw = cv2.HoughLinesP(
        lane_clean, rho=1, theta=np.pi / 180,
        threshold=40, minLineLength=min_seg, maxLineGap=25)
    if lines_raw is None:
        return []
    # OpenCV 4.x 는 (N, 1, 4), 5.x 는 (N, 4) 로 반환 → (N, 4) 로 통일 (노트북 대비 유일한 코드 변경)
    lines_raw = np.asarray(lines_raw).reshape(-1, 4)

    segs = []
    for l in lines_raw:
        x1, y1, x2, y2 = l
        if abs(x2 - x1) < 2 and abs(y2 - y1) < 2: continue
        if abs(y2 - y1) < 5: continue  # 수평선 제외
        dydx = (x2 - x1) / (y2 - y1)
        c = x1 - dydx * y1
        length = float(np.hypot(x2 - x1, y2 - y1))
        y_top, y_bot = min(y1, y2), max(y1, y2)
        x_mid = dydx * ((y_top + y_bot) / 2) + c
        segs.append((dydx, c, y_top, y_bot, length, x_mid))

    segs.sort(key=lambda s: -s[4])
    clusters = []
    for s in segs:
        dydx, c, y_top, y_bot, length, x_mid = s
        merged = False
        for cl in clusters:
            if abs(cl['dydx_avg'] - dydx) < 0.2 and abs(cl['x_mid_avg'] - x_mid) < 40:
                cl['members'].append(s)
                total_len = sum(m[4] for m in cl['members'])
                cl['dydx_avg'] = sum(m[0] * m[4] for m in cl['members']) / total_len
                cl['x_mid_avg'] = sum(m[5] * m[4] for m in cl['members']) / total_len
                merged = True
                break
        if not merged:
            clusters.append({'members': [s], 'dydx_avg': dydx, 'x_mid_avg': x_mid})

    lines = []
    for cl in clusters:
        members = cl['members']
        total_len = sum(m[4] for m in members)
        if total_len < 40: continue
        dydx = cl['dydx_avg']
        c_avg = sum(m[1] * m[4] for m in members) / total_len
        y_top = min(m[2] for m in members)
        y_bot = max(m[3] for m in members)
        lines.append({
            'dydx': float(dydx),
            'c': float(c_avg),
            'y_top': float(y_top),
            'y_bot': float(y_bot),
            'length': float(total_len),
        })
    return lines


def line_x_at(line: dict, y: float) -> float:
    return line['dydx'] * y + line['c']


def vp_from_lanes(lines: list, W: int, H: int) -> tuple:
    """Method L — pairwise 교점 length-weighted median.
    Returns (vp_x, vp_y, confidence)."""
    if len(lines) < 2:
        return W // 2, int(H * 0.4), 0.0

    inters = []
    weights = []
    for i, j in combinations(range(len(lines)), 2):
        l1, l2 = lines[i], lines[j]
        if abs(l2['dydx'] - l1['dydx']) < 1e-4: continue
        vy = (l1['c'] - l2['c']) / (l2['dydx'] - l1['dydx'])
        vx = l1['dydx'] * vy + l1['c']
        if not (-W * 0.2 <= vx <= W * 1.2): continue
        if not (-H * 0.5 <= vy <= H * 0.8): continue
        inters.append((vx, vy))
        weights.append(l1['length'] * l2['length'])

    if len(inters) == 0:
        return W // 2, int(H * 0.4), 0.0

    inters = np.array(inters)
    weights = np.array(weights)
    weights /= weights.sum()
    order_x = np.argsort(inters[:, 0])
    cw = np.cumsum(weights[order_x])
    vx = inters[order_x[np.searchsorted(cw, 0.5)], 0]
    order_y = np.argsort(inters[:, 1])
    cw = np.cumsum(weights[order_y])
    vy = inters[order_y[np.searchsorted(cw, 0.5)], 1]

    std_x = float(np.std(inters[:, 0])) / W
    conf = float(np.clip(1.0 - std_x * 2, 0.0, 1.0))
    return int(vx), int(vy), conf


def identify_ego_lane(lines: list, W: int, H: int, sample_y: float = None) -> dict:
    """이미지 하단에서 W/2 주변 좌/우 가장 가까운 line 한 쌍.
    Returns {left, right, conf}. Symmetry fallback 포함."""
    if len(lines) == 0:
        return {'left': None, 'right': None, 'conf': 0.0}

    sy = sample_y if sample_y is not None else H * 0.85
    left_cands, right_cands = [], []
    for l in lines:
        x_at_sy = l['dydx'] * sy + l['c']
        if x_at_sy < W * 0.5:
            left_cands.append((abs(x_at_sy - W * 0.5), l))
        else:
            right_cands.append((abs(x_at_sy - W * 0.5), l))

    left = min(left_cands, key=lambda t: t[0])[1] if left_cands else None
    right = min(right_cands, key=lambda t: t[0])[1] if right_cands else None

    if left and right:
        return {'left': left, 'right': right, 'conf': 1.0}
    elif left:
        right_synth = {
            'dydx': -left['dydx'],
            'c': W - left['c'],
            'y_top': left['y_top'], 'y_bot': left['y_bot'],
            'length': left['length'] * 0.5,
        }
        return {'left': left, 'right': right_synth, 'conf': 0.5}
    elif right:
        left_synth = {
            'dydx': -right['dydx'],
            'c': W - right['c'],
            'y_top': right['y_top'], 'y_bot': right['y_bot'],
            'length': right['length'] * 0.5,
        }
        return {'left': left_synth, 'right': right, 'conf': 0.5}
    return {'left': None, 'right': None, 'conf': 0.0}


def check_lane_viol_strict(bbox, ego: dict, H: int) -> str:
    """bbox 가 ego lane 의 좌/우 line 가로지름 → 'from_left'/'from_right'/''."""
    if ego['left'] is None or ego['right'] is None:
        return ''
    x0, y0, x1, y1 = bbox
    sample_ys = np.linspace(y0, y1, 10)
    straddle_left = 0
    straddle_right = 0
    for y in sample_ys:
        lx = line_x_at(ego['left'], y)
        rx = line_x_at(ego['right'], y)
        if x0 < lx < x1: straddle_left += 1
        if x0 < rx < x1: straddle_right += 1
    if straddle_left >= 3: return 'from_left'
    if straddle_right >= 3: return 'from_right'
    return ''


def check_lane_viol_x1(bbox, W: int) -> str:
    """이미지 중앙 가로지름 + 너비 > 30%W → 'from_left'/'from_right'/''."""
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    if bw < 0.30 * W: return ''
    if not (x0 < W / 2 < x1): return ''
    cx = (x0 + x1) / 2
    return 'from_right' if cx > W / 2 else 'from_left'


# ============================================================
# Object Extraction
# ============================================================
def extract_objects(mask: np.ndarray, min_area: int = 200) -> list:
    """Connected Components 로 객체 분리.
    Returns: list of {class_id, class_name, bbox, area, cx, cy}."""
    objs = []
    for cid in (4, 5, 6):
        bin_m = (mask == cid).astype(np.uint8)
        if bin_m.sum() < min_area: continue
        n, lbl = cv2.connectedComponents(bin_m, connectivity=8)
        for k in range(1, n):
            ys, xs = np.where(lbl == k)
            if len(xs) < min_area: continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            objs.append({
                'class_id': int(cid),
                'class_name': CLASS_NAMES[cid],
                'bbox': (x0, y0, x1, y1),
                'area': int(len(xs)),
                'cx': int((x0 + x1) / 2),
                'cy': int((y0 + y1) / 2),
            })
    return objs


# ============================================================
# Distance Estimator (3-Estimator Fusion)
# ============================================================
def estimate_focal_from_lanes(lanes, vp_y, img_w, img_h, min_sample=2):
    """차선폭 3.5m prior 로 focal 역산. Mobileye Gen 1 방식."""
    if len(lanes) < 2: return None

    cx = img_w / 2
    # VP 근사로 중앙 거리 체크
    if len(lanes) >= 2:
        xs = [l['dydx'] * vp_y + l['c'] for l in lanes]
        approx_vp_x = float(np.median(xs))
        if abs(approx_vp_x - cx) > img_w * 0.2:
            return None  # VP 가 너무 치우치면 공식 성립 안 함

    y_sample = img_h * 0.9
    left_lane, right_lane = None, None
    for l in lanes:
        x_sample = l['dydx'] * y_sample + l['c']
        if x_sample < cx - 20:
            if left_lane is None or abs(x_sample - cx) < abs(left_lane['dydx'] * y_sample + left_lane['c'] - cx):
                left_lane = l
        elif x_sample > cx + 20:
            if right_lane is None or abs(x_sample - cx) < abs(right_lane['dydx'] * y_sample + right_lane['c'] - cx):
                right_lane = l
    if left_lane is None or right_lane is None: return None

    y_check = img_h * 0.9
    xl_check = left_lane['dydx'] * y_check + left_lane['c']
    xr_check = right_lane['dydx'] * y_check + right_lane['c']
    if xl_check >= xr_check: return None

    consts = []
    for y_frac in (0.75, 0.85, 0.95):
        y = img_h * y_frac
        if y <= vp_y + 20: continue
        x_left = left_lane['dydx'] * y + left_lane['c']
        x_right = right_lane['dydx'] * y + right_lane['c']
        w_px = x_right - x_left
        if w_px < 20: continue
        if x_left < -img_w * 0.5 or x_right > img_w * 1.5: continue
        const = w_px * (y - vp_y)
        if const <= 0: continue
        consts.append(const)
    if len(consts) < min_sample: return None

    const = float(np.median(consts))
    f_px = const * CAMERA_HEIGHT_M / LANE_WIDTH_M
    if not (img_w * 0.3 < f_px < img_w * 2.5): return None
    return float(f_px)


def distance_from_ground(bbox, vp_y, f_px, camera_h=CAMERA_HEIGHT_M):
    """d = f_px · h / (y_bottom - vp_y)."""
    x0, y0, x1, y1 = bbox
    dy = y1 - vp_y
    if dy <= 5: return None
    d = f_px * camera_h / dy
    if not (0.5 < d < 200): return None
    return float(d)


def classify_object(bbox, class_id):
    """bbox aspect → coarse class ('car', 'truck', 'motorcycle', etc)."""
    x0, y0, x1, y1 = bbox
    w = max(x1 - x0, 1); h = max(y1 - y0, 1)
    ar = w / h
    if class_id == 5: return 'motorcycle'
    if class_id == 6: return 'person'
    if ar > 2.5: return 'truck'
    if ar > 0.7: return 'car'
    return 'motorcycle'


def distance_from_size(bbox, obj_class, f_px):
    """d = f_px · W_real / w_bbox_px."""
    x0, y0, x1, y1 = bbox
    w_px = max(x1 - x0, 1)
    W_real = OBJECT_WIDTH_M.get(obj_class, 1.8)
    d = f_px * W_real / w_px
    if not (0.5 < d < 200): return None
    return float(d)


def distance_from_depth(depth_map, bbox, alpha=30.0, beta=0.0, pct=90, is_metric=False):
    """is_metric=True (Depth Pro): bbox median (m).
    is_metric=False (DA V2): inverse depth → α/p + β."""
    x0, y0, x1, y1 = bbox
    patch = depth_map[y0:y1+1, x0:x1+1]
    if patch.size == 0: return None

    if is_metric:
        valid = patch[(patch > 0.3) & (patch < 100)]
        if valid.size < 5: return None
        d = float(np.median(valid))
        if not (0.3 < d < 100): return None
        return d

    p = float(np.percentile(patch, pct)) / 255.0
    if p < 0.05: return None
    d = alpha / p + beta
    if not (0.5 < d < 200): return None
    return float(d)


def fuse_distances(estimates, weights=None, depth_is_metric=False):
    """3-estimator fusion with outlier reject + weighted mean."""
    if weights is None:
        weights = ESTIMATOR_WEIGHTS_METRIC if depth_is_metric else ESTIMATOR_WEIGHTS_INVERSE

    valid = {k: v for k, v in estimates.items() if v is not None}
    n_valid = len(valid)
    if n_valid == 0:
        return {'d_fused_m': None, 'd_bin': 'unknown', 'd_bin_idx': -1,
                'confidence': 'none', 'n_valid': 0, 'estimates': estimates}

    values = list(valid.values())
    med = float(np.median(values))
    filtered = {k: v for k, v in valid.items() if 0.33 * med < v < 3.0 * med}
    if len(filtered) == 0: filtered = valid

    total_w = sum(weights[k] for k in filtered)
    d_fused = sum(filtered[k] * weights[k] / total_w for k in filtered)

    d_bin_idx = 3
    for i, (lo, hi) in enumerate(DISTANCE_BINS):
        if lo <= d_fused < hi:
            d_bin_idx = i; break

    bins_agreed = set()
    for v in valid.values():
        for i, (lo, hi) in enumerate(DISTANCE_BINS):
            if lo <= v < hi: bins_agreed.add(i); break

    if n_valid >= 3 and len(bins_agreed) == 1: conf = 'high'
    elif n_valid >= 2 and d_bin_idx in bins_agreed: conf = 'medium'
    elif n_valid >= 1: conf = 'low'
    else: conf = 'none'

    return {
        'd_fused_m': round(d_fused, 2),
        'd_bin': DISTANCE_BIN_NAMES[d_bin_idx],
        'd_bin_idx': d_bin_idx,
        'confidence': conf,
        'n_valid': n_valid,
        'estimates': {k: round(v, 2) if v else None for k, v in estimates.items()},
    }


class DistanceEstimator:
    """3-estimator fusion 거리 추정기."""

    def __init__(self, depth_is_metric: bool = False):
        self.f_px: Optional[float] = None
        self.camera_h: float = CAMERA_HEIGHT_M
        self.alpha: float = 30.0
        self.beta: float = 0.0
        self.depth_is_metric: bool = depth_is_metric

    def calibrate(self, lanes, vp_y, img_w, img_h):
        f = estimate_focal_from_lanes(lanes, vp_y, img_w, img_h)
        if f is None:
            self.f_px = img_w * DEFAULT_FOCAL_RATIO
            return False
        self.f_px = f
        return True

    def calibrate_depth_alpha(self, depth_map, moveables, vp_y):
        """DA V2 α,β 를 d_ground 로 fit. Depth Pro 면 skip."""
        if self.depth_is_metric: return
        if self.f_px is None or len(moveables) < 2: return
        pts = []
        for o in moveables:
            d_g = distance_from_ground(o['bbox'], vp_y, self.f_px, self.camera_h)
            if d_g is None: continue
            x0, y0, x1, y1 = o['bbox']
            patch = depth_map[y0:y1+1, x0:x1+1]
            if patch.size == 0: continue
            p = float(np.percentile(patch, 90)) / 255.0
            if p < 0.05: continue
            pts.append((1.0 / p, d_g))
        if len(pts) < 2: return
        X = np.array([p[0] for p in pts])
        Y = np.array([p[1] for p in pts])
        try:
            alpha, beta = np.polyfit(X, Y, 1)
            if 1 < alpha < 200:
                self.alpha = float(alpha)
                self.beta = float(beta)
        except Exception:
            pass

    def estimate(self, bbox, class_id, depth_map, vp_y):
        if self.f_px is None:
            self.f_px = float(DEFAULT_FOCAL_RATIO * 1000)  # 대략 fallback

        obj_class = classify_object(bbox, class_id)
        d_ground = distance_from_ground(bbox, vp_y, self.f_px, self.camera_h)
        d_size = distance_from_size(bbox, obj_class, self.f_px)
        d_depth = None
        if depth_map is not None:
            d_depth = distance_from_depth(
                depth_map, bbox, self.alpha, self.beta,
                is_metric=self.depth_is_metric)
        fused = fuse_distances({'ground': d_ground, 'size': d_size, 'depth': d_depth},
                               depth_is_metric=self.depth_is_metric)
        fused['obj_class'] = obj_class
        return fused


# ============================================================
# 4-Zone Ego Corridor
# ============================================================
def _y_at(vp_y, alpha, H):
    return vp_y + alpha * (H - 1 - vp_y)


def _lane_x_at_zone(vp_x, vp_y, x_bottom, H, y):
    denom = max(H - 1 - vp_y, 1e-6)
    t = (y - vp_y) / denom
    return vp_x + t * (x_bottom - vp_x)


def build_corridor(vp_x, vp_y, x_left_bottom, x_right_bottom, H, W,
                   vp_conf=1.0, alphas=None, side_expand=SIDE_EXPAND):
    """4-zone nested polygon + sidelobe.
    Returns dict 또는 None (vp_conf < 0.3 또는 lane_w < 20 시)."""
    if vp_conf < 0.3: return None
    if alphas is None: alphas = DEFAULT_ALPHAS

    y_bot = H - 1
    Lx, Rx = x_left_bottom, x_right_bottom
    if Rx - Lx < 20: return None

    def trapezoid(alpha_top):
        yt = _y_at(vp_y, alpha_top, H)
        xtl = _lane_x_at_zone(vp_x, vp_y, Lx, H, yt)
        xtr = _lane_x_at_zone(vp_x, vp_y, Rx, H, yt)
        return np.array([[xtl, yt], [xtr, yt],
                         [Rx, y_bot], [Lx, y_bot]], dtype=np.int32)

    polys = {
        'critical': trapezoid(alphas['critical']),
        'danger':   trapezoid(alphas['danger']),
        'caution':  trapezoid(alphas['caution']),
    }
    lane_w_bot = Rx - Lx
    dx = side_expand * lane_w_bot
    polys['sidelobe_left'] = np.array(
        [[vp_x, vp_y], [Lx, y_bot], [Lx - dx, y_bot]], dtype=np.int32)
    polys['sidelobe_right'] = np.array(
        [[vp_x, vp_y], [Rx + dx, y_bot], [Rx, y_bot]], dtype=np.int32)

    masks = {}
    for name, poly in polys.items():
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [poly], 1)
        masks[name] = m

    return {'polygons': polys, 'masks': masks,
            'vp': (int(vp_x), int(vp_y)),
            'lane_bottom': (int(Lx), int(Rx)),
            'alphas': alphas}


def classify_zone(bbox, corridor, soft_promote_ratio=0.30):
    """Bbox → severity (0=safe, 1=caution, 2=danger, 3=critical)."""
    info = {
        'anchor_zone': 'safe', 'bbox_ratios': {},
        'sidelobe_hit': None, 'sidelobe_ratio': 0.0, 'in_corridor': False,
    }
    if corridor is None: return 0, info

    x0, y0, x1, y1 = bbox
    H, W = corridor['masks']['critical'].shape
    foot_x = max(0, min(W - 1, int((x0 + x1) / 2)))
    foot_y = max(0, min(H - 1, int(y1)))

    anchor = 'safe'
    for name in ('critical', 'danger', 'caution'):
        if corridor['masks'][name][foot_y, foot_x] > 0:
            anchor = name; break
    info['anchor_zone'] = anchor
    info['in_corridor'] = anchor != 'safe'

    bbox_area = max((x1 - x0) * (y1 - y0), 1)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0: return 0, info

    for name in ('critical', 'danger', 'caution',
                 'sidelobe_left', 'sidelobe_right'):
        patch = corridor['masks'][name][cy0:cy1, cx0:cx1]
        info['bbox_ratios'][name] = round(int(patch.sum()) / bbox_area, 3)

    sev_map = {'safe': 0, 'caution': 1, 'danger': 2, 'critical': 3}
    sev = sev_map[anchor]

    if info['bbox_ratios']['critical'] > soft_promote_ratio:
        sev = max(sev, 3)
    elif info['bbox_ratios']['danger'] > soft_promote_ratio:
        sev = max(sev, 2)
    elif info['bbox_ratios']['caution'] > soft_promote_ratio:
        sev = max(sev, 1)

    sl_max = max(info['bbox_ratios']['sidelobe_left'],
                 info['bbox_ratios']['sidelobe_right'])
    if sl_max > 0.40:
        info['sidelobe_hit'] = ('left' if info['bbox_ratios']['sidelobe_left']
                                > info['bbox_ratios']['sidelobe_right'] else 'right')
        info['sidelobe_ratio'] = round(sl_max, 3)

    return sev, info


# ============================================================
# 4 Warning Modules
# ============================================================
@dataclass
class WarningResult:
    name: str
    level: int
    reason: str = ''
    bbox: Optional[tuple] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {'name': self.name, 'level': self.level, 'reason': self.reason,
                'bbox': list(self.bbox) if self.bbox else None, **self.extra}


@dataclass
class V5State:
    fcw: WarningResult
    ldw: WarningResult
    bsw: WarningResult
    head: WarningResult
    d_front_m: Optional[float] = None
    d_left_m: Optional[float] = None
    d_right_m: Optional[float] = None
    f_calibrated: bool = False
    f_px: Optional[float] = None

    @property
    def max_severity(self) -> int:
        return max(self.fcw.level, self.ldw.level, self.bsw.level, self.head.level)

    @property
    def bin(self) -> str:
        return SEVERITY_TO_BIN[self.max_severity]

    @property
    def active_warnings(self) -> list:
        return [w for w in (self.fcw, self.ldw, self.bsw, self.head) if w.level > 0]

    @property
    def top_warning(self) -> Optional[WarningResult]:
        act = self.active_warnings
        if not act: return None
        return max(act, key=lambda w: w.level)


def detect_fcw(front_objs, img_w, img_h):
    """정면 가장 가까운 객체 기반."""
    if not front_objs: return WarningResult('FCW', 0)
    obj = min(front_objs, key=lambda o: o['dist_info']['d_fused_m'] or 999)
    d = obj['dist_info']['d_fused_m']
    if d is None: return WarningResult('FCW', 0, 'no distance estimate')

    x0, y0, x1, y1 = obj['bbox']
    area_ratio = ((x1 - x0) * (y1 - y0)) / (img_w * img_h)

    if d < FCW_DIST_CRITICAL or area_ratio > FCW_BBOX_RATIO_CRITICAL:
        return WarningResult('FCW', 3,
            f'정면 {d:.1f}m (bbox {area_ratio*100:.0f}%)',
            bbox=obj['bbox'],
            extra={'distance_m': round(d, 2), 'area_ratio': round(area_ratio, 3)})
    elif d < FCW_DIST_WARNING:
        return WarningResult('FCW', 2, f'정면 {d:.1f}m', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2)})
    elif d < FCW_DIST_CAUTION:
        return WarningResult('FCW', 1, f'정면 {d:.1f}m', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2)})
    return WarningResult('FCW', 0)


def detect_ldw(ego, img_w, img_h, vp_y, ego_x=None):
    """ego_x vs ego_lane center / lane_w 기반 편향·이탈."""
    if ego['left'] is None or ego['right'] is None:
        return WarningResult('LDW', 0, 'ego_lane 식별 실패')

    y_check = img_h * 0.95
    lx = line_x_at(ego['left'], y_check)
    rx = line_x_at(ego['right'], y_check)
    lane_w = rx - lx
    if lane_w <= 20:
        return WarningResult('LDW', 0, 'ego_lane 너비 이상')
    lane_center = (lx + rx) / 2
    if ego_x is None: ego_x = img_w / 2

    lateral_offset = ego_x - lane_center
    lateral_ratio = abs(lateral_offset) / lane_w

    # 측정 불가 guard (sus seg 방어)
    if lateral_ratio > 0.5:
        return WarningResult('LDW', 0,
            f'측정 불가 (편향 {lateral_ratio*100:.0f}%, 감지 오류 의심)',
            extra={'lateral_ratio': round(lateral_ratio, 3), 'unreliable': True})

    direction = '우측' if lateral_offset > 0 else '좌측'
    # L2 이탈 제거: 30% 이상 편향은 confidence 부족으로 측정 불가 처리
    # (single-frame 으로 이탈 확정 어려움, FP 방지)
    if lateral_ratio > LDW_LATERAL_WARNING or ego_x < lx or ego_x > rx:
        return WarningResult('LDW', 0,
            f'[측정 불가] {direction} 이탈 의심 ({lateral_ratio*100:.0f}%, 확정 불가)',
            extra={'lateral_ratio': round(lateral_ratio, 3),
                   'unreliable': True, 'reason_detail': 'possible_departure'})
    elif lateral_ratio > LDW_LATERAL_CAUTION:
        return WarningResult('LDW', 1, f'{direction} 편향 ({lateral_ratio*100:.0f}%)',
            extra={'lateral_ratio': round(lateral_ratio, 3),
                   'ego_x': int(ego_x), 'lane_center': int(lane_center)})
    return WarningResult('LDW', 0,
        extra={'lateral_ratio': round(lateral_ratio, 3)})


def detect_bsw(left_objs, right_objs, strict_violators, x1_violators):
    """측면 가장 가까운 객체 + 침범 여부."""
    side_objs = left_objs + right_objs
    if not side_objs: return WarningResult('BSW', 0)
    obj = min(side_objs, key=lambda o: o['dist_info']['d_fused_m'] or 999)
    d = obj['dist_info']['d_fused_m']
    if d is None: return WarningResult('BSW', 0, 'no distance')
    side = '좌측' if obj['cx_rel'] < 0.5 else '우측'
    violating = any(o['bbox'] == obj['bbox'] for o in strict_violators + x1_violators)

    if d < BSW_DIST_WARNING and violating:
        return WarningResult('BSW', 3, f'{side} 침범 {d:.1f}m', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2), 'side': side, 'violating': True})
    elif d < BSW_DIST_WARNING:
        return WarningResult('BSW', 2, f'{side} 근접 {d:.1f}m', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2), 'side': side})
    elif d < BSW_DIST_CAUTION:
        return WarningResult('BSW', 1, f'{side} {d:.1f}m', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2), 'side': side})
    return WarningResult('BSW', 0)


def detect_head(front_objs, strict_violators, x1_violators, img_w, img_h):
    """정면 역주행 (cut-in 은 제외)."""
    if not front_objs: return WarningResult('HEAD', 0)
    violator_bboxes = [o['bbox'] for o in strict_violators + x1_violators]
    head_cands = [o for o in front_objs if o['bbox'] in violator_bboxes]
    if not head_cands: return WarningResult('HEAD', 0, 'no front violator')

    obj = min(head_cands, key=lambda o: o['dist_info']['d_fused_m'] or 999)
    d = obj['dist_info']['d_fused_m']
    if d is None: return WarningResult('HEAD', 0)

    x0, y0, x1, y1 = obj['bbox']
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    aspect = bw / bh
    area_ratio = (bw * bh) / (img_w * img_h)
    cx_rel = ((x0 + x1) / 2) / img_w

    is_frontal_center = (0.45 <= cx_rel <= 0.55)   # 중앙 10% 만
    is_frontal_aspect = (aspect >= 2.0)           # 넓은 정면 차량만
    if not (is_frontal_center and is_frontal_aspect):
        return WarningResult('HEAD', 0,
            f'cut-in 가능성 (cx={cx_rel:.2f}, asp={aspect:.1f})',
            extra={'skipped_cut_in': True, 'cx_rel': round(cx_rel, 2),
                   'aspect': round(aspect, 2)})

    # 거리가 너무 멀면 HEAD 발동 안 함 (극보수화)
    if d is None or d > 4.0:
        return WarningResult('HEAD', 0, f'cut-in 가능성 (거리 {d}m)')

    if d < HEAD_DIST_CRITICAL:
        return WarningResult('HEAD', 3,
            f'역주행 정면 {d:.1f}m (aspect {aspect:.1f})', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2), 'aspect': round(aspect, 2),
                   'area_ratio': round(area_ratio, 3)})
    elif d < HEAD_DIST_WARNING:
        return WarningResult('HEAD', 2, f'역주행 의심 {d:.1f}m', bbox=obj['bbox'],
            extra={'distance_m': round(d, 2)})
    return WarningResult('HEAD', 0)


def _warning_distance_m(w):
    """WarningResult 에서 거리(m) 추출 — extra 우선, 없으면 reason 파싱."""
    extra = getattr(w, 'extra', None) or {}
    for k in ('distance_m', 'dist_m', 'd_m'):
        v = extra.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    reason = getattr(w, 'reason', '') or ''
    m = re.search(r'(\d+\.?\d*)\s*m', reason)
    return float(m.group(1)) if m else None


def suppress_short_range_warnings(state, min_valid=2.0):
    """2m 이내 거리는 ego 오분류로 간주:
      - 해당 warning 의 level 을 0 으로 downgrade
      - state.d_front_m / d_left_m / d_right_m 도 2m 미만이면 None 처리
    state 는 in-place 수정."""
    thresholds = {'FCW': MIN_DIST_FCW, 'BSW': MIN_DIST_BSW, 'HEAD': MIN_DIST_HEAD}
    for w in [state.fcw, state.bsw, state.head]:
        if w.level == 0:
            continue
        min_d = thresholds.get(w.name, min_valid)
        dist = _warning_distance_m(w)
        if dist is not None and dist < min_d:
            orig = w.reason or f'L{w.level}'
            w.level = 0
            w.reason = f'[억제] {orig} < {min_d}m (ego 오분류 의심)'
            w.bbox = None
            if w.extra is None:
                w.extra = {}
            w.extra['suppressed_short_range'] = True

    # 위치별 거리도 일관성 맞춤 — 2m 미만은 None 처리
    for attr in ('d_front_m', 'd_left_m', 'd_right_m'):
        v = getattr(state, attr, None)
        if v is not None and v < min_valid:
            try:
                setattr(state, attr, None)
            except (AttributeError, TypeError):
                pass


# ============================================================
# Main Pipeline — assess_v5()
# ============================================================
def get_ego_mask(mask):
    """My bike (5) + Rider (6) → close 5x5 → dilate 5x5 → CC closest to (W/2, H).
    Returns (H, W) uint8."""
    mh, mw = mask.shape
    bike_base = ((mask == 5) | (mask == 6)).astype(np.uint8)
    if bike_base.sum() < 200:
        return np.zeros_like(bike_base)
    k5 = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(bike_base, cv2.MORPH_CLOSE, k5)
    dilated = cv2.dilate(closed, np.ones((5, 5), np.uint8), iterations=1)
    n_cc, labels = cv2.connectedComponents(dilated, connectivity=8)
    if n_cc <= 1:
        return dilated
    anchor = np.array([mw / 2, mh])
    best_cc, best_dist = 1, float('inf')
    for cc_id in range(1, n_cc):
        ys_cc, xs_cc = np.where(labels == cc_id)
        if len(xs_cc) < 200: continue
        centroid = np.array([xs_cc.mean(), ys_cc.mean()])
        dist = np.linalg.norm(centroid - anchor)
        if dist < best_dist:
            best_dist = dist; best_cc = cc_id
    return (labels == best_cc).astype(np.uint8)


def assess_v5(img_np, mask, depth_map, img_name='', focal_px=None,
              depth_is_metric=True):
    """v5 메인 파이프라인.

    img_np:        (H, W, 3) RGB uint8
    mask:          (H, W) uint8, 7-class
    depth_map:     (H, W) float32 또는 None
    focal_px:      Depth Pro 의 focal (metric 시), 또는 None → 차선폭 역산 시도
    depth_is_metric: True (Depth Pro), False (DA V2)
    """
    H, W = mask.shape

    # 1. Lane → VP → Ego lane
    lines = detect_lane_lines(mask)
    vp_x, vp_y, vp_conf = vp_from_lanes(lines, W, H)
    if vp_conf < 0.3:
        vp_x, vp_y = W // 2, int(H * 0.4)
    ego = identify_ego_lane(lines, W, H)

    # 2. Distance estimator init
    est = DistanceEstimator(depth_is_metric=depth_is_metric)
    if focal_px is not None and focal_px > 0:
        est.f_px = float(focal_px)
        f_ok = True
    else:
        f_ok = est.calibrate(lines, vp_y, W, H)

    # 3. Object extraction + ego mask filter
    all_objs = extract_objects(mask)
    raw_moveables = [o for o in all_objs if o['class_id'] == 4]

    ego_mask = get_ego_mask(mask)

    def is_ego_fragment(bbox):
        if ego_mask.sum() == 0: return False
        x0, y0, x1, y1 = bbox
        patch = ego_mask[y0:y1+1, x0:x1+1]
        overlap = int(patch.sum())
        bbox_px = max((x1 - x0) * (y1 - y0), 1)
        thr = 0.55 if (x1 - x0) * (y1 - y0) >= 3500 else 0.40
        return overlap / bbox_px > thr

    moveables = [o for o in raw_moveables if not is_ego_fragment(o['bbox'])]
    n_filtered = len(raw_moveables) - len(moveables)

    if depth_map is not None and len(moveables) >= 2:
        est.calibrate_depth_alpha(depth_map, moveables, vp_y)

    # 4. Zone corridor
    corridor = None
    if ego['left'] is not None and ego['right'] is not None and vp_conf >= 0.3:
        y_bot = H - 1
        Lx_bot = line_x_at(ego['left'], y_bot)
        Rx_bot = line_x_at(ego['right'], y_bot)
        corridor = build_corridor(vp_x, vp_y, Lx_bot, Rx_bot, H, W,
                                   vp_conf=vp_conf)

    # 5. Per-object distance + zone
    objs_with_dist = []
    for o in moveables:
        info = est.estimate(o['bbox'], o['class_id'], depth_map, vp_y)
        cx_rel = o['cx'] / W
        zone_sev, zone_info = classify_zone(o['bbox'], corridor)
        objs_with_dist.append({
            'bbox': o['bbox'], 'class_id': o['class_id'],
            'cx': o['cx'], 'cy': o['cy'], 'cx_rel': cx_rel,
            'dist_info': info,
            'zone_sev': zone_sev, 'zone_info': zone_info,
        })

    # 6. 방향별 분류 + violation
    front_objs = [o for o in objs_with_dist if 0.35 <= o['cx_rel'] <= 0.65]
    left_objs = [o for o in objs_with_dist if o['cx_rel'] < 0.35]
    right_objs = [o for o in objs_with_dist if o['cx_rel'] > 0.65]

    strict_violators = []
    x1_violators = []
    if ego['left'] and ego['right']:
        for o in objs_with_dist:
            if check_lane_viol_strict(o['bbox'], ego, H):
                strict_violators.append(o)
    for o in objs_with_dist:
        if check_lane_viol_x1(o['bbox'], W):
            x1_violators.append(o)

    # 7. ego_x = ego_mask centroid (실제 my bike 위치)
    if ego_mask.sum() > 200:
        ys_em, xs_em = np.where(ego_mask > 0)
        ego_cx = float(xs_em.mean())
    else:
        ego_cx = None

    # 8. 4 warnings
    fcw = detect_fcw(front_objs, W, H)
    ldw = detect_ldw(ego, W, H, vp_y, ego_x=ego_cx)
    bsw = detect_bsw(left_objs, right_objs, strict_violators, x1_violators)
    head = detect_head(front_objs, strict_violators, x1_violators, W, H)

    # 9. Zone clamp patches
    zone_adjustments = []
    if corridor is not None:
        # Patch 1: FCW L3 by area_ratio override → zone 안일 때만 유지 (cut-in 예외)
        cut_in_scenario = (bsw.level >= 2 or head.level >= 2)
        if (fcw.level == 3 and fcw.extra.get('area_ratio', 0) > 0.25
                and fcw.bbox is not None and not cut_in_scenario):
            matching = next(
                (o for o in objs_with_dist if tuple(o['bbox']) == tuple(fcw.bbox)),
                None)
            if matching is not None and matching['zone_sev'] < 2:
                fcw = WarningResult(
                    'FCW', 2, fcw.reason + ' [zone clamp→L2]',
                    bbox=fcw.bbox,
                    extra={**fcw.extra, 'zone_clamped': True,
                           'zone_sev': matching['zone_sev']})
                zone_adjustments.append('FCW L3→L2 (zone clamp, no cut-in)')

        # Patch 2: BSW sidelobe boost
        if bsw.level == 0:
            for o in objs_with_dist:
                sl_hit = o['zone_info'].get('sidelobe_hit')
                if sl_hit and o['zone_info'].get('sidelobe_ratio', 0) > 0.45:
                    d = o['dist_info'].get('d_fused_m')
                    d_str = f'{d:.1f}m' if d else '거리불명'
                    bsw = WarningResult(
                        'BSW', 2, f'{sl_hit} sidelobe {d_str} [zone boost]',
                        bbox=o['bbox'],
                        extra={'sidelobe': True, 'side': sl_hit,
                               'zone_boost': True, 'distance_m': d})
                    zone_adjustments.append(f'BSW L0→L2 ({sl_hit} sidelobe)')
                    break

    # 10. Distance summary
    def closest_d(lst):
        ds = [o['dist_info']['d_fused_m'] for o in lst
              if o['dist_info']['d_fused_m'] is not None]
        return min(ds) if ds else None

    state = V5State(
        fcw=fcw, ldw=ldw, bsw=bsw, head=head,
        d_front_m=closest_d(front_objs),
        d_left_m=closest_d(left_objs),
        d_right_m=closest_d(right_objs),
        f_calibrated=f_ok,
        f_px=round(est.f_px, 1) if est.f_px else None,
    )

    # 2m 이내는 ego 오분류 가정 — warning 억제 + 거리값 clamp
    suppress_short_range_warnings(state)
    # HEAD 사후 검증: FCW 동시 활성 & 근거리 일치하지 않으면 억제 (극보수화)
    if state.head.level > 0:
        fcw_ok = state.fcw.level > 0 and (
            state.fcw.extra and state.fcw.extra.get('distance_m', 999) <= 4.5)
        if not fcw_ok:
            state.head.level = 0
            state.head.reason = '[억제] FCW 동시 활성 조건 미충족 (역주행 확신 부족)'
            state.head.bbox = None
    # [removed] lane violator → BSW 승격 — FP 많아서 제거 (노트북과 동일하게 비활성).

    return {
        'image': img_name,
        'bin': state.bin,
        'max_severity': state.max_severity,
        'warnings': {
            'FCW': fcw.to_dict(), 'LDW': ldw.to_dict(),
            'BSW': bsw.to_dict(), 'HEAD': head.to_dict(),
        },
        'active_warnings': [w.name for w in state.active_warnings],
        'top_warning': state.top_warning.name if state.top_warning else None,
        'top_reason': state.top_warning.reason if state.top_warning else '',
        'd_front_m': round(state.d_front_m, 2) if state.d_front_m else None,
        'd_left_m': round(state.d_left_m, 2) if state.d_left_m else None,
        'd_right_m': round(state.d_right_m, 2) if state.d_right_m else None,
        'f_calibrated': state.f_calibrated,
        'f_px': state.f_px,
        'vp': {'x': int(vp_x), 'y': int(vp_y), 'conf': round(float(vp_conf), 3)},
        'n_lines': len(lines),
        'n_moveable': len(moveables),
        'n_strict_viol': len(strict_violators),
        'n_x1_viol': len(x1_violators),
        'n_ego_fragments_filtered': n_filtered,
        'corridor_active': corridor is not None,
        'zone_adjustments': zone_adjustments,
        'per_obj_zones': [
            {'bbox': list(o['bbox']), 'zone_sev': o['zone_sev'],
             'anchor': o['zone_info']['anchor_zone'],
             'sidelobe': o['zone_info'].get('sidelobe_hit')}
            for o in objs_with_dist
        ],
        '_state': state,
        '_ego': ego,
        '_lines': lines,
        '_objs': objs_with_dist,
        '_corridor': corridor,
    }


# ============================================================
# Public API
# ============================================================
def assess(image_path, mask_path):
    """이미지 + 7-class mask 파일 → result dict (노트북 assess_v5 반환 구조 그대로).

    노트북 RiskAssessorV5(seg_checkpoints=None, use_depth=None).assess_with_mask() 와 같은 경로:
    depth 없음 → 거리 = ground + size estimator, focal 은 차선폭 역산 (실패 시 W×1.2).
    """
    img_np = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path))
    if mask.ndim != 2:
        raise ValueError(f'mask 는 (H, W) 단일 채널이어야 함: {mask_path} shape={mask.shape}')
    if mask.shape[:2] != img_np.shape[:2]:
        raise ValueError(
            f'mask shape {mask.shape} != image shape {img_np.shape[:2]}')
    return assess_v5(img_np, mask.astype(np.uint8), None,
                     img_name=Path(image_path).name,
                     focal_px=None, depth_is_metric=False)


def result_to_json(result):
    """JSON 직렬화 가능한 dict 만 추출 (내부 객체 키 제거)."""
    return {k: v for k, v in result.items() if not k.startswith('_')}


# ============================================================
# CLI — python -m roadscope.rules samples
# ============================================================
def _json_default(o):
    """numpy 스칼라가 섞여 들어와도 저장되도록."""
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f'JSON 직렬화 불가: {type(o)}')


def _fmt(v):
    if v is None or v == '' or v == []:
        return '—'
    if isinstance(v, list):
        return ', '.join(str(x) for x in v)
    return str(v).replace('|', '\\|')


def _write_compare(rows, expected_path, out_path):
    """expected_outputs.json 대비 비교표 (불일치는 고치지 않고 그대로 기록)."""
    n_match = sum(1 for _, r, e in rows if e and e['bin'] == r['bin'])
    n_exp = sum(1 for _, _, e in rows if e)

    L = []
    L.append('# MVP 결과 vs expected_outputs.json\n')
    L.append(f'- 실행 조건: GT mask 주입 (`samples/masks/`) + depth 없음 (거리 = ground + size)')
    L.append(f'- 기대 출력 조건: U-Net R34 5-fold ensemble seg + Depth Pro (`{expected_path.as_posix()}` `_meta`)')
    L.append(f'- 조건이 다르므로 거리 수치 차이는 예상된 것. 불일치는 수정하지 않고 기록만 한다.')
    L.append(f'- **등급 일치: {n_match}/{n_exp}**\n')

    L.append('## 요약\n')
    L.append('| 이미지 | 기대 등급 | MVP 등급 | 일치 | 기대 활성 경고 | MVP 활성 경고 | 기대 top_reason | MVP top_reason |')
    L.append('|---|---|---|---|---|---|---|---|')
    for name, r, e in rows:
        if e is None:
            L.append(f'| {name} | (기대값 없음) | {r["bin"]} | — | — | {_fmt(r["active_warnings"])} | — | {_fmt(r["top_reason"])} |')
            continue
        mark = '✓' if e['bin'] == r['bin'] else '✗'
        L.append(f'| {name} | {e["bin"]} | {r["bin"]} | {mark} | {_fmt(e["active_warnings"])} | '
                 f'{_fmt(r["active_warnings"])} | {_fmt(e["top_reason"])} | {_fmt(r["top_reason"])} |')
    L.append('')

    for name, r, e in rows:
        L.append(f'## {name}\n')
        L.append('기대 출력에는 경고별 레벨이 없어서 "활성 여부"와 top_reason 만 비교할 수 있다.\n')
        L.append('| 경고 | MVP 레벨 | MVP 문구 | 기대 |')
        L.append('|---|---|---|---|')
        for wname in ('FCW', 'LDW', 'BSW', 'HEAD'):
            w = r['warnings'][wname]
            if e is None:
                exp = '—'
            elif wname in e['active_warnings']:
                exp = '활성'
                if e.get('top_warning') == wname:
                    exp += f' (top: {_fmt(e["top_reason"])})'
            else:
                exp = '비활성'
            L.append(f'| {wname} | L{w["level"]} | {_fmt(w["reason"])} | {exp} |')
        L.append('')
        L.append('| 항목 | 기대 | MVP |')
        L.append('|---|---|---|')
        for key, label in (('d_front_m', '정면 거리(m)'), ('d_left_m', '좌 거리(m)'),
                           ('d_right_m', '우 거리(m)'), ('n_moveable', 'Moveable 수')):
            ev = e.get(key) if e else None
            L.append(f'| {label} | {_fmt(ev)} | {_fmt(r.get(key))} |')
        L.append(f'| f_px (calibrated) | — | {r["f_px"]} ({r["f_calibrated"]}) |')
        L.append(f'| VP (x, y, conf) | — | ({r["vp"]["x"]}, {r["vp"]["y"]}, {r["vp"]["conf"]}) |')
        L.append(f'| corridor / zone 보정 | — | {"ON" if r["corridor_active"] else "OFF"} / {_fmt(r["zone_adjustments"])} |')
        L.append('')

    out_path.write_text('\n'.join(L), encoding='utf-8')


def main(argv):
    if len(argv) != 2:
        print('사용법: python -m roadscope.rules <samples 폴더>')
        return 2
    samples = Path(argv[1])
    img_dir, mask_dir = samples / 'images', samples / 'masks'
    expected_path = samples / 'expected_outputs.json'
    from roadscope import config
    out_dir = config.REPO_ROOT / 'outputs'
    out_dir.mkdir(exist_ok=True)

    expected = {}
    if expected_path.exists():
        data = json.loads(expected_path.read_text(encoding='utf-8'))
        expected = {e['filename']: e for e in data.get('results', [])}

    rows = []
    for img_path in sorted(img_dir.glob('*.png')):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f'[skip] mask 없음: {mask_path}')
            continue
        r = assess(img_path, mask_path)
        out_json = out_dir / f'{img_path.stem}.json'
        out_json.write_text(
            json.dumps(result_to_json(r), indent=1, ensure_ascii=False,
                       default=_json_default),
            encoding='utf-8')
        print(f'{img_path.name}: {r["bin"]}  {r["active_warnings"]}  → {out_json.name}')
        rows.append((img_path.name, r, expected.get(img_path.name)))

    _write_compare(rows, expected_path, out_dir / 'compare.md')
    print(f'비교표 저장: {out_dir / "compare.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
