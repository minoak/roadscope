# SPDX-License-Identifier: AGPL-3.0-only
"""car_models.py — 자동차 모드 사전학습 모델 (t29, DECISIONS 2026-10-09)

- YOLOPv2 (BDD100K, MIT 코드): 매 프레임 차선·주행 가능 영역. 차량 검출 출력은 쓰지 않는다 (YOLO11m 유지).
  엔진의 lane_focus 마스크와 같은 형식 (uint8, 2 = 주행 가능 영역, 3 = 차선) 으로 돌려준다.
- Mask2Former (Mapillary Vistas semantic): 사전 점검에서 몇 프레임만 — Ego Vehicle + Car Mount = 내 차 영역.
전처리·후처리는 t28 모델 시험 (개발 기록, 시험 스크립트는 리포에 없음) 과 YOLOPv2 공식 demo 를 따른다.
가중치 학습 데이터 (BDD100K, Mapillary Vistas) 는 연구용 조건이 붙을 수 있다 (상업 이용 전 확인, DECISIONS).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import cv2
import torch

from roadscope import config

MODELS_DIR = config.MODELS_DIR                            # 환경변수 ROADSCOPE_MODELS_DIR (docs/models.md)
YOLOP_WEIGHTS = MODELS_DIR / 'yolopv2.pt'                 # TorchScript
M2F_MODEL = 'facebook/mask2former-swin-large-mapillary-vistas-semantic'   # HF 캐시 (오프라인)
M2F_EGO_LABELS = ('Ego Vehicle', 'Car Mount')


# ============================================================
# YOLOPv2 — 차선·주행 가능 영역
# ============================================================
def load_yolopv2(device):
    model = torch.jit.load(str(YOLOP_WEIGHTS), map_location=device)
    return model.float().eval()


@torch.inference_mode()
def yolop_mask(model, bgr, device):
    """BGR 프레임 → (H, W) uint8 마스크: 2 = 주행 가능 영역, 3 = 차선 (겹치면 차선), 0 = 나머지.
    공식 demo 와 같이 1280×720 으로 맞춘 뒤 640×384 letterbox (위아래 12px 회색), 출력의 12:372 행을
    2배 bilinear 로 키워 720×1280 마스크를 만들고 원래 크기로 되돌린다."""
    H, W = bgr.shape[:2]
    im = bgr if (W, H) == (1280, 720) else cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).to(device).float()[None] / 255
    _, seg, ll = model(x)
    da = torch.nn.functional.interpolate(seg[:, :, 12:372, :], scale_factor=2, mode='bilinear')
    da = torch.argmax(da, 1)[0]
    ln = torch.nn.functional.interpolate(ll[:, :, 12:372, :], scale_factor=2, mode='bilinear')
    ln = torch.round(ln)[0, 0]
    out = (da > 0).to(torch.uint8) * 2
    out[ln > 0] = 3
    out = out.cpu().numpy()
    if (W, H) != (1280, 720):
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
    return out


def lane_pair_from_mask(lane, max_lines=None):
    """차선 이진 마스크 → 내 차로 좌우 차선 쌍 (t28 probe fit_lane_pair 와 같은 규칙).

    y=0.18H~0.95H 에서 HoughLinesP 씨앗 → 근처 ±6px 차선 픽셀로 x = a·y + b 두 번 다시 맞춤 (40행 이상,
    세로 0.10H 이상, |a| 0.12~4) → 같은 선 합치기 → y=0.85H 에서 화면 중앙을 끼고 바깥으로 벌어지는 쌍
    (교점 x 0.1~0.9W, y 0.05~0.80H, 폭 0.15~1.5W, 교점이 선 관측 시작보다 위). 점수 = 짧은 쪽 행 수 − 0.1×폭.
    속도 (t29): 이미 맞춘 선과 같은 씨앗 (y=0.85H 위치 25px·기울기 0.25 안) 은 다시 맞추지 않고, 행별 대표 x 는
    중앙값 대신 평균 (±6px 창 안이라 차이 작음). probe 의 프레임당 0.4~4.8초 → 수십 ms.
    Returns (vx, vy, a_R − a_L) 또는 None. a = 옆 거리 / 카메라 높이 (평지) 라 a_R − a_L = 차로폭 / Hc.
    """
    h, w = lane.shape
    work = lane.astype(np.uint8).copy()
    work[:int(.18 * h)] = 0
    work[int(.95 * h):] = 0
    if work.sum() < 60:
        return None
    lines = cv2.HoughLinesP(work * 255, 1, np.pi / 360, threshold=25, minLineLength=45, maxLineGap=35)
    if lines is None:
        return None
    yy, xx = np.nonzero(work)
    yy_f, xx_f = yy.astype(np.float64), xx.astype(np.float64)
    cands = []
    yref = .85 * h
    segs = sorted(lines[:, 0], key=lambda l: -np.hypot(l[2] - l[0], l[3] - l[1]))   # 긴 씨앗부터
    for x1, y1, x2, y2 in segs:
        if abs(y2 - y1) < 35:
            continue
        a = (float(x2) - float(x1)) / (float(y2) - float(y1))
        if not .12 <= abs(a) <= 4:
            continue
        b = x1 - a * y1
        if any(abs((a * yref + b) - c['x_ref']) < 25 and abs(a - c['a']) < .25 for c in cands):
            continue
        ok = True
        for _ in range(2):
            sel = np.abs(xx_f - (a * yy_f + b)) < 6
            if sel.sum() < 60:
                ok = False
                break
            ys = yy[sel]
            cnt = np.bincount(ys)
            sm = np.bincount(ys, weights=xx_f[sel])
            uy = np.nonzero(cnt)[0]
            a, b = np.polyfit(uy, sm[uy] / cnt[uy], 1)
        if not ok:
            continue
        sel = np.abs(xx_f - (a * yy_f + b)) < 6
        if sel.sum() < 60:
            continue
        uy = np.unique(yy[sel])
        if len(uy) < 40 or np.ptp(uy) < .10 * h or not .12 <= abs(a) <= 4:
            continue
        xr = float(a * yref + b)
        if not -.3 * w < xr < 1.3 * w:
            continue
        cands.append({'a': float(a), 'b': float(b), 'rows': len(uy), 'y_min': int(uy.min()), 'x_ref': xr})
    uniq = []
    for c in sorted(cands, key=lambda c: c['rows'], reverse=True):
        if not any(abs(c['x_ref'] - u['x_ref']) < 25 and abs(c['a'] - u['a']) < .25 for u in uniq):
            uniq.append(c)
    best = None
    for L in uniq:
        if L['a'] >= 0 or L['x_ref'] >= .5 * w:
            continue
        for R in uniq:
            if R['a'] <= 0 or R['x_ref'] <= .5 * w:
                continue
            vy = (R['b'] - L['b']) / (L['a'] - R['a'])
            vx = L['a'] * vy + L['b']
            width = R['x_ref'] - L['x_ref']
            if not (.1 * w <= vx <= .9 * w and .05 * h <= vy <= .80 * h and .15 * w <= width <= 1.5 * w):
                continue
            if vy > min(L['y_min'], R['y_min']) + 15:
                continue
            score = min(L['rows'], R['rows']) - .10 * width
            if best is None or score > best[0]:
                best = (score, vx, vy, R['a'] - L['a'])
    return None if best is None else (best[1], best[2], best[3])


# ============================================================
# Mask2Former — 내 차 영역 (사전 점검 전용: 올리고, 몇 프레임, 내린다)
# ============================================================
def load_mask2former(device):
    """Returns (processor, model, ego_ids). 오프라인 캐시만 쓴다 (네트워크 다운로드 없음)."""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    proc = AutoImageProcessor.from_pretrained(M2F_MODEL, local_files_only=True, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_MODEL, local_files_only=True)
    model = model.to(device).float().eval()
    ego_ids = [int(k) for k, v in model.config.id2label.items() if v in M2F_EGO_LABELS]
    return proc, model, ego_ids


@torch.inference_mode()
def m2f_ego_mask(m2f, bgr, device):
    """BGR 프레임 → (H, W) bool 내 차 영역 (Ego Vehicle ∪ Car Mount). 입력은 processor 설정 (384×384)."""
    proc, model, ego_ids = m2f
    H, W = bgr.shape[:2]
    inputs = proc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors='pt').to(device)
    out = model(**inputs)
    sem = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
    return np.isin(sem, ego_ids)


def vote_ego(masks, min_area_frac=0.001):
    """여러 프레임 내 차 마스크 → 과반 투표 → 화면 테두리에 닿는 덩어리만 (떠 있는 오분류 제거).
    내 차 (보닛·대시보드·A필러·룸미러·거치대) 는 화면 가장자리와 이어져 있다. Returns bool 마스크 또는 None."""
    if not masks:
        return None
    v = np.mean(np.stack(masks), 0) >= 0.5
    n, lbl, st, _ = cv2.connectedComponentsWithStats(v.astype(np.uint8), connectivity=8)
    H, W = v.shape
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        touch = x == 0 or y == 0 or x + w >= W or y + h >= H
        keep[k] = touch and a >= min_area_frac * H * W
    return keep[lbl]


def hood_row_from_ego(ego, x_lo=0.3, x_hi=0.7, frac=0.5):
    """내 차 마스크 → 보닛 행: 화면 가로 x_lo~x_hi 열에서 내 차 비율이 frac 이상인 행이 화면 맨 아래부터
    끊기지 않고 이어진 블록의 맨 윗행. 아래가 내 차가 아니면 None (보닛이 안 보임)."""
    H, W = ego.shape
    rowfrac = ego[:, int(x_lo * W):int(x_hi * W)].mean(1)
    if rowfrac[-1] < frac:
        return None
    r = H - 1
    while r > 0 and rowfrac[r - 1] >= frac:
        r -= 1
    return int(r)
