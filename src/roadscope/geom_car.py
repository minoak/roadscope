# SPDX-License-Identifier: AGPL-3.0-only
"""geom_car.py — 자동차 블랙박스 기하 (t20, DECISIONS 2026-10-09)

engine.py 의 --vehicle car 모드만 쓴다. 오토바이 모드는 이 모듈을 쓰지 않는다.
근거: 개발 기록의 자동차 블랙박스 기하 실측 (2026-10-09, docs/dev-log/ 참고. 원본 분석 자료는 리포에 없음)
  (stage1 foe.py·foe2.py·stage1.py·hood2.py, stage2 vehicle_h.py·veh_regress.py·ttc.py 에서 필요한 부분만 옮김)

평지·핀홀 모델 — 수평선 y_h, 소실점 열 x_vp, 카메라 높이 Hc, 초점거리 f:
    바닥 위치 거리  Z = f·Hc / (y_bottom − y_h)
    차폭 거리       Z = f·W / w                       (W: 차폭 prior, w: 박스 폭 px)
    옆 거리         X = Hc·(x_bottom_center − x_vp) / (y_bottom − y_h)   (f 불필요, 왼쪽 음수)
TTC — 박스 폭 증가율 1/τ = d ln w / dt, 바닥 위치 변화율 1/τ = d ln (y_bottom − y_h) / dt (f, Hc 불필요).
      기록만 하고 판정에는 쓰지 않는다 (DECISIONS 2026-10-09: 위험도는 그 순간의 장면으로 평가)
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import cv2

# ============================================================
# 상수 (DECISIONS 2026-10-09 와 연구 보고서 8절)
# ============================================================
REF_W = 1280                  # 픽셀 기준값들의 기준 폭 (Nexar 1280×720)
F_REF_PX = 600.0              # 초점거리 prior — 1280 폭 기준 600 px, 폭에 비례 (점선 자 실측 560~665)
CAM_H_DEFAULT_M = 1.4         # 카메라 높이를 못 구했을 때
CAM_H_RANGE_M = (0.8, 3.6)    # 이 밖이면 추정 실패로 본다 (승용 1.1~1.7, 트럭 캡 2.5~3.4 실측)
HORIZON_DEFAULT = 0.55        # 수평선을 못 구했을 때 (H 비율) — 실측 27영상 중앙값. 이때 바닥 거리는 쓰지 않는다
VEH_WIDTH_M = {'car': 1.86, 'bus': 2.5, 'truck': 2.5}   # 승용 중앙값 1.86, 법정 최대 2.5
OTHER_WIDTH_M = {'motorcycle': 0.8, 'bicycle': 0.6, 'person': 0.5}   # mvp.OBJECT_WIDTH_M 과 같음
WIDTH_GATE_M = (1.6, 2.5)     # 바닥 거리로 역산한 차폭이 이 밖이면 차폭 거리를 버린다 (경차 하한 ~ 법정 최대)
LANE_WIDTH_M = 3.5            # 차선 비율로 카메라 높이 보정할 때 차로폭 (한국 표준, mvp.LANE_WIDTH_M)
EGO_LANE_HALF_M = 2.5         # |X| ≤ 이 값이면 내 차로 (DECISIONS 2026-10-09)
LANE_LINE_M = LANE_WIDTH_M / 2    # 차로 중앙에 있을 때 차선까지 옆 거리
CUTIN_INNER_M = 1.25          # 차로 침범: 박스 바닥 안쪽 모서리 |X| 가 이보다 작고 (차선 1.75m 를 0.5m 이상 넘어옴)
                              # 바깥쪽이 차선 밖 (DECISIONS 2026-10-09, t26. t20~t24 는 1.5m)
VP_X_RANGE = (0.2, 0.8)       # 소실점 x 허용 범위 (화면 폭 비율, DECISIONS 2026-10-09). 실측 0.30~0.54W
EDGE_PX = 3                   # 화면 가장자리에서 이 안이면 잘린 박스

# 차 실내 마스크 (t27, solve_interior)
INT_CELL_PX = 32              # 정지·움직임 증거를 모으는 칸 크기 (1280 폭 기준)
INT_STATIC_PX = 0.6           # 0.1초 쌍에서 이보다 덜 움직인 점 = 정지 (1280 폭 기준, 연구 hood2 와 같음)
INT_MOVING_PX = 2.0           # 이보다 많이 움직인 점 = 움직임
INT_MIN_PAIRS = 15            # 내 차가 움직이는 쌍이 이보다 적으면 테두리 정지 영역·고정 박스를 만들지 않는다
INT_STATIC_MIN = 3            # 칸이 정지로 인정되는 정지점 수 (구간 전체 합)
INT_MOVING_MAX_FRAC = 0.1     # 정지 칸의 움직인 점 비율 상한 (구간 전체)
INT_PROTECT_HALF_W = 0.2      # 보호 영역: 소실점 열 ± 이 폭 (W 비율), 수평선 0.15H 위부터 아래 — 앞차 자리
INT_EXTRA_EVERY = 5           # 보정 구간 뒤에는 쌍 5개에 하나만 모은다 (약 0.5초 간격, 속도)
INT_DET_EVERY_SEC = 0.5       # 보정 구간 뒤에도 이 간격으로 YOLO 만 돌려 고정 박스를 구간 전체에서 본다
INT_TRACK_MIN_FRAMES = 10     # 고정 박스 판단에 필요한 내 차 움직임 YOLO 프레임 수
INT_TRACK_PRESENT = 0.5       # 반복 검출: 내 차 움직임 프레임 중 등장 비율 ('영상 대부분'. neg_01711 실내 person 은 전체 58%)
INT_TRACK_IOU = 0.5           # 고정 박스: 중앙 박스와 이 IoU 이상이면 '같은 자리' (어두운 실내 박스는 위아래로 흔들림)
INT_TRACK_STATIC = 0.7        # 고정 박스: 같은 자리 비율
INT_BOX_MOVING_MAX = 0.5      # 고정 박스 자리 칸 중 움직임 칸 비율이 이 이상이면 실내가 아님 (같은 속도로 옆을 달린 차:
                              # 차가 없을 때 배경이 움직여 움직임 칸이 된다). 미만이면 움직임 칸을 뺀 부분만 넣는다.
                              # neg_01711 실내 물체 박스는 가장자리에 바깥 풍경이 섞여 움직임 칸이 20% 를 넘는다
INT_BOX_MAX_W = 0.25          # 고정 박스 폭 상한 (W 비율). A필러·룸미러·워터마크는 좁다. 화면 절반을 덮는 환각 박스
                              # (pos_00413) 가 앞차 자리까지 실내로 칠하던 문제. 대시보드 박스는 보닛 띠 규칙이 맡는다
INT_BOX_FRAC = 0.5            # 박스 면적의 이 비율 이상이 실내면 판정에서 뺀다 (YOLO 박스)
INT_BOX_FRAC_AUX = 0.3        # 되살린 박스 (저확신·마스크) 는 더 엄격히 (moto ego 가드 RESCUE_EGO_BOX 와 같은 값)

# 차선 영점 (t40, DECISIONS 2026-10-09) — 구간마다 차선이 실제 보이는 가장 아래 행 y_ref 를 정하고, 그 행에서
# 내 차로 폭과 차로 안 평소 위치 b (t32 식) 를 잰다. LDW 는 매 프레임 y_ref 에서만 잰다.
# (t32 까지의 0.95H 고정 행은 자동차 보닛·대시보드에 가려진 곳이라 차선을 외삽했다. 보닛 띠 중앙값 0.23H)
LANE_ZERO_VIS_FRAC = 0.5      # y_ref 후보 행: 좌우 실제 선이 다 잡힌 프레임 중 이 비율 이상에서 두 선이 함께 보인다.
                              # YOLOPv2 는 점선 틈도 이어 그려 보닛 위까지 보임 비율이 0.9~1.0 (t40 개발 세트 2fps 실측)
                              # 이라 절반이면 옆 차에 가리는 영상 (pos_00004 오른쪽 선) 만 위로 올라간다
LANE_ZERO_MARGIN = 0.02       # 보이는 가장 아래 행에서 위로 띄우는 여유 (H 비율, 720p 14px): 보닛 행 탐색 간격 4px·
                              # motion_edge 여유 4px 의 두 배 + YOLOPv2 출력 2배 확대 번짐 2px
LANE_ZERO_MIN_DY = 0.08       # y_ref 는 수평선 아래 이 이상 (H 비율). 더 위는 1px 이 수 cm 를 넘어 측정이 거칠다
LANE_ZERO_TOL = 0.02          # 선 위치 ± 이 폭 (W 비율) 안의 차선 픽셀을 그 선의 것으로 본다
LANE_ZERO_BAND = 2            # y_ref ± 이 행 (720p 기준) 의 픽셀로 그 행의 선 위치를 잰다 (YOLOPv2 출력 2배 확대 몫)
LANE_ZERO_W_RANGE_M = (2.75, 3.75)   # y_ref 차로 폭 검산 범위. 밖이면 영점 신뢰도 low
EGO_CAR_W_M = 1.86            # 내 차 폭 (VEH_WIDTH_M car). y_ref 에서 px 로 바꿔 차선까지 간격을 기록한다
LANE_BASE_MAX_M = 0.5         # 평소 위치 상한 (±m)
LANE_BASE_MIN_FRAMES = 30     # 최소 근거 프레임: max(30, ceil(본 프레임 × 10%))
LANE_BASE_MIN_FRAC = 0.10
LANE_ZERO_RELAX_MIN_FRAMES = 15   # [t41 VA] 영점 최소 근거 완화값 = V0 의 절반: max(15, ceil(본 프레임 × 5%)).
LANE_ZERO_RELAX_MIN_FRAC = 0.05   # 쌍 프레임·근거 프레임 두 관문에 같이 쓴다 (y_ref·폭·b 규칙은 그대로)

# 차선 신뢰 (t44, DECISIONS 2026-10-10 민옥 결정: 차로폭 정상 범위 3.0 ± 1.0 m).
# 영점 y_ref 의 차로 폭이 이 범위 밖이거나 영점이 실패하면 그 구간 차선을 믿지 않는다 → LDW 측정 불가,
# 차선을 수평선·소실점·카메라 높이 계산에서도 뺀다 (FOE·차량 박스 회귀·차량 높이만).
# 순환을 끊으려고 이 판정의 차로 폭은 차선 없이 구한 수평선·카메라 높이로 잰다 (lane_trust)
LANE_TRUST_W_RANGE_M = (2.0, 4.0)

# 보정 단계
CAL_SEC = 8.0               # 구간 앞부분 이 시간 동안 표본을 모은다
CAL_MAX_SEC = 20.0            # 광류 소실점이 부족하면 여기까지 늘린다 (정지 출발 등)
FLOW_HZ = 10.0                # 광류 프레임 쌍 간격 (연구: 30fps 에서 3프레임)
FOE_MIN_PAIRS = 5             # 소실점 확정에 필요한 통과 쌍 수 (연구 stage1 클립 기준과 같음)
FOE_TARGET_PROV = 15          # 보정 구간을 CAL_SEC 에서 끝내려면 필요한 잠정 통과 쌍 수 (모자라면 CAL_MAX_SEC 까지)
FOE_MAX_IQR_PX = 60.0         # 통과 쌍 FOE 행·열의 IQR 상한 (1280 폭 기준). 실측 클립 IQR 중앙값 7/15px, p90 21/63px.
                              # 넘으면 불안정 (pos_00905 트럭 야적장: 연구에서도 unstable 로 제외)
YAW_MAX_DPS = 5.0             # 회전 중 프레임 제외 (°/s)
AGREE_PX = 25.0               # 수평선 교차 확인 일치 기준 (1280 폭 기준 px). 실측 FOE↔차선 16/17 이 25px 안
LANE_MIN_FRAMES = 5           # 차선 소실점·높이에 필요한 프레임 수
VEH_MIN_OBS = 3               # 차량 높이: track 하나에 필요한 관측 수 (평가 fps 기준)
VEH_STRONG_TRACKS = 3         # 차량 높이를 차선보다 믿는 track 수 (실측: 3개 이상이면 영상 안 IQR/중앙값 10%)
CAM_H_AGREE = 1.3             # 차량·차선 높이 비가 이 안이면 평균 (실측 비 0.75~1.25)
REG_MIN_BOXES = 20            # 박스 회귀 수평선: 박스 수, track 수, 폭 범위
REG_MIN_TRACKS = 2
REG_MIN_WRATIO = 2.0

# TTC (기록만 — 판정에 안 씀. t20 의 '두 방법 모두 3초/2초 미만 → 위험/치명' 은 t22 에서 뺐다)
TTC_WIN_SEC = 0.5
TTC_MIN_N = 5                 # 창 안 관측 수 (10fps 에서 0.5초 창은 6개)
TTC_MIN_SPAN = 0.35           # 창 안 관측 시간 폭 (초)
TTC_SE_MAX = 0.15             # 1/τ 표준오차 상한 (/s) — 넘으면 τ 를 기록하지 않는다 (연구 ttc.py 와 같은 기준)
TTC_MIN_CONF = 0.4
TTC_MIN_W_PX = 15             # 1280 폭 기준

# 박스 없는 근접 물체 보험 (Depth Pro)
DG_RANGE_M = 5.0              # 이 거리 안 물체 → 위험 (FCW L2 기준과 같음)
DG_HALF_W_M = 1.0             # 검사 띠 반폭: 거리 DG_RANGE_M 에서 |X| ≤ 1m 에 해당하는 열
DG_RATIO = 0.7                # 같은 행 지면 거리의 이 비율보다 가까우면 지면이 아닌 물체
DG_MIN_FRAC = 0.25            # 검사 띠에서 박스 밖 물체 픽셀 비율이 이 이상이면 경보
DG_MIN_ROAD_PX = 1000         # 축척 맞춤에 필요한 도로 픽셀 수 (1280 폭 기준)
DG_FLOOR = 0.05               # Depth Pro 원값 하한 (자동차 모드 — 0.5m 하한에 붙는 문제, t19)
DG_SCALE_KEEP = 10            # 축척 이력 (도로가 안 보이는 프레임에서 중앙값 사용)
DG_CONFS = ('high', 'medium')  # 이 수평선 신뢰도 구간에서만 보험을 켠다. low 구간은 보닛·지면 기하를 믿기 어렵다
                              # (pos_00905: 트럭 흰 보닛을 광류가 못 잡아 내 보닛을 물체로 봄, 180프레임 중 168 경보)


def _scale(W):
    return W / REF_W


# ============================================================
# 광류 소실점 (FOE) — 회전 보정 포함. 연구 stage1 foe.py / foe2.py 그대로
# ============================================================
def _track(g0, g1, mask):
    """LK 추적 (forward-backward 0.7px 검사). Returns (점, 이동량) 또는 (None, None)."""
    pts = cv2.goodFeaturesToTrack(g0, 1500, 0.005, 8, mask=mask)
    if pts is None:
        return None, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=3)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=3)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0r - pts, axis=2)[:, 0] < 0.7)
    return pts[ok, 0], p1[ok, 0] - pts[ok, 0]


def _ls_foe(p, d, w=None):
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    if w is None:
        w = np.ones(len(p))
    A = n * np.sqrt(w)[:, None]
    b = (n * p).sum(1) * np.sqrt(w)
    q, *_ = np.linalg.lstsq(A, b, rcond=None)
    return q


def _ang_err(q, p, d):
    r = p - q
    c = (r * d).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(d, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _ransac_foe(p, d, W, H, iters=300, thr=4.0, return_inliers=False):
    """회전 보정 전 FOE (초기값용). 시드 고정 — 같은 영상이면 같은 결과.
    return_inliers=True 면 (FOE, inlier mask) — 실내 마스크의 '내 차 움직임' 판정용."""
    rng = np.random.default_rng(0)
    best, bn, N = None, 0, len(p)
    if N < 20:
        return (None, None) if return_inliers else None
    for _ in range(iters):
        i, j = rng.choice(N, 2, replace=False)
        q = _ls_foe(p[[i, j]], d[[i, j]])
        if not np.all(np.isfinite(q)) or abs(q[0]) > 5 * W or abs(q[1]) > 5 * H:
            continue
        inl = _ang_err(q, p, d) < thr
        if inl.sum() > bn:
            bn, best = inl.sum(), inl
    if best is None:
        return (None, None) if return_inliers else None
    for _ in range(3):
        q = _ls_foe(p[best], d[best], w=np.linalg.norm(d[best], axis=1))
        best = _ang_err(q, p, d) < thr
    return (q, best) if return_inliers else q


def _rotflow(P, w, f, W, H):
    x = P[:, 0] - W / 2
    y = P[:, 1] - H / 2
    wx, wy, wz = w
    u = x * y / f * wx - (f + x * x / f) * wy + y * wz
    v = (f + y * y / f) * wx - x * y / f * wy - x * wz
    return np.stack([u, v], 1)


def _resid(th, P, Dd, f, W, H):
    q, w = th[:2], th[2:]
    t = Dd - _rotflow(P, w, f, W, H)
    r = P - q
    rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-6)
    return t[:, 0] * rn[:, 1] - t[:, 1] * rn[:, 0]


def _fit_rot(P, Dd, q0, f, W, H):
    """FOE (x, y) 와 회전 3성분을 같이 푼다 (robust least squares). Returns (θ, inlier 수, inlier mask)."""
    from scipy.optimize import least_squares
    best = None
    for qi in [q0, (W / 2, 0.45 * H), (W / 2, 0.6 * H), (W / 2, 0.35 * H)]:
        if qi is None:
            continue
        th0 = np.array([qi[0], qi[1], 0, 0, 0], float)
        xs = [100, 100, 1e-3, 1e-3, 1e-3]
        s = least_squares(_resid, th0, args=(P, Dd, f, W, H), loss='cauchy', f_scale=0.5, x_scale=xs)
        inl = np.abs(_resid(s.x, P, Dd, f, W, H)) < 0.7
        for _ in range(2):
            if inl.sum() < 10:
                break
            s = least_squares(_resid, s.x, args=(P[inl], Dd[inl], f, W, H), loss='soft_l1',
                              f_scale=0.5, x_scale=xs)
            inl = np.abs(_resid(s.x, P, Dd, f, W, H)) < 0.7
        tr = Dd - _rotflow(P, s.x[2:], f, W, H)
        inl &= (tr * (P - s.x[:2])).sum(1) > 0      # 바깥으로 퍼지는 흐름만
        sc = int(inl.sum())
        if best is None or sc > best[1]:
            best = (s.x, sc, inl)
    return best


def hood_split(Y, M, H, s, y0):
    """추적점 (행 Y, 이동량 M) → 움직이는 점 위 / 정지점 아래로 가장 잘 나뉘는 행 (연구 hood2.py).
    Returns (행 또는 None, 상태 'flow' | 'motion_edge' | 'none_visible' | 'few_moving').
    정지점이 띠를 이루지 않으면 움직이는 점의 아래 끝을 경계로 (t20, 야간 검은 대시보드)."""
    keep = Y >= y0
    Y, M = Y[keep], M[keep]
    st, mv = M < 0.6 * s, M > 2.0 * s
    nst, nmv = int(st.sum()), int(mv.sum())

    def no_static_band():
        if nmv >= 20:
            y_low = float(np.percentile(Y[mv], 99))
            if y_low < 0.95 * H:
                return int(min(H, y_low + 4 * s)), 'motion_edge'
        return H, 'none_visible'          # 움직이는 점이 맨 아래까지 = 보닛이 안 보인다

    if nst < 20:
        return no_static_band()
    if nmv < 20:
        return None, 'few_moving'
    ys = np.arange(y0 + 16 * H / 720, H + 1, 4 * H / 720)
    sc = np.array([(mv & (Y < y)).sum() / nmv + (st & (Y >= y)).sum() / nst for y in ys])
    y = int(ys[np.nonzero(sc >= sc.max() - 0.015)[0].max()])
    if (st & (Y >= y)).sum() < 15:
        return no_static_band()
    return y, 'flow'


class FlowCollector:
    """보정 구간의 광류 표본. 프레임 쌍 (t, t + step) 마다 add_pair (연구 stage1.py·hood2.py).

    쌍마다 회전 보정 FOE 를 풀고, 보닛 띠 판정용 하단 중앙 추적점 (행, 이동량) 을 모은다.
    정지·회전·fit 불량 판정은 solve() 에서 한다 (도로 움직임 판정에 잠정 FOE 행과 보닛이 필요해서).
    """

    def __init__(self, W, H, fps):
        self.W, self.H, self.fps = W, H, fps
        self.s = _scale(W)
        self.step = max(1, int(round(fps / FLOW_HZ)))
        self.f_nom = 700.0 * self.s          # 회전 모델 명목 초점 (연구와 같음)
        self.pairs = []
        self.hood_pts = []
        self.n_prov_ok = 0                   # 잠정 통과 쌍 수 (구간 연장 판단용)
        self.mask_all = np.full((H, W), 255, np.uint8)
        self.mask_all[:int(round(40 * H / 720))] = 0      # 상단 타임스탬프 띠
        self.mask_low = np.zeros((H, W), np.uint8)
        self.mask_low[int(0.45 * H):, int(0.25 * W):int(0.75 * W)] = 255
        self.n_hood_tries = 0

    def add_pair(self, g0, g1, t_sec):
        s, W, H = self.s, self.W, self.H
        rec = {'t': round(t_sec, 3)}
        self.pairs.append(rec)
        p, d = _track(g0, g1, self.mask_all)
        if p is None or len(p) < 30:
            rec['why'] = 'notrack'
        else:
            m = np.linalg.norm(d, axis=1)
            rec['px'], rec['py'], rec['mag'] = (p[:, 0].astype(np.float32),
                                                p[:, 1].astype(np.float32), m.astype(np.float32))
            sel = m > 0.8 * s
            if sel.sum() < 30:
                rec['why'] = 'few_moving_feats'
            else:
                P, Dd = p[sel].astype(np.float64), d[sel].astype(np.float64)
                if len(P) > 800:                       # 속도 — 고정 시드로 800점
                    idx = np.random.default_rng(1).choice(len(P), 800, replace=False)
                    P, Dd = P[idx], Dd[idx]
                q = _ransac_foe(P, Dd, W, H)
                th, sc, inl = _fit_rot(P, Dd, q, self.f_nom, W, H)
                rec.update(x=float(th[0]), y=float(th[1]), inl=float(inl.mean()), ninl=sc,
                           yaw=float(np.degrees(th[3]) * self.fps / self.step))
                if (rec['inl'] >= 0.4 and sc >= 25 and abs(rec['yaw']) <= YAW_MAX_DPS
                        and np.percentile(m, 75) >= 1.5 * s
                        and -0.5 * H < rec['y'] < 1.5 * H and -0.5 * W < rec['x'] < 1.5 * W):
                    self.n_prov_ok += 1
        # 보닛 띠: 하단 중앙만 (연구 hood2.py 는 쌍 두 개에 한 번 — 정지 출발 영상에서 표본이 모자라 매 쌍으로)
        self.n_hood_tries += 1
        p2, d2 = _track(g0, g1, self.mask_low)
        if p2 is not None and len(p2) >= 10:
            m2 = np.linalg.norm(d2, axis=1)
            if np.percentile(m2, 75) >= 2 * s:
                self.hood_pts.append((p2[:, 1].astype(np.float32), m2.astype(np.float32)))

    def _hood(self, foe_y):
        """움직이는 동안 정지해 있는 하단 띠의 시작 행. Returns (행 또는 None, 상태).

        정지점이 없을 때 (야간에 대시보드가 검어 추적점이 안 잡히는 경우 — pos_00936):
        움직이는 점이 화면 맨 아래 5% 까지 내려오면 보닛이 안 보이는 것 (H), 그보다 위에서 끝나면
        움직이는 점의 아래 끝 (99 퍼센타일) 을 보닛 경계로 본다 ('motion_edge').
        """
        if not self.hood_pts:
            return None, 'no_motion'
        y0 = int(max(0.45 * self.H, min(foe_y + 0.1 * self.H, 0.8 * self.H)))
        Y = np.concatenate([a for a, _ in self.hood_pts])
        M = np.concatenate([b for _, b in self.hood_pts])
        return hood_split(Y, M, self.H, self.s, y0)

    def solve(self):
        """Returns dict: FOE (x, y, n_ok, iqr), 보닛 행·상태, 제외 사유 집계."""
        W, H, s = self.W, self.H, self.s
        fits = [r for r in self.pairs if 'y' in r]
        fy = [r['y'] for r in fits if r['inl'] > 0.4 and -0.5 * H < r['y'] < 1.5 * H]
        foe_y0 = float(np.median(fy)) if fy else 0.45 * H
        hood_row, hood_status = self._hood(foe_y0)
        lim = min(0.9 * H, (hood_row if hood_row else H) - 10 * s)
        why_n = {}
        ok = []
        for r in self.pairs:
            if 'y' not in r:
                why_n[r.get('why', '?')] = why_n.get(r.get('why', '?'), 0) + 1
                continue
            px, py, mag = r['px'], r['py'], r['mag']
            road = (py > foe_y0 + 0.08 * H) & (py < lim) & (px > 0.15 * W) & (px < 0.85 * W)
            road_mag = float(np.median(mag[road])) if road.sum() >= 10 else 0.0
            why = []
            if road_mag < 1.5 * s: why.append('stopped')
            if r['inl'] < 0.4 or r['ninl'] < 25: why.append('poorfit')
            if abs(r['yaw']) > YAW_MAX_DPS: why.append('turning')
            if not (-0.5 * H < r['y'] < 1.5 * H and -0.5 * W < r['x'] < 1.5 * W): why.append('outside')
            if why:
                for w_ in why:
                    why_n[w_] = why_n.get(w_, 0) + 1
            else:
                ok.append(r)
        out = {'n_pairs': len(self.pairs), 'n_ok': len(ok), 'excluded': why_n,
               'hood_row': hood_row, 'hood_status': hood_status}
        if ok:
            xs = np.array([r['x'] for r in ok])
            ys = np.array([r['y'] for r in ok])
            out.update(x=float(np.median(xs)), y=float(np.median(ys)),
                       iqr_x=float(np.subtract(*np.percentile(xs, [75, 25]))),
                       iqr_y=float(np.subtract(*np.percentile(ys, [75, 25]))))
        return out


# ============================================================
# 차 실내 마스크 (t27, DECISIONS 2026-10-09) — 대시보드·보닛·A필러·룸미러·반사·워터마크
# ============================================================
class InteriorCollector:
    """구간 전체에서 '내 차가 움직이는 프레임 쌍' 의 LK 추적점을 칸 (INT_CELL_PX) 별로 모은다.

    내 차 움직임 판정 (쌍마다): 움직이는 점 (> 0.8px) 40개 이상, 화면 좌·우 1/3 양쪽에 6개 이상,
    회전 보정 없는 RANSAC FOE 의 방사형 inlier 50% 이상. 정지 중 옆 차만 지나가는 쌍을 빼기 위한 조건.
    칸마다 정지점 (< INT_STATIC_PX) 과 움직이는 점 (> INT_MOVING_PX) 수를 센다. 보닛 띠 대체 추정용으로
    화면 가로 10~90% 점의 (행, 이동량) 도 모은다.
    """

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.s = _scale(W)
        self.cell = max(8, int(round(INT_CELL_PX * self.s)))
        self.gw, self.gh = -(-W // self.cell), -(-H // self.cell)
        self.static = np.zeros((self.gh, self.gw), np.int32)
        self.moving = np.zeros((self.gh, self.gw), np.int32)
        self.rows_pts = []
        self.pairs = []          # [(t, ego_moving)]
        self.n_moving = 0

    def add_pair(self, g0, g1, t_sec):
        W, H, s = self.W, self.H, self.s
        p, d = _track(g0, g1, None)
        ego = False
        if p is not None and len(p) >= 60:
            m = np.linalg.norm(d, axis=1)
            sel = m > 0.8 * s
            xs = p[sel, 0]
            if sel.sum() >= 40 and (xs < W / 3).sum() >= 6 and (xs > 2 * W / 3).sum() >= 6:
                P, D = p[sel].astype(np.float64), d[sel].astype(np.float64)
                if len(P) > 600:
                    idx = np.random.default_rng(2).choice(len(P), 600, replace=False)
                    P, D = P[idx], D[idx]
                _, inl = _ransac_foe(P, D, W, H, iters=100, return_inliers=True)   # 50% 판정이라 100회로 충분
                ego = inl is not None and inl.mean() >= 0.5
        self.pairs.append((round(t_sec, 3), ego))
        if not ego:
            return
        self.n_moving += 1
        cx = np.clip((p[:, 0] // self.cell).astype(int), 0, self.gw - 1)
        cy = np.clip((p[:, 1] // self.cell).astype(int), 0, self.gh - 1)
        st, mv = m < INT_STATIC_PX * s, m > INT_MOVING_PX * s
        np.add.at(self.static, (cy[st], cx[st]), 1)
        np.add.at(self.moving, (cy[mv], cx[mv]), 1)
        keep = (p[:, 0] > 0.1 * W) & (p[:, 0] < 0.9 * W)
        self.rows_pts.append((p[keep, 1].astype(np.float32), m[keep].astype(np.float32)))


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def solve_interior(ic: InteriorCollector, cal, win_dets, tracker_cls, m2f_ego=None, m2f_info=None):
    """구간의 내 차 (실내) 마스크를 만든다 — 사전 점검 (t27·t29). cal 의 보닛을 (모를 때만) 보강하고
    cal.interior_mask 등을 붙인다.

    마스크 = (1) 보닛 띠 (cal.hood_row, 보닛을 알 때만)
           ∪ (2) Mask2Former 내 차 영역 (Ego Vehicle + Car Mount, 보정 구간 몇 프레임 과반 투표, 화면 테두리에
                 닿는 덩어리만 — car_models.vote_ego). 움직임과 무관해 정지 영상에서도 쓴다
           ∪ (3) 테두리 정지 영역: 구간 내내 정지한 칸 (정지점 INT_STATIC_MIN 이상, 움직이는 점 비율
                 INT_MOVING_MAX_FRAC 이하) 중 화면 왼쪽·오른쪽·위 테두리와 정지 칸으로 이어진 덩어리 (워터마크 등).
                 보호 영역 (수평선 0.15H 위부터 아래, 소실점 열 ± INT_PROTECT_HALF_W·W) 은 넣지 않는다 — 앞차 자리
           ∪ (4) 반복 검출 박스: 구간 전체를 INT_DET_EVERY_SEC 간격으로 본 YOLO 결과 중 내 차가 움직이는 프레임의
                 INT_TRACK_PRESENT 이상에서 같은 자리 (중앙 박스와 IoU ≥ INT_TRACK_IOU) 같은 크기 (면적 ±30%) 로
                 INT_TRACK_STATIC 이상 잡힌 박스. 왼쪽·오른쪽·위 테두리에 닿고 폭 INT_BOX_MAX_W 이하, 중심이 보호
                 영역 밖이어야 한다 (A필러·룸미러 위 환각 박스 — neg_01711 왼쪽 끝 person). 그 자리 칸 중 구간 전체
                 움직임 칸이 INT_BOX_MOVING_MAX 이상이면 버리고 (같은 속도로 옆을 달린 차), 아니면 움직임 칸과 보호
                 영역을 뺀 부분만 넣는다
    앞차와 구분: 앞차는 보호 영역 안에 있고 테두리에 닿지 않으며, 거리가 변하면 크기·위치가 바뀌고 차로 변경·
    정지·출발로 구간 대부분 동안 같은 자리에 있지 않다. 내 차가 움직이는 프레임만 세므로 정지 중 같이 서 있는
    앞차도 세지 않는다. 박스는 면적의 INT_BOX_FRAC 이상이 마스크 안일 때만 뺀다.
    내 차 움직임 쌍이 INT_MIN_PAIRS 미만이면 (3)(4) 를 만들지 않는다 (실패 → 박스를 지우지 않는 쪽).
    win_dets: 구간 YOLO 결과 [(t, dets)] — 보정 구간은 평가 프레임마다, 뒤는 INT_DET_EVERY_SEC 간격.
    """
    W, H, s, c = ic.W, ic.H, ic.s, ic.cell
    info = {'n_pairs': len(ic.pairs), 'n_moving_pairs': ic.n_moving, 'm2f': m2f_info or {'status': 'none'}}
    # 보닛 띠 보강: 보정 구간 광류로 못 잡았으면 (기본값) 구간 전체의 내 차 움직임 점으로 다시 나눈다
    if not cal.hood_known and ic.rows_pts and ic.n_moving >= INT_MIN_PAIRS:
        Y = np.concatenate([a for a, _ in ic.rows_pts])
        M = np.concatenate([b for _, b in ic.rows_pts])
        y0 = int(max(0.45 * H, min(cal.horizon_y + 0.1 * H, 0.8 * H)))
        row, st_ = hood_split(Y, M, H, s, y0)
        info['hood_fallback'] = {'row': row, 'status': st_}
        if row is not None:
            cal.hood_row, cal.hood_src = int(row), 'interior_' + st_
    mask = np.zeros((H, W), bool)
    if cal.hood_known and cal.hood_row < H:
        mask[int(cal.hood_row):] = True
    info['hood_frac'] = round(float(mask.mean()), 4)
    if m2f_ego is not None and m2f_ego.shape == (H, W):
        mask |= m2f_ego
        info['m2f']['frac'] = round(float(m2f_ego.mean()), 4)
    if ic.n_moving < INT_MIN_PAIRS:
        info['status'] = 'no_motion'
        _attach_interior(cal, mask, info)
        return info
    tot = ic.static + ic.moving
    st_cell = (ic.static >= INT_STATIC_MIN) & (ic.moving <= INT_MOVING_MAX_FRAC * tot)
    mv_cell = (ic.moving >= 2) & (ic.moving >= 0.3 * tot)
    ycs = (np.arange(ic.gh) + 0.5) * c
    xcs = (np.arange(ic.gw) + 0.5) * c
    protect = ((ycs[:, None] >= cal.horizon_y - 0.15 * H)
               & (np.abs(xcs[None, :] - cal.vp_x) <= INT_PROTECT_HALF_W * W))
    cand = st_cell & ~protect
    from scipy import ndimage
    lbl, n = ndimage.label(cand)
    border = set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1])) | set(np.unique(lbl[0, :]))
    border.discard(0)
    edge_cells = np.isin(lbl, list(border))
    edge_px = np.kron(edge_cells, np.ones((c, c), bool))[:H, :W]
    mask |= edge_px
    info['edge_static_frac'] = round(float(edge_px.mean()), 4)
    # (4) 반복 검출 박스 — 보정 구간의 평가 프레임은 INT_DET_EVERY_SEC 간격으로 솎아 구간 전체와 같은 밀도로
    mv_ts = np.array([t for t, e in ic.pairs if e])
    thin, last = [], None
    for t, d in win_dets:
        if last is None or t - last >= INT_DET_EVERY_SEC - 1e-6:
            thin.append((t, d))
            last = t
    info['n_scan_frames'] = len(thin)
    frames = [(t, d) for t, d in thin if mv_ts.size and np.abs(mv_ts - t).min() <= 0.2]
    mv_px = np.kron(mv_cell, np.ones((c, c), bool))[:H, :W]
    protect_px = np.kron(protect, np.ones((c, c), bool))[:H, :W]
    boxes = []
    if len(frames) >= INT_TRACK_MIN_FRAMES:
        tr = tracker_cls(iou_threshold=0.3, max_ghost_frames=3)
        hist = {}
        for k, (t, dets) in enumerate(frames):
            tracks = tr.update(k, [(tuple(o['bbox']), 0) for o in dets])
            ids = {tuple(x.bbox): x.track_id for x in tracks if x.ghost_count == 0}
            for o in dets:
                tid = ids.get(tuple(o['bbox']))
                if tid is not None:
                    hist.setdefault(tid, []).append((tuple(o['bbox']), o['cls']))
        tol = 3 * EDGE_PX
        for tid, h in hist.items():
            if len(h) < INT_TRACK_PRESENT * len(frames):
                continue
            arr = np.array([b for b, _ in h], float)
            med = tuple(int(v) for v in np.median(arr, 0))
            area_med = max((med[2] - med[0]) * (med[3] - med[1]), 1)
            static_frac = float(np.mean([_box_iou(b, med) >= INT_TRACK_IOU
                                         and 0.7 <= (b[2] - b[0]) * (b[3] - b[1]) / area_med <= 1 / 0.7
                                         for b, _ in h]))
            if static_frac < INT_TRACK_STATIC:
                continue
            x0, y0b, x1, y1 = med
            touch = x0 <= tol or x1 >= W - 1 - tol or y0b <= tol
            cxm, cym = (x0 + x1) / 2, (y0b + y1) / 2
            in_protect = cym >= cal.horizon_y - 0.15 * H and abs(cxm - cal.vp_x) <= INT_PROTECT_HALF_W * W
            if not touch or x1 - x0 > INT_BOX_MAX_W * W or in_protect:
                continue
            gx0, gx1 = x0 // c, min(ic.gw - 1, x1 // c)
            gy0, gy1 = y0b // c, min(ic.gh - 1, y1 // c)
            mv_in = float(mv_cell[gy0:gy1 + 1, gx0:gx1 + 1].mean())
            rec = {'bbox': list(med), 'cls': h[0][1], 'n': len(h), 'static_frac': round(static_frac, 2),
                   'moving_cells': round(mv_in, 2)}
            if mv_in >= INT_BOX_MOVING_MAX:
                rec['rejected'] = 'moving_cells'
            else:                       # 박스 자리 중 움직임 칸·보호 영역이 아닌 부분만
                mask[y0b:y1 + 1, x0:x1 + 1] |= ~(mv_px[y0b:y1 + 1, x0:x1 + 1]
                                                 | protect_px[y0b:y1 + 1, x0:x1 + 1])
            boxes.append(rec)
    info['static_boxes'] = boxes
    info['n_scan_frames_moving'] = len(frames)
    info['status'] = 'ok'
    _attach_interior(cal, mask, info)
    return info


def _attach_interior(cal, mask, info):
    info['frac'] = round(float(mask.mean()), 4)
    cal.interior_mask = mask
    cal.interior_sat = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    cal.interior_info = info


def interior_frac(bbox, cal):
    """박스 면적 중 실내 마스크 안 비율 (마스크가 없으면 0)."""
    sat = getattr(cal, 'interior_sat', None)
    if sat is None:
        return 0.0
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x1, y1 = min(x1, cal.W - 1), min(y1, cal.H - 1)
    if x1 < x0 or y1 < y0:
        return 0.0
    tot = sat[y1 + 1, x1 + 1] - sat[y0, x1 + 1] - sat[y1 + 1, x0] + sat[y0, x0]
    return float(tot) / ((x1 - x0 + 1) * (y1 - y0 + 1))


# ============================================================
# 차선 영점 (t40) — 구간마다 y_ref (차선이 실제 보이는 가장 아래 행), 그 행의 차로 폭, 차로 안 평소 위치 b.
# LDW 만 쓴다 (옆 거리 X·BSW·FCW·차선 고르기는 그대로). 결과는 cal.lane_baseline (t32 키 이름 유지)
# ============================================================
def pick_pair_at(lines, y, x_ref):
    """차선들 (x = dydx·y + c) 중 행 y 에서 x_ref 의 왼쪽·오른쪽으로 가장 가까운 선. Returns (left, right), 없으면 None."""
    left = right = None
    for l in lines:
        x = l['dydx'] * y + l['c']
        if x < x_ref:
            if left is None or x > left[0]:
                left = (x, l)
        elif right is None or x < right[0]:
            right = (x, l)
    return (left[1] if left else None), (right[1] if right else None)


def _row_cumsum(lane):
    """차선 픽셀 bool (H, W) → 행별 누적합 (H, W+1). 구간 [a, b) 의 픽셀 수 = cs[y, b] − cs[y, a]."""
    cs = np.zeros((lane.shape[0], lane.shape[1] + 1), np.int32)
    np.cumsum(lane, axis=1, out=cs[:, 1:])
    return cs


def line_rows_visible(cs, line, y0, tol):
    """행 y0 부터 화면 아래까지, 선 x(y) ± tol 안에 차선 픽셀이 있는 행 (bool 배열). 화면 밖으로 나간 행은 False."""
    H, W1 = cs.shape
    ys = np.arange(y0, H)
    xs = line['dydx'] * ys + line['c']
    a = np.clip(np.floor(xs - tol), 0, W1 - 1).astype(int)
    b = np.clip(np.ceil(xs + tol) + 1, 0, W1 - 1).astype(int)
    return (cs[ys, b] - cs[ys, a]) > 0


def line_x_measured(lane, line, y, band, tol):
    """행 y ± band 에서 선 x(y) ± tol 안 차선 픽셀의 평균 x = 그 행에서 실제로 보인 선 위치. 픽셀이 2개 미만이면 None
    (연장하지 않는다)."""
    H, W = lane.shape
    x = line['dydx'] * y + line['c']
    a, b = int(max(0, np.floor(x - tol))), int(min(W, np.ceil(x + tol) + 1))
    y0, y1 = max(0, y - band), min(H, y + band + 1)
    if b <= a or y1 <= y0:
        return None
    xs = np.nonzero(lane[y0:y1, a:b])[1]
    return float(xs.mean() + a) if len(xs) >= 2 else None


def lane_zero_params(W, H):
    """(선 허용 폭 px, 측정 행 ± px, 여유 px)"""
    return LANE_ZERO_TOL * W, max(1, int(round(LANE_ZERO_BAND * H / 720))), int(round(LANE_ZERO_MARGIN * H))


def lane_zero(cal, frames, lane_of, n_examined, relaxed=False):
    """구간의 차선 영점. frames[j] = j 번째 평가 프레임의 검증된 차선 (engine.find_lanes 의 lines, 실패면 []),
    lane_of(j) = 그 프레임의 차선 픽셀 bool (H, W) — 보닛 띠·내 차 영역의 픽셀을 지운 것.
    relaxed (t41 VA) 면 1)·3) 의 최소 프레임을 max(LANE_ZERO_RELAX_MIN_FRAMES, ceil(본 프레임 × LANE_ZERO_RELAX_MIN_FRAC)) 로
    낮춘다. 다른 규칙은 같아서 원래 기준을 통과한 구간은 결과가 같다 (required_frames 만 다르다).

    1) 보닛 행 바로 위 (잠정 행) 에서 소실점 열 좌우로 가장 가까운 실제 선 두 개를 고른 프레임 (쌍 프레임) 마다,
       행별로 두 선이 함께 보이는지 센다. 쌍 프레임의 LANE_ZERO_VIS_FRAC 이상에서 보이는 가장 아래 행 − 여유 = y_ref
    2) 프레임마다 y_ref 에서 다시 고른 좌우 선이 그 행에서 실제로 보이면 (line_x_measured) 차로 폭·중심을 잰다.
       평소 위치 표본 = (소실점 x − 차로 중심) × k, k = 카메라 높이 / (y_ref − 수평선) (t32 식을 y_ref 에서)
    3) 근거 프레임이 max(30, ceil(본 프레임 × 10%)) 미만이면 영점 실패 → 이 구간 LDW 는 전부 측정 불가
    4) 검산: 차로 폭 (m) = 폭 px × k 가 LANE_ZERO_W_RANGE_M 밖이면 신뢰도 low. 카메라 높이를 못 쟀거나 (default)
       차량·차선 높이가 어긋났으면 (disagree) 높이를, 아니면 차선 쪽을 의심한다
    Returns dict (cal.lane_baseline 에 넣는다. 키는 frame_schema.md 의 car_calib[].lane_baseline)."""
    import math
    W, H = cal.W, cal.H
    tol, band, margin = lane_zero_params(W, H)
    if relaxed:
        required = max(LANE_ZERO_RELAX_MIN_FRAMES, math.ceil(LANE_ZERO_RELAX_MIN_FRAC * n_examined))
    else:
        required = max(LANE_BASE_MIN_FRAMES, math.ceil(LANE_BASE_MIN_FRAC * n_examined))
    out = {'ldw_measurable': False, 'reason': '', 'y_ref': None, 'y_ref_ratio': None, 'y_visible': None,
           'vis_frac': None, 'pair_frames': 0, 'lane_w_px': None, 'lane_w_m': None, 'm_per_px': None,
           'ego_w_px': None, 'baseline_m': None, 'raw_median_m': None, 'capped': False, 'ldw_shift_px': 0.0,
           'evidence_frames': 0, 'examined_frames': int(n_examined), 'required_frames': required,
           'conf': 'failed', 'conf_reasons': [], 'suspect': None, 'implied_cam_h_m': None,
           'check_independent': not cal.cam_h_src.startswith('lane')}

    def fail(reason):
        out['reason'] = reason
        return out

    if not cal.horizon_ok:
        return fail('no_horizon')
    y_lo = int(math.ceil(cal.horizon_y + LANE_ZERO_MIN_DY * H))
    if y_lo >= H - margin:
        return fail('no_visible_row')
    # 1) 행별 보임 비율
    y_p = max(y_lo, min(int(cal.hood_row), H) - margin)
    vis = np.zeros(H - y_lo)
    n_pair = 0
    for j, lines in enumerate(frames):
        left, right = pick_pair_at(lines, y_p, cal.vp_x)
        if left is None or right is None:
            continue
        cs = _row_cumsum(lane_of(j))
        vis += line_rows_visible(cs, left, y_lo, tol) & line_rows_visible(cs, right, y_lo, tol)
        n_pair += 1
    out['pair_frames'] = n_pair
    if n_pair < required:
        return fail('insufficient_lane_pairs')
    frac = vis / n_pair
    ok = np.nonzero(frac >= LANE_ZERO_VIS_FRAC)[0]
    if not len(ok) or ok.max() < margin:      # 보이는 행이 없거나, 여유를 빼면 수평선 쪽 한계 (y_lo) 위
        return fail('no_visible_row')
    y_vis = y_lo + int(ok.max())
    y_ref = y_vis - margin
    k = cal.cam_h / (y_ref - cal.horizon_y)
    out.update(y_ref=int(y_ref), y_ref_ratio=round(y_ref / H, 3), y_visible=int(y_vis),
               vis_frac=round(float(frac[y_ref - y_lo]), 3), m_per_px=round(k, 5),
               ego_w_px=round(EGO_CAR_W_M / k, 1))
    # 2) y_ref 에서 차로 폭·평소 위치
    widths, samples = [], []
    for j, lines in enumerate(frames):
        left, right = pick_pair_at(lines, y_ref, cal.vp_x)
        if left is None or right is None:
            continue
        lane = lane_of(j)
        xl, xr = line_x_measured(lane, left, y_ref, band, tol), line_x_measured(lane, right, y_ref, band, tol)
        if xl is None or xr is None or xr - xl <= 20 * W / REF_W:
            continue
        widths.append(xr - xl)
        samples.append((cal.vp_x - (xl + xr) / 2) * k)
    out['evidence_frames'] = len(samples)
    if len(samples) < required:
        return fail('insufficient_valid_lane_pairs')
    raw = float(np.median(samples))
    b = max(-LANE_BASE_MAX_M, min(LANE_BASE_MAX_M, raw))
    w_px = float(np.median(widths))
    w_m = w_px * k
    out.update(ldw_measurable=True, reason='ok', baseline_m=round(b, 4), raw_median_m=round(raw, 4),
               capped=bool(abs(raw) > LANE_BASE_MAX_M), ldw_shift_px=round(-b / k, 1),
               lane_w_px=round(w_px, 1), lane_w_m=round(w_m, 3),
               implied_cam_h_m=round(LANE_WIDTH_M * (y_ref - cal.horizon_y) / w_px, 3))
    # 4) 검산
    if not LANE_ZERO_W_RANGE_M[0] <= w_m <= LANE_ZERO_W_RANGE_M[1]:
        out['conf'] = 'low'
        out['conf_reasons'].append(f'lane_width_{w_m:.2f}m_out_of_{LANE_ZERO_W_RANGE_M[0]}-{LANE_ZERO_W_RANGE_M[1]}')
        out['suspect'] = 'cam_h' if (cal.cam_h_src == 'default' or 'disagree' in cal.cam_h_src) else 'lane'
    else:
        out['conf'] = 'high'
    return out


def lane_trust(lb, lane_cal, ref):
    """구간 차선을 믿을지 (t44). lb = lane_zero 결과 (cal.lane_baseline), lane_cal = 차선을 쓴 보정값 (lb 를 잰 것),
    ref = 차선을 뺀 보정값 (decide_calib 에 차선 없이 — 불신이면 바꿔 쓸 값).

    y_ref 차로 폭 (m) 두 개가 모두 LANE_TRUST_W_RANGE_M 안이어야 믿는다.
    1) lane_w_m = lb.lane_w_m (차선을 쓴 보정값으로 잰 폭, t40 값). 믿으면 이 보정값을 그대로 쓰므로 먼저 이것이 맞아야 한다.
    2) lane_w_m_check = 폭 px × 검산 높이 / (y_ref − ref 수평선) — 순환 끊기. 1) 의 수평선·높이는 차선에서 나왔을 수 있다
       (차선 교점 수평선, 차선 높이 = 3.5 m / 차선 기울기 차 → 그 높이로 재면 폭이 늘 3.5 m 근처).
       - 수평선: 늘 차선 없는 것 (FOE → 차량 박스 회귀). 없으면 검산할 기준이 없어 불신
       - 높이: 차선보다 믿을 만한 독립 측정 = 차량 박스 track VEH_STRONG_TRACKS 개 이상 (decide_calib 이 차선과 어긋날 때
         차량을 고르는 기준과 같다) 일 때만 그 값. 아니면 lane_cal 높이 그대로 — 높이 순환은 남는다 (cam_h_independent false).
         기본값 1.4 m 는 측정이 아니라 쓰지 않는다 (트럭 pos_00283 은 차선 높이 2.49 m 가 맞는데 1.4 m 로 재면 1.94 m 로
         불신 → 근접 보험이 사고 밖 100% 경보). track 1~2 개 차량 높이도 크게 튄다 (세단 pos_00077 2.82 m → 6.05 m 로 불신)
    불신 사유와 범위 (exclude_lanes, t48 DECISIONS 2026-10-10):
    - 폭이 범위 밖 (1) lane_width_out_of_range, 2) lane_width_out_of_range_check) → exclude_lanes true: LDW 측정 불가 +
      차선을 수평선·소실점·카메라 높이·교차 확인·근접 보험 축척에서 뺀다 (engine 이 차선 뺀 보정값으로 바꾼다)
    - 영점 실패 (근거 부족 등) / 검산 불가 (차선 없는 수평선 없음, y_ref 가 그 수평선 아래 LANE_ZERO_MIN_DY 보다 가까움)
      → exclude_lanes false: LDW 만 측정 불가, 보정값은 t40 그대로 (차선이 틀렸다는 근거가 아니라 잴 근거가 모자란 것)
    Returns dict (cal.lane_trust, meta car_calib[].lane_trust)."""
    lb = lb or {}
    lo, hi = LANE_TRUST_W_RANGE_M
    n_veh = (ref.detail.get('vehicle_height') or {}).get('n_tracks', 0) if ref.cam_h_src == 'vehicle' else 0
    h_indep = n_veh >= VEH_STRONG_TRACKS
    h, h_src = (ref.cam_h, ref.cam_h_src) if h_indep else (lane_cal.cam_h, lane_cal.cam_h_src)
    out = {'trusted': False, 'exclude_lanes': False, 'reason': '', 'lane_w_m': lb.get('lane_w_m'),
           'lane_w_m_check': None, 'range_m': [lo, hi], 'lane_w_px': lb.get('lane_w_px'), 'y_ref': lb.get('y_ref'),
           'check': {'horizon_y': round(ref.horizon_y, 1), 'horizon_ok': ref.horizon_ok, 'horizon_src': ref.horizon_src,
                     'cam_h_m': round(h, 3), 'cam_h_src': h_src, 'cam_h_independent': h_indep,
                     'vehicle_tracks': n_veh}}
    if not lb.get('ldw_measurable'):
        out['reason'] = f"lane_zero_failed:{lb.get('reason') or 'none'}"
        return out
    dy = lb['y_ref'] - ref.horizon_y
    checkable = ref.horizon_ok and dy >= LANE_ZERO_MIN_DY * ref.H
    w_chk = lb['lane_w_px'] * h / dy if checkable else None
    if checkable:
        out['lane_w_m_check'] = round(w_chk, 3)
    if not lo <= lb['lane_w_m'] <= hi:
        out.update(reason='lane_width_out_of_range', exclude_lanes=True)
    elif not ref.horizon_ok:
        out['reason'] = 'no_lane_free_horizon'
    elif not checkable:
        out['reason'] = 'y_ref_near_lane_free_horizon'
    elif not lo <= w_chk <= hi:
        out.update(reason='lane_width_out_of_range_check', exclude_lanes=True)
    else:
        out.update(trusted=True, reason='ok')
    return out


def lanes_excluded(cal):
    """이 구간이 차선을 모든 계산에서 뺀 구간인가 (t44·t48: 폭이 범위 밖이라 차선 뺀 보정값으로 바꾼 구간)."""
    return bool((getattr(cal, 'lane_trust', None) or {}).get('applied'))


def calib_geom(cal):
    """보정값 중 판정에 쓰는 기하 (meta 비교용)."""
    return {'horizon_y': round(cal.horizon_y, 1), 'horizon_ok': cal.horizon_ok, 'horizon_src': cal.horizon_src,
            'horizon_conf': cal.horizon_conf, 'vp_x': round(cal.vp_x, 1), 'vp_x_src': cal.vp_x_src,
            'cam_h_m': round(cal.cam_h, 3), 'cam_h_src': cal.cam_h_src}


# ============================================================
# 차선 소실점·차선 비율 높이 / 차량 박스 회귀 수평선 / 차량 박스 카메라 높이
# ============================================================
def lane_pair_geom(left, right):
    """자차선 두 줄 (x = k·y + c) → (교점 x, 교점 y, k_R − k_L). 평행이면 None.
    평지에서 k = 옆 거리 / 카메라 높이 → k_R − k_L = 차로폭 / Hc (초점거리 불필요)."""
    kL, cL, kR, cR = left['dydx'], left['c'], right['dydx'], right['c']
    if abs(kR - kL) < 1e-4:
        return None
    vy = (cL - cR) / (kR - kL)
    return kL * vy + cL, vy, kR - kL


def lane_summary(samples, W, H):
    """프레임별 (vx, vy, k_diff) → 차선 소실점과 차선 기반 카메라 높이."""
    s = _scale(W)
    pts = [(vx, vy, kd) for vx, vy, kd in samples
           if -0.2 * W < vx < 1.2 * W and 0.05 * H < vy < 0.9 * H]
    out = {'n_frames': len(pts)}
    if len(pts) < LANE_MIN_FRAMES:
        return out
    a = np.array(pts)
    out.update(x=float(np.median(a[:, 0])), y=float(np.median(a[:, 1])),
               iqr_y=float(np.subtract(*np.percentile(a[:, 1], [75, 25]))))
    if out['iqr_y'] > 3 * AGREE_PX * s:      # 프레임마다 크게 흔들리면 차선 쌍이 일관되지 않다
        out['rejected'] = 'unstable'
        return out
    kd = a[:, 2][a[:, 2] > 0.3]
    if len(kd) >= LANE_MIN_FRAMES:
        h = LANE_WIDTH_M / float(np.median(kd))
        out['cam_h'] = round(h, 3)
        out['cam_h_ok'] = bool(CAM_H_RANGE_M[0] <= h <= CAM_H_RANGE_M[1])
    return out


def _rear_ok(b, x_vp, y_h, hood_row, W, H, conf=0.5):
    """연구 common2.rear_ok — 정면(후면)을 보인 차량 박스. y_h=None 이면 수평선 조건을 뺀다 (회귀용)."""
    x0, y0, x1, y1 = b['bbox']
    w, h, c = x1 - x0, y1 - y0, b['cls']
    s = _scale(W)
    if c not in VEH_WIDTH_M or b['conf'] < conf:
        return False
    if x0 < EDGE_PX or y0 < EDGE_PX or x1 > W - EDGE_PX or y1 > H - EDGE_PX or w < 24 * s:
        return False
    if y_h is not None and y1 - y_h < 8 * s:
        return False
    if y1 > hood_row - 4 * s:
        return False
    if abs((x0 + x1) / 2 - x_vp) > 1.0 * w:
        return False
    ar = h / max(w, 1)
    return (0.55 <= ar <= 1.3) if c == 'car' else (0.6 <= ar <= 2.0)


def box_regression_horizon(samples, x_vp, hood_row, W, H):
    """승용 후면 박스의 바닥 y 와 폭 w 를 직선으로 맞춘 절편 (w=0 인 높이) = 수평선 (veh_regress.py)."""
    rows = [b for b in samples if b['cls'] == 'car' and _rear_ok(b, x_vp, None, hood_row, W, H)]
    out = {'n_boxes': len(rows), 'n_tracks': len({b['tid'] for b in rows})}
    if len(rows) < REG_MIN_BOXES or out['n_tracks'] < REG_MIN_TRACKS:
        return out
    w = np.array([b['bbox'][2] - b['bbox'][0] for b in rows], float)
    yb = np.array([b['bbox'][3] for b in rows], float)
    if w.max() / max(w.min(), 1) < REG_MIN_WRATIO:
        return out
    from scipy.stats import theilslopes
    slope, icpt, _, _ = theilslopes(yb, w)
    out['y'] = float(icpt)
    return out


def vehicle_height(samples, y_h, x_vp, hood_row, W, H):
    """Hc = W·(y_bottom − y_h)/w 를 track 별 중앙값으로 (vehicle_h.py). 승용 track 우선, 없으면 버스·트럭."""
    per = {}
    for b in samples:
        if not _rear_ok(b, x_vp, y_h, hood_row, W, H):
            continue
        x0, _, x1, y1 = b['bbox']
        per.setdefault((b['tid'], b['cls']), []).append(VEH_WIDTH_M[b['cls']] * (y1 - y_h) / (x1 - x0))
    tracks = {k: float(np.median(v)) for k, v in per.items() if len(v) >= VEH_MIN_OBS}
    car = [h for (tid, c), h in tracks.items() if c == 'car']
    big = [h for (tid, c), h in tracks.items() if c != 'car']
    use, cls = (car, 'car') if car else (big, 'bus/truck')
    out = {'n_tracks_car': len(car), 'n_tracks_bus_truck': len(big)}
    if use:
        h = float(np.median(use))
        out.update(cam_h=round(h, 3), cls=cls, n_tracks=len(use),
                   cam_h_ok=bool(CAM_H_RANGE_M[0] <= h <= CAM_H_RANGE_M[1]))
    return out


# ============================================================
# 보정 결과
# ============================================================
_CONF_NUM = {'high': 0.9, 'medium': 0.7, 'low': 0.4, 'failed': 0.0}


@dataclass
class CarCalib:
    """영상 구간 하나의 기하 보정값. horizon_ok=False 면 수평선이 기본값이라 바닥 거리·X 를 쓰지 않는다."""
    W: int
    H: int
    horizon_y: float
    vp_x: float
    horizon_ok: bool
    horizon_src: str            # foe | lane | vehicle_regression | default
    horizon_conf: str           # high | medium | low | failed
    vp_x_src: str               # foe | lane | center
    hood_row: int               # 보닛·대시보드 띠가 시작하는 행 (없으면 H)
    hood_src: str               # flow | flow_motion_edge | flow_none_visible | manual | default
    cam_h: float
    cam_h_src: str              # vehicle | lane | vehicle+lane | vehicle(lane_disagree) | lane(vehicle_disagree) | default
    f_px: float
    f_src: str = 'prior_600px@1280'
    window: tuple = (0.0, 0.0)  # 보정에 쓴 구간 (초)
    detail: dict = field(default_factory=dict)

    @property
    def conf_num(self):
        return _CONF_NUM[self.horizon_conf]

    @property
    def hood_known(self):
        """보닛 행을 실제로 잡았거나 (광류) 수동 지정했으면 True. 추정 실패 (기본 비율) 면 False —
        이때 기본 비율을 확정 보닛처럼 써서 박스를 지우거나 바닥을 가렸다고 보지 않는다 (t24, 미탐 우선)."""
        return self.hood_src != 'default'

    def to_dict(self):
        return {
            'horizon_y': round(self.horizon_y, 1), 'horizon_ratio': round(self.horizon_y / self.H, 3),
            'vp_x': round(self.vp_x, 1), 'horizon_ok': self.horizon_ok,
            'horizon_src': self.horizon_src, 'horizon_conf': self.horizon_conf, 'vp_x_src': self.vp_x_src,
            'hood_row': int(self.hood_row), 'hood_ratio': round(1 - self.hood_row / self.H, 3),
            'hood_src': self.hood_src,
            'cam_h_m': round(self.cam_h, 3), 'cam_h_src': self.cam_h_src,
            'f_px': round(self.f_px, 1), 'f_src': self.f_src,
            'window_sec': [round(self.window[0], 2), round(self.window[1], 2)],
            'interior': getattr(self, 'interior_info', None),
            'lane_baseline': getattr(self, 'lane_baseline', None),
            # t44 차선 신뢰 (engine --no-lane-trust 면 키 없음 = t40 meta 그대로)
            **({'lane_trust': self.lane_trust} if getattr(self, 'lane_trust', None) is not None else {}),
            'detail': self.detail,
        }


def default_calib(W, H, hood_row=None, hood_src='default'):
    return CarCalib(W=W, H=H, horizon_y=HORIZON_DEFAULT * H, vp_x=W / 2, horizon_ok=False,
                    horizon_src='default', horizon_conf='failed', vp_x_src='center',
                    hood_row=int(H if hood_row is None else hood_row), hood_src=hood_src,
                    cam_h=CAM_H_DEFAULT_M, cam_h_src='default', f_px=F_REF_PX * _scale(W))


def decide_calib(W, H, foe, lane, reg_fn, veh_fn, hood_row, hood_src, window):
    """세 방법 결과를 모아 보정값과 출처·신뢰도를 정한다.

    foe:  FlowCollector.solve() 결과, lane: lane_summary() 결과
    reg_fn(x_vp) → box_regression_horizon 결과, veh_fn(y_h, x_vp) → vehicle_height 결과
    수평선: 광류 FOE (통과 쌍 FOE_MIN_PAIRS 이상) > 차선 소실점 > 차량 박스 회귀 > 기본값.
    신뢰도: 기본 방법과 다른 방법이 AGREE_PX 안에서 맞으면 high, 비교할 것이 없으면 medium,
            맞지 않거나 차선·회귀만 있으면 low, 아무것도 없으면 failed.
    소실점 x 는 화면 폭 VP_X_RANGE 안이어야 한다 (밖이면 그 출처를 버림. 모두 밖이면 x 가운데·신뢰도 low).
    """
    s = _scale(W)
    tol = AGREE_PX * s
    foe_ok = (foe.get('n_ok', 0) >= FOE_MIN_PAIRS and 'y' in foe
              and foe['iqr_y'] <= FOE_MAX_IQR_PX * s and foe['iqr_x'] <= 2 * FOE_MAX_IQR_PX * s)
    if 'y' in foe and not foe_ok:
        foe['rejected'] = 'too_few_pairs' if foe.get('n_ok', 0) < FOE_MIN_PAIRS else 'unstable'
    lane_ok = 'y' in lane and not lane.get('rejected')
    in_range = lambda y: 0.05 * H < y < 0.95 * H
    # 수평선 범위를 먼저 거르고, 소실점 x 와 수평선 y 를 같은 출처에서 함께 고른다 (t24: 범위 밖으로 탈락한
    # FOE 의 x 와 차선 VP 의 y 가 섞이던 문제). 차량 회귀는 x 가 없어 FOE·차선이 모두 탈락하면 화면 가운데
    pts = [(k, p) for k, p, ok in (('foe', foe, foe_ok), ('lane', lane, lane_ok)) if ok and in_range(p['y'])]
    for k, p, ok in (('foe', foe, foe_ok), ('lane', lane, lane_ok)):
        if ok and not in_range(p['y']):
            p['rejected'] = 'horizon_out_of_range'
    # 소실점 x 범위 (VP_X_RANGE, t26): 밖이면 그 출처는 버리고 다른 출처를 쓴다. 남은 출처가 없으면
    # 수평선은 그 출처 값을 두고 x 만 화면 가운데로, 신뢰도 low (pos_00754: 정지 영상의 차선 교점 x −12px)
    x_in = lambda x: VP_X_RANGE[0] * W <= x <= VP_X_RANGE[1] * W
    good = [(k, p) for k, p in pts if x_in(p['x'])]
    for k, p in pts:
        if not x_in(p['x']):
            p['rejected'] = 'vp_x_out_of_range'
    x_forced = False
    if good:
        x_vp, x_src = good[0][1]['x'], good[0][0]
    elif pts:
        x_vp, x_src, x_forced = W / 2, 'center(vp_x_out_of_range)', True
        good = pts[:1]
    else:
        x_vp, x_src = W / 2, 'center'
    reg = reg_fn(x_vp)
    cands = [(k, p['y']) for k, p in good]
    if 'y' in reg and in_range(reg['y']):
        cands.append(('vehicle_regression', reg['y']))
    checks = {}
    if not cands:
        cal = default_calib(W, H, hood_row, hood_src)
        cal.window = window
        cal.detail = {'foe': foe, 'lane': lane, 'vehicle_regression': reg, 'checks': checks,
                      'reason': '수평선 추정 실패 — 기본값 (바닥 거리·X 안 씀)'}
        lh = lane.get('cam_h') if lane.get('cam_h_ok') else None
        if lh is not None:                     # 차선 높이는 수평선 없이 구해진다
            cal.cam_h, cal.cam_h_src = lh, 'lane'
        return cal
    src, y_h = cands[0]
    for k, v in cands[1:]:
        checks[k] = {'y': round(v, 1), 'diff_px': round(v - y_h, 1), 'agree': bool(abs(v - y_h) <= tol)}
    agree = any(c['agree'] for c in checks.values())
    if src == 'foe':
        conf = 'high' if agree else ('medium' if not checks else 'low')
    elif src == 'lane':
        conf = 'medium' if agree else 'low'
    else:
        conf = 'low'
    if x_forced:
        conf = 'low'
    # 카메라 높이: 차량 박스 (기본) + 차선 비율 보정. 둘이 CAM_H_AGREE 배 안이면 평균, 어긋나면 근거가 많은 쪽
    # (차량 track VEH_STRONG_TRACKS 개 이상이면 차량, 아니면 차선 — 실측에서 track 1~2개 값은 크게 튄다: pos_00077)
    veh = veh_fn(y_h, x_vp)
    vh = veh.get('cam_h') if veh.get('cam_h_ok') else None
    lh = lane.get('cam_h') if lane.get('cam_h_ok') else None
    if vh is not None and lh is not None and max(vh, lh) / min(vh, lh) > CAM_H_AGREE:
        if veh.get('n_tracks', 0) >= VEH_STRONG_TRACKS:
            cam_h, h_src = vh, 'vehicle(lane_disagree)'
        else:
            cam_h, h_src = lh, 'lane(vehicle_disagree)'
    elif vh is not None and lh is not None:
        cam_h, h_src = (vh + lh) / 2, 'vehicle+lane'
    elif vh is not None:
        cam_h, h_src = vh, 'vehicle'
    elif lh is not None:
        cam_h, h_src = lh, 'lane'
    else:
        cam_h, h_src = CAM_H_DEFAULT_M, 'default'
    return CarCalib(W=W, H=H, horizon_y=float(y_h), vp_x=float(x_vp), horizon_ok=True,
                    horizon_src=src, horizon_conf=conf, vp_x_src=x_src,
                    hood_row=int(hood_row), hood_src=hood_src, cam_h=float(cam_h), cam_h_src=h_src,
                    f_px=F_REF_PX * s, window=window,
                    detail={'foe': foe, 'lane': lane, 'vehicle_regression': reg,
                            'vehicle_height': veh, 'checks': checks})


# ============================================================
# 객체 기하 — 거리 두 가지, 옆 거리 X, 차로 침범
# ============================================================
def object_geom(bbox, cls, cal: CarCalib):
    """박스 → 바닥 위치 거리, 차폭 거리, 경보 거리 (가까운 쪽), X.

    - 화면 좌우 끝에 잘린 박스: 차폭 거리 안 씀 (보이는 폭이 줄어 멀게 나온다)
    - 바닥이 보닛 띠나 화면 아래 끝에 닿은 박스: 바닥 거리 안 씀 (실제 바닥은 더 아래 = 더 가깝다).
      대신 보닛 경계 (화면 아래 끝) 까지의 거리가 상한이다 ('hood_bound'). 차폭 거리가 있어도 상한과 비교해
      가까운 쪽을 쓴다 (t24: 차폭 6.5m 가 상한 4.88m 를 이겨 위험이 주의로 내려가던 문제, neg_01988 8.8초).
      보닛 추정이 실패했으면 (cal.hood_known=False) 기본 비율 대신 화면 아래 끝으로 본다.
      보정 신뢰도 low 구간은 상한을 쓰지 않는다 (t26, DECISIONS 2026-10-09) — 바닥·차폭 거리가 모두 없으면 거리 없음
    - 차량 (car/bus/truck, 차종 모르는 보조 박스 포함): 바닥 거리로 역산한 차폭 Hc·w/(y_b − y_h) 가 1.6~2.5m 밖이면 두 값이 맞지 않는
      조합이다. 대부분 옆면이 섞인 박스 (차폭 거리가 가깝게 나옴) 라 차폭 거리를 버리고 바닥 거리를 쓴다.
      차폭이 1.6m 보다 작게 역산되는 경우도 바닥 거리가 가까운 쪽이라 같은 선택이 된다 (미탐 우선)
    - 수평선 보정 실패 (horizon_ok=False): 바닥 거리 안 씀, X 는 차폭 거리로 환산
    """
    W, H = cal.W, cal.H
    s = _scale(W)
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    cx = (x0 + x1) / 2
    side_cut = x0 <= EDGE_PX or x1 >= W - 1 - EDGE_PX
    bottom_lim = min(cal.hood_row if cal.hood_known else H, H - 1 - EDGE_PX)
    bottom_hidden = y1 >= bottom_lim - 4 * s
    dy = y1 - cal.horizon_y
    f, Hc = cal.f_px, cal.cam_h
    gated = []
    ground = bound = None
    ground_ok = cal.horizon_ok and not bottom_hidden and dy > 8 * s
    if ground_ok:
        ground = f * Hc / dy
    elif not cal.horizon_ok:
        gated.append('ground:수평선 기본값')
    elif bottom_hidden:
        gated.append('ground:바닥 가림')
        if cal.horizon_conf == 'low':     # t26: 신뢰도 low 구간은 근접 상한을 쓰지 않는다 (neg_01711 거리 대부분이 상한)
            gated.append('hood_bound:보정 신뢰도 low')
        elif bottom_lim - cal.horizon_y > 8 * s:
            bound = f * Hc / (bottom_lim - cal.horizon_y)
    else:
        gated.append('ground:수평선 위')
    prior = VEH_WIDTH_M.get(cls) or OTHER_WIDTH_M.get(cls, 1.8)
    width = None
    if side_cut:
        gated.append('width:박스 잘림')
    elif w >= 4:
        width = f * prior / w
    w_impl = None
    width_used = width
    # 차폭 게이트: 차량과 차종을 모르는 보조 박스 (mask_object, unknown_near — 차폭 1.8m prior 를 쓰는 박스).
    # 보조 박스는 붙은 차 여러 대를 한 덩어리로 잡는 경우가 있어 게이트가 더 필요하다. 사람·이륜차는 뺀다
    if cls not in OTHER_WIDTH_M and ground is not None and width is not None:
        w_impl = Hc * w / dy
        if not (WIDTH_GATE_M[0] <= w_impl <= WIDTH_GATE_M[1]):
            width_used = None
            gated.append('width:차폭 범위 밖')
    valid = {k: v for k, v in (('ground', ground), ('width', width_used), ('hood_bound', bound))
             if v is not None and 0.5 < v < 200}
    if valid:
        used = min(valid, key=valid.get)
        d = valid[used]
    else:
        used, d = None, None
    # 옆 거리 X — 바닥이 보이면 f 없는 식, 아니면 쓴 거리로 환산
    if ground is not None:
        k = Hc / dy
    elif d is not None:
        k = d / f
    else:
        k = None
    X = Xl = Xr = None
    if k is not None:
        X, Xl, Xr = (cx - cal.vp_x) * k, (x0 - cal.vp_x) * k, (x1 - cal.vp_x) * k
    return {'ground': ground, 'width': width, 'width_used': width_used, 'width_implied_m': w_impl,
            'bound': bound, 'd': d, 'used': used, 'gated': gated,
            'X': X, 'X_l': Xl, 'X_r': Xr, 'w_px': w, 'dy': dy,
            'side_cut': side_cut, 'bottom_hidden': bottom_hidden, 'ground_ok': ground is not None}


def cutin_side(g):
    """차로 침범 — 박스 바닥이 내 차로 경계선 (±1.75m) 을 걸치고 안쪽 모서리가 CUTIN_INNER_M 안.
    CUTIN_INNER_M 1.25m = 차선을 0.5m 이상 넘어온 경우만 (DECISIONS 2026-10-09: 박스에 차 옆면이 섞여 안쪽으로
    넓게 잡히는 오탐 억제).
    Returns 'from_left' | 'from_right' | ''."""
    Xl, Xr = g['X_l'], g['X_r']
    if Xl is None or Xr is None:
        return ''
    if Xl > 0 and Xl < CUTIN_INNER_M and Xr > LANE_LINE_M:
        return 'from_right'
    if Xr < 0 and -Xr < CUTIN_INNER_M and -Xl > LANE_LINE_M:
        return 'from_left'
    return ''


def corridor_bottom_x(cal: CarCalib):
    """내 차로 (|X| ≤ EGO_LANE_HALF_M) 경계선의 화면 하단 x. 선은 소실점을 지난다."""
    k = EGO_LANE_HALF_M / cal.cam_h * (cal.H - 1 - cal.horizon_y)
    return cal.vp_x - k, cal.vp_x + k


def corridor_alphas(cal: CarCalib, d_crit, d_danger, d_caution):
    """mvp.build_corridor 의 구역 높이 (소실점→하단 비율) 를 거리로 정한다: 거리 d 인 행의 비율 = d_bottom / d."""
    d_bottom = cal.f_px * cal.cam_h / max(cal.H - 1 - cal.horizon_y, 1)
    a = lambda d: float(np.clip(d_bottom / d, 0.01, 0.99))
    return {'critical': a(d_crit), 'danger': a(d_danger), 'caution': a(d_caution)}


# ============================================================
# TTC — 박스 폭 증가율, 바닥 위치 변화율 (0.5초 창 직선 맞춤, 연구 ttc.py 를 인과적 창으로). 기록만
# ============================================================
def _log_rate(ts, vals):
    """ln(vals) 를 시간에 직선으로 맞춘 기울기와 표준오차. 조건 미달이면 None."""
    n = len(ts)
    if n < TTC_MIN_N or ts[-1] - ts[0] < TTC_MIN_SPAN:
        return None
    t = np.asarray(ts, float)
    y = np.log(np.asarray(vals, float))
    tt = t - t.mean()
    sxx = float((tt ** 2).sum())
    b = float((tt * (y - y.mean())).sum() / sxx)
    res = y - y.mean() - b * tt
    se = float(np.sqrt((res ** 2).sum() / (n - 2) / sxx))
    return b, se


class TtcTracker:
    """track_id 별 최근 관측 (t, 박스 폭, 바닥 − 수평선). 폭이 잘렸거나 바닥이 가려진 관측은 그 방법에서만 뺀다."""

    def __init__(self):
        self.hist = {}

    def reset(self):
        self.hist = {}

    def add(self, tid, t, w, dy):
        h = self.hist.setdefault(tid, [])
        if h and h[-1][0] >= t:
            return
        h.append((t, w, dy))
        while h and h[0][0] < t - TTC_WIN_SEC - 0.6:
            h.pop(0)

    def prune(self, alive):
        self.hist = {k: v for k, v in self.hist.items() if k in alive}

    def ttc(self, tid, t):
        """Returns {'scale_s', 'ground_s', 'rate_scale', 'se_scale', 'rate_ground', 'se_ground'} (None 포함)."""
        h = [r for r in self.hist.get(tid, []) if t - TTC_WIN_SEC - 1e-6 <= r[0] <= t + 1e-6]
        out = {}
        for name, idx in (('scale', 1), ('ground', 2)):
            pts = [(r[0], r[idx]) for r in h if r[idx] is not None and r[idx] > 0]
            fit = _log_rate([p[0] for p in pts], [p[1] for p in pts]) if pts else None
            tau = None
            if fit is not None:
                out[f'rate_{name}'], out[f'se_{name}'] = round(fit[0], 4), round(fit[1], 4)
                if fit[0] > 0 and fit[1] < TTC_SE_MAX:
                    tau = 1.0 / fit[0]
            out[f'{name}_s'] = tau
        return out


# ============================================================
# 박스 없는 근접 물체 보험 (Depth Pro) — 원값을 거리로 쓰지 않고, 같은 프레임 도로 픽셀로 축척을 맞춘다
# ============================================================
def depth_guard(depth, lane_mask, cal: CarCalib, boxes, scales):
    """내 앞 (소실점 열 ± DG_HALF_W_M 띠, 수평선 ~ 보닛) 에 박스로 덮이지 않은 가까운 물체가 있는지.

    1) 축척: 도로·차선 픽셀 (class 2·3, 박스 밖, 수평선과 보닛 사이 아래쪽 3/4) 에서
       s = 중앙값(지면 기하 거리 f·Hc/(y − y_h) ÷ Depth Pro 원값). 도로가 부족하면 직전 축척들의 중앙값.
       [t48] 차선을 뺀 구간 (lanes_excluded — 영점 차로 폭이 범위 밖) 은 차선 픽셀 (class 3) 을 빼고 도로만 쓴다
    2) 물체 픽셀: s·D < DG_RANGE_M 이고 s·D < DG_RATIO × 그 행의 지면 거리 (지면이면 같아야 한다)
    3) 박스 밖 물체 픽셀이 띠의 DG_MIN_FRAC 이상이면 경보. 거리 = 물체 픽셀 s·D 의 20 퍼센타일
    scales: 구간별 축척 이력 (list, 이 함수가 갱신). Returns dict.
    """
    if not cal.horizon_ok:
        return {'status': 'no_geom'}
    H, W = depth.shape
    s_px = _scale(W)
    y_top = int(np.ceil(cal.horizon_y + 2))
    y_bot = int(min(cal.hood_row, H))
    if y_bot - y_top < 10:
        return {'status': 'no_geom'}
    rows = np.arange(y_top, y_bot, dtype=np.float32)
    zg = (cal.f_px * cal.cam_h / (rows - cal.horizon_y))[:, None]       # 행별 지면 거리
    im = getattr(cal, 'interior_mask', None)
    covered = im.copy() if im is not None and im.shape == (H, W) else np.zeros((H, W), bool)   # 실내는 물체 아님
    for b in boxes:
        covered[b[1]:b[3] + 1, b[0]:b[2] + 1] = True
    # 1) 축척
    r0 = int(y_top + 0.25 * (y_bot - y_top))
    band_d = depth[r0:y_bot]
    ground_cls = (2,) if lanes_excluded(cal) else (2, 3)
    road = np.isin(lane_mask[r0:y_bot], ground_cls) & ~covered[r0:y_bot] & (band_d > DG_FLOOR)
    scale_src = 'frame'
    if road.sum() >= DG_MIN_ROAD_PX * s_px * s_px:
        ratio = np.broadcast_to(zg[r0 - y_top:], band_d.shape)[road] / band_d[road]
        sc = float(np.median(ratio))
        scales.append(sc)
        del scales[:-DG_SCALE_KEEP]
    elif scales:
        sc, scale_src = float(np.median(scales)), 'history'
    else:
        return {'status': 'no_scale'}
    # 2) 검사 띠
    hw = cal.f_px * DG_HALF_W_M / DG_RANGE_M
    xa, xb = int(max(0, cal.vp_x - hw)), int(min(W, cal.vp_x + hw + 1))
    if xb - xa < 4:
        return {'status': 'no_geom'}
    dm = depth[y_top:y_bot, xa:xb] * sc
    obst = (dm < DG_RANGE_M) & (dm < DG_RATIO * zg) & ~covered[y_top:y_bot, xa:xb]
    frac = float(obst.sum()) / dm.size
    out = {'status': 'clear', 'scale': round(sc, 3), 'scale_src': scale_src, 'frac': round(frac, 3)}
    if frac >= DG_MIN_FRAC:
        out.update(status='alarm', distance_m=round(float(np.percentile(dm[obst], 20)), 2))
    return out
