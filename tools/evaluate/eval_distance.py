"""eval_distance.py — 거리 추정 방식 비교 (roadscope.engine --dump-estimates 의 estimates.jsonl).

    python tools/evaluate/eval_distance.py runs_out/<영상1> runs_out/<영상2> ... [--out runs_out/distance_eval.md]
        [--continuous accident_0029_v6_hud]          # 시간 안정성을 볼 연속 영상 (폴더 이름)
        [--probe VIDEO FRAME x0,y0,x1,y1 CLS LABEL]   # 검출되지 않은 박스를 직접 재기 (GPU 필요)

정답 거리가 없어서 간접 지표만 본다:
  커버리지 / 방식 간 일치도 (depth 대비 비율) / 상식 위반 (가까운 게 확실한데 5m 초과, 사람 키 1.7m
  환산 거리와 1.5배 이상 차이) / 시간 안정성 (같은 track_id 연속 평가의 |Δd|/d) / 의심 사례 /
  방식별 판정 시뮬레이션 (engine.judge 의 경고 단계를 그대로 다시 계산)

방식 이름:
  fused        지금 판정에 쓰는 융합값 (ground·size·depth 가중 평균, 이상치 제거)
  ground       박스 하단 y + VP + 카메라 높이 1.0m
  size         박스 폭 + 차종 폭 prior
  depth        Depth Pro 박스 전체 중앙값 (지금 fused 의 depth 성분)
  depth_center Depth Pro 박스 가운데 절반 중앙값
  depth_bottom Depth Pro 박스 아래 20% 중앙값
  depth_p25    Depth Pro 박스 전체 하위 25%
  *_ffix       영상 1개 안에서 초점거리를 중앙값으로 고정해 다시 스케일한 값 (연속 영상만 의미 있음).
               Depth Pro 의 미터 깊이와 ground·size 는 모두 그 프레임 초점거리에 비례한다
  fused_gated  후보 조합: 지금 융합에서 무효한 성분만 뺀다 — 화면 가장자리에 잘린 박스면 size 제외
               (폭이 줄어 멀게 나옴), 기본 VP 를 쓴 프레임이면 ground 제외 (지평선 근거 없음)
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from roadscope import rules as mvp
from roadscope import engine

BASE = ['fused', 'fused_gated', 'ground', 'size', 'depth', 'depth_center', 'depth_bottom', 'depth_p25']
FFIX = ['depth_ffix', 'depth_bottom_ffix', 'ground_ffix', 'size_ffix']
EDGE_PX = 3              # 박스가 화면 가장자리에서 이 안이면 잘린 것으로 본다
CLOSE_AREA = 0.15        # 박스 면적 비율 — 이 이상이면 가까운 게 확실
CLOSE_BOTTOM = 0.95      # 박스 하단이 화면 높이의 95% 아래
PERSON_H_M = 1.7
SIM_METHODS = ['fused', 'fused_gated', 'depth', 'depth_center', 'depth_bottom',
               'depth_ffix', 'depth_bottom_ffix']


# ------------------------------------------------------------
# 읽기 / 파생값
# ------------------------------------------------------------
def truncated(bbox, W, H):
    x0, y0, x1, y1 = bbox
    return x0 <= EDGE_PX or y0 <= EDGE_PX or x1 >= W - 1 - EDGE_PX or y1 >= H - 1 - EDGE_PX


def fused_gated(ground, size, depth, bbox, W, H, vp_default):
    est = {'ground': None if vp_default else ground,
           'size': None if truncated(bbox, W, H) else size,
           'depth': depth}
    return mvp.fuse_distances(est, depth_is_metric=True)['d_fused_m']

def load_run(run_dir):
    run_dir = Path(run_dir)
    frames = [json.loads(l) for l in open(run_dir / 'estimates.jsonl', encoding='utf-8')]
    meta = json.loads((run_dir / 'meta.json').read_text(encoding='utf-8'))
    f_med = float(np.median([f['f_px'] for f in frames])) if frames else None
    for f in frames:
        k = f_med / f['f_px'] if (f_med and f['f_px']) else None
        for o in f['objects']:
            for m in ('depth', 'depth_bottom', 'ground', 'size'):
                o[m + '_ffix'] = round(o[m] * k, 2) if (k and o.get(m) is not None) else None
            o['fused_gated'] = fused_gated(o.get('ground'), o.get('size'), o.get('depth'),
                                           o['bbox'], f['W'], f['H'], f['vp_default'])
    return {'name': run_dir.name, 'dir': str(run_dir), 'frames': frames, 'meta': meta, 'f_med': f_med}


def objects(run):
    for f in run['frames']:
        for o in f['objects']:
            yield f, o


def q(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else float('nan')


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{nd}f}'


# ------------------------------------------------------------
# 지표
# ------------------------------------------------------------
def coverage(run, methods):
    objs = [o for _, o in objects(run)]
    n = len(objs)
    return n, {m: (sum(o.get(m) is not None for o in objs) / n if n else float('nan')) for m in methods}


def agreement(run, methods):
    out = {}
    for m in methods:
        r = [o[m] / o['depth'] for _, o in objects(run)
             if o.get(m) is not None and o.get('depth')]
        out[m] = (len(r), q(r, 25), q(r, 50), q(r, 75))
    return out


def close_violations(run, methods):
    close = [(f, o) for f, o in objects(run)
             if o['area_ratio'] >= CLOSE_AREA or o['bbox'][3] >= CLOSE_BOTTOM * f['H']]
    res = {m: sum(1 for _, o in close if o.get(m) is not None and o[m] > 5.0) for m in methods}
    return len(close), res, close


def person_check(run, methods):
    """사람 박스 높이로 환산한 거리 d_h = f·1.7/h 대비 1.5배 이상 벗어난 수. 잘린 박스는 제외.
    d_h 와 Depth Pro·ground·size 는 모두 같은 프레임 초점거리에 비례하므로 비율은 초점 오차와 무관."""
    rows = []
    for f, o in objects(run):
        if o['cls'] != 'person':
            continue
        x0, y0, x1, y1 = o['bbox']
        if y0 <= 1 or y1 >= f['H'] - 2 or y1 - y0 < 10:
            continue
        rows.append((f, o, f['f_px'] * PERSON_H_M / (y1 - y0)))
    out = {}
    for m in methods:
        r = [o[m] / dh for _, o, dh in rows if o.get(m) is not None]
        bad = sum(1 for v in r if v > 1.5 or v < 1 / 1.5)
        out[m] = (len(r), bad, q(r, 50))
    return len(rows), out, rows


def temporal(run, methods):
    """같은 track_id 가 연속 평가 프레임에 다시 나온 쌍의 |Δd| / 평균 d.
    트래커 유령 (ghost) 은 뺀다 — 예측값이라 측정 안정성과 무관하다."""
    step = run['meta']['fps_src'] / run['meta']['fps_eval']
    tracks = {}
    for f, o in objects(run):
        if o['track_id'] is not None and not o.get('ghost'):
            tracks.setdefault(o['track_id'], []).append((f['frame_idx'], o))
    out = {m: [] for m in methods}
    for seq in tracks.values():
        seq.sort(key=lambda t: t[0])
        for (i1, a), (i2, b) in zip(seq, seq[1:]):
            if i2 - i1 > 1.5 * step:
                continue
            for m in methods:
                if a.get(m) and b.get(m):
                    out[m].append(abs(b[m] - a[m]) / ((a[m] + b[m]) / 2))
    return {m: (len(v), q(v, 50), q(v, 75), q(v, 90)) for m, v in out.items()}


# ------------------------------------------------------------
# 판정 시뮬레이션 — engine.judge 의 경고 단계 (8~10) 를 같은 순서로 다시 계산
# ------------------------------------------------------------
def simulate_frame(f, method):
    W, H = f['W'], f['H']
    vp_x = f['vp_used'][0]
    objs = []
    for o in f['objects']:
        d = o.get(method)
        lat = (o['cx'] - vp_x) * d / f['f_px'] if (d is not None and f['f_px']) else None
        objs.append({'bbox': tuple(o['bbox']), 'cx': o['cx'], 'cx_rel': o['cx_rel'],
                     'dist_info': {'d_fused_m': d}, 'lateral_m': lat,
                     'zone_sev': o['zone_sev'], 'strict': o['strict'], 'x1': o['x1']})
    front = [o for o in objs if 0.35 <= o['cx_rel'] <= 0.65]
    left = [o for o in objs if o['cx_rel'] < 0.35]
    right = [o for o in objs if o['cx_rel'] > 0.65]
    strict = [o for o in objs if o['strict']]
    x1 = [o for o in objs if o['x1']]
    violators = strict + x1

    fcw = mvp.detect_fcw(front, W, H)
    ldw = mvp.WarningResult('LDW', f['levels']['LDW'])
    bsw = mvp.detect_bsw(left, right, strict, x1)
    if bsw.level == 0:
        bsw = engine.bsw_l1(objs, violators) or bsw
    head = mvp.detect_head(front, strict, x1, W, H)
    if f['corridor_active']:
        cut_in = bsw.level >= 2 or head.level >= 2
        if (fcw.level == 3 and fcw.extra.get('area_ratio', 0) > 0.25
                and fcw.bbox is not None and not cut_in):
            m = next((o for o in objs if o['bbox'] == tuple(fcw.bbox)), None)
            if m is not None and m['zone_sev'] < 2:
                fcw = mvp.WarningResult('FCW', 2, fcw.reason, bbox=fcw.bbox,
                                        extra={**fcw.extra, 'zone_clamped': True})
    ds = lambda lst: min([o['dist_info']['d_fused_m'] for o in lst
                          if o['dist_info']['d_fused_m'] is not None], default=None)
    state = mvp.V5State(fcw=fcw, ldw=ldw, bsw=bsw, head=head,
                        d_front_m=ds(front), d_left_m=ds(left), d_right_m=ds(right))
    mvp.suppress_short_range_warnings(state)
    if state.head.level > 0:
        ok = state.fcw.level > 0 and (state.fcw.extra and state.fcw.extra.get('distance_m', 999) <= 4.5)
        if not ok:
            state.head.level = 0
    if state.bsw.level == 0 and violators:
        state.bsw = engine.bsw_l1(objs, violators)
    return {'FCW': state.fcw.level, 'BSW': state.bsw.level, 'HEAD': state.head.level,
            'LDW': state.ldw.level, 'bin': state.max_severity}


def simulate(run, method):
    sims = [simulate_frame(f, method) for f in run['frames']]
    return {
        'bin': Counter(s['bin'] for s in sims),
        'FCW': Counter(s['FCW'] for s in sims),
        'BSW': Counter(s['BSW'] for s in sims),
        'HEAD': Counter(s['HEAD'] for s in sims),
        'sims': sims,
    }


# ------------------------------------------------------------
# 직접 재기 (검출 안 된 박스)
# ------------------------------------------------------------
def segment_f_fixed(run, frame_idx):
    """meta.json 의 초점 고정 구간에서 frame_idx 가 속한 구간의 f_fixed (없으면 None)."""
    lock = run['meta'].get('focal_lock') or {}
    for sg in lock.get('segments', []):
        if sg['start_frame'] <= frame_idx <= sg['end_frame']:
            return sg['f_fixed']
    return None


def probe(video, frame_idx, bbox, cls, f_med=None, f_fixed=None):
    import cv2
    models = engine.load_lane_models(engine.LANE_CKPTS, 'cuda')
    proc, dmodel = engine.load_depth_pro('cuda')
    cap = cv2.VideoCapture(str(video))
    i, frame = 0, None
    while True:   # 순차 디코드 (engine 과 같은 프레임)
        ok, fr = cap.read()
        if not ok:
            break
        if i == frame_idx:
            frame = fr
            break
        i += 1
    cap.release()
    if frame is None:
        return None
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = engine.predict_lane_mask(models, rgb, 'cuda')
    lane = engine.find_lanes(mask, mvp.get_ego_mask(mask))
    vp_y = lane['vp'][1] if lane['vp'][2] >= 0.3 else int(H * 0.4)
    depth, f_px = engine.predict_depth(proc, dmodel, rgb, 'cuda')
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    obj_class = cls if cls in mvp.OBJECT_WIDTH_M else 'moveable'
    g = mvp.distance_from_ground(bbox, vp_y, f_px)
    s = mvp.distance_from_size(bbox, obj_class, f_px)
    d = mvp.distance_from_depth(depth, bbox, is_metric=True)
    fused = mvp.fuse_distances({'ground': g, 'size': s, 'depth': d}, depth_is_metric=True)['d_fused_m']
    patch = depth[y0:y1 + 1, x0:x1 + 1]
    vp_default = lane['vp'][2] < 0.3
    out = {'f_px': round(f_px, 1), 'vp_default': vp_default, 'area_ratio': round(bw * bh / (W * H), 3),
           'fused': fused, 'ground': g, 'size': s, 'depth': d,
           'fused_gated': fused_gated(g, s, d, bbox, W, H, vp_default),
           'depth_center': engine._median_valid(depth[y0 + bh // 4:y1 - bh // 4 + 1, x0 + bw // 4:x1 - bw // 4 + 1]),
           'depth_bottom': engine._median_valid(depth[y1 - max(bh // 5, 1):y1 + 1, x0 + bw // 5:x1 - bw // 5 + 1]),
           'depth_p25': round(float(np.percentile(patch[(patch > 0.3) & (patch < 100)], 25)), 2)}
    if f_med:
        k = f_med / f_px
        for m in ('depth', 'depth_bottom', 'ground', 'size'):
            out[m + '_ffix'] = round(out[m] * k, 2) if out[m] is not None else None
    if f_fixed:
        # engine (t14) 이 이 박스를 검출했다면 낼 최종값: 구간 고정 초점 + 무효 성분 제외
        k = f_fixed / f_px
        g2 = mvp.distance_from_ground(bbox, vp_y, f_fixed)
        s2 = mvp.distance_from_size(bbox, obj_class, f_fixed)
        d2 = d * k if (d is not None and 0.3 < d * k < 100) else None
        out['final_locked'] = fused_gated(g2, s2, d2, bbox, W, H, vp_default)
        out['f_fixed'] = f_fixed
    return out


# ------------------------------------------------------------
# 보고서
# ------------------------------------------------------------
def why_changed(base_rec, cur_rec, cur_est):
    """기준에서 L2 이상이던 경고가 지금 어떻게 됐는지 한 줄 (같은 박스의 거리·제외 성분·초점 배율)."""
    est_by_box = {tuple(o['bbox']): o for o in (cur_est or {}).get('objects', [])}
    lock = (cur_est or {}).get('lock') or {}
    parts = []
    for k in ('FCW', 'BSW', 'HEAD'):
        w = base_rec['warnings'][k]
        if w['level'] < 2:
            continue
        txt = f'{k} L{w["level"]} "{w["reason"]}" → L{cur_rec["warnings"][k]["level"]}'
        bo = next((o for o in base_rec['objects'] if o['warning'] == k), None)
        co = next((o for o in cur_rec['objects'] if bo and o['bbox'] == bo['bbox']), None)
        if bo and co:
            txt += f' / 같은 박스 거리 {bo["distance_m"]}→{co["distance_m"]}m'
            e = est_by_box.get(tuple(bo['bbox']))
            if e:
                txt += (f' (depth {e["depth"]}, ground {e["ground"]}, size {e["size"]}'
                        f'{", 제외 " + "·".join(e["gated"]) if e["gated"] else ""}'
                        f'{", depth 배율 " + str(lock.get("depth_scale")) if lock else ""})')
        elif bo:
            txt += ' / 같은 박스 없음'
        parts.append(txt)
    return '; '.join(parts)


def baseline_section(runs, baselines, continuous):
    L = ['\n## 7. 기준 실행 대비 (--baseline)\n']
    L.append('| 영상 | 등급 기준 → 지금 (안전/주의/위험/치명) | FCW L1/L2/L3 | BSW L1/L2/L3 | 초점 고정 구간 | 2회차 시간 |')
    L.append('|---|---|---|---|---|---|')
    g = lambda c, ks: '/'.join(str(c.get(k, 0)) for k in ks)
    for r in runs:
        b = baselines.get(r['name'])
        if not b:
            continue
        cb, cc = (Counter(x['bin'] for x in recs) for recs in (b['records'], r['records']))
        fb, fc = (Counter(x['warnings']['FCW']['level'] for x in recs) for recs in (b['records'], r['records']))
        sb, sc = (Counter(x['warnings']['BSW']['level'] for x in recs) for recs in (b['records'], r['records']))
        lock = r['meta'].get('focal_lock') or {}
        L.append(f'| {r["name"]} | {g(cb, [0, 1, 2, 3])} → {g(cc, [0, 1, 2, 3])} '
                 f'| {g(fb, [1, 2, 3])} → {g(fc, [1, 2, 3])} | {g(sb, [1, 2, 3])} → {g(sc, [1, 2, 3])} '
                 f'| {lock.get("n_segments", "끔")} | {lock.get("pass2_sec", "—")}s |')

    L.append('\n### 7-0. 객체 출처별 수 (frames.jsonl objects 의 source) — 기준 / 지금\n')
    L.append('| 영상 | 기준 | 지금 | 두 성분 가까운 쪽 (near_of_two) 적용 객체 |')
    L.append('|---|---|---|---|')
    for r in runs:
        b = baselines.get(r['name'])
        if not b:
            continue
        cnt = lambda recs: dict(Counter(o.get('source', 'yolo') for x in recs for o in x['objects']))
        n2 = sum('near_of_two' in o.get('gated', []) for _, o in objects(r))
        L.append(f'| {r["name"]} | {cnt(b["records"])} | {cnt(r["records"])} | {n2} |')

    L.append('\n### 7-1. 위험 이상 (L2+) 프레임 — 기준 / 지금\n')
    L.append('| 영상 | 기준 | 지금 |')
    L.append('|---|---|---|')
    fmt_l2 = lambda recs: ', '.join(f'{x["t_sec"]:.1f}({x["bin"]} {x["top_warning"]})'
                                    for x in recs if x['bin'] >= 2) or '—'
    for r in runs:
        b = baselines.get(r['name'])
        if b:
            L.append(f'| {r["name"]} | {fmt_l2(b["records"])} | {fmt_l2(r["records"])} |')

    L.append('\n### 7-2. 등급이 내려간 위험 이상 프레임과 이유 (미탐 우선 원칙 점검)\n')
    est_by_frame = {(r['name'], f['frame_idx']): f for r in runs for f in r['frames']}
    rows = []
    for r in runs:
        b = baselines.get(r['name'])
        if not b:
            continue
        cur = {x['frame_idx']: x for x in r['records']}
        for x in b['records']:
            y = cur.get(x['frame_idx'])
            if y and x['bin'] >= 2 and y['bin'] < x['bin']:
                rows.append(f'- {r["name"]} {x["t_sec"]:.1f}s: 등급 {x["bin"]}→{y["bin"]} — '
                            + why_changed(x, y, est_by_frame.get((r['name'], x['frame_idx']))))
    L.extend(rows or ['- 없음'])

    L.append('\n### 7-3. 시간 안정성 (fused, 연속 영상) — 기준 / 지금\n')
    L.append('| 영상 | 쌍 | 중앙 | 75% | 90% |')
    L.append('|---|---|---|---|---|')
    for r in runs:
        b = baselines.get(r['name'])
        if not b or r['name'] not in continuous:
            continue
        for tag, run in (('기준', b), ('지금', r)):
            n, p50, p75, p90 = temporal(run, ['fused'])['fused']
            L.append(f'| {r["name"]} ({tag}) | {n} | {fmt(p50)} | {fmt(p75)} | {fmt(p90)} |')
    return L


def write_report(runs, continuous, probes, out_path, baselines=None, title='거리 추정 방식 비교'):
    L = [f'# {title}\n',
         '정답 거리가 없어 간접 지표만 본다. 판정 코드는 그대로이고, 값은 `engine.py --dump-estimates` 의 '
         '`estimates.jsonl` 에서 계산했다 (`tools/eval_distance.py`).\n',
         '- 방식: ' + ', '.join(f'`{m}`' for m in BASE + FFIX) + ' (설명은 eval_distance.py 머리말)',
         '- `*_ffix` 는 영상 안에서 초점거리를 중앙값으로 고정한 값. 연속 영상(accident)에서만 의미가 있다\n']

    L.append('## 0. 영상과 초점거리\n')
    L.append('| 영상 | 평가 프레임 | 객체 | 초점거리/W (min / 중앙 / max) | 기본 VP 사용 프레임 |')
    L.append('|---|---|---|---|---|')
    for r in runs:
        fs = [f['f_px'] / f['W'] for f in r['frames']]
        n = sum(len(f['objects']) for f in r['frames'])
        L.append(f'| {r["name"]} | {len(r["frames"])} | {n} | {min(fs):.2f} / {np.median(fs):.2f} / {max(fs):.2f} '
                 f'| {sum(f["vp_default"] for f in r["frames"])} |')
    L.append('\nf_px 출처: ' + ', '.join(f'{r["name"]} {dict(Counter(f["f_src"] for f in r["frames"]))}' for r in runs) + '\n')

    obj_runs = [r for r in runs if any(f['objects'] for f in r['frames'])]
    methods = BASE + FFIX

    L.append('## 1. 커버리지 (값이 나온 객체 비율)\n')
    L.append('| 영상 | 객체 | ' + ' | '.join(methods) + ' |')
    L.append('|---|---|' + '---|' * len(methods))
    for r in obj_runs:
        n, cov = coverage(r, methods)
        L.append(f'| {r["name"]} | {n} | ' + ' | '.join(f'{cov[m] * 100:.0f}%' for m in methods) + ' |')

    L.append('\n## 2. 방식 간 일치도 — Depth Pro(`depth`) 대비 비율 (사분위 1 / 중앙 / 3)\n')
    am = ['fused', 'fused_gated', 'ground', 'size', 'depth_center', 'depth_bottom', 'depth_p25']
    L.append('| 영상 | ' + ' | '.join(am) + ' |')
    L.append('|---|' + '---|' * len(am))
    for r in obj_runs:
        ag = agreement(r, am)
        L.append(f'| {r["name"]} | ' + ' | '.join(f'{fmt(ag[m][1])} / **{fmt(ag[m][2])}** / {fmt(ag[m][3])}'
                                                  for m in am) + ' |')

    L.append('\n## 3. 상식 위반\n')
    L.append(f'### 3-1. 가까운 게 확실한 객체 (박스 면적 ≥ {CLOSE_AREA:.0%} 또는 하단이 화면 {CLOSE_BOTTOM:.0%} 아래) 중 5m 초과\n')
    L.append('| 영상 | 대상 | ' + ' | '.join(methods) + ' |')
    L.append('|---|---|' + '---|' * len(methods))
    for r in obj_runs:
        n, res, _ = close_violations(r, methods)
        L.append(f'| {r["name"]} | {n} | ' + ' | '.join(str(res[m]) for m in methods) + ' |')
    L.append(f'\n### 3-2. 사람 — 키 {PERSON_H_M}m 환산 거리 대비 1.5배 이상 차이 (위반 수 / 비교 수, 중앙 비율)\n')
    L.append('| 영상 | 사람 박스 | ' + ' | '.join(methods) + ' |')
    L.append('|---|---|' + '---|' * len(methods))
    for r in obj_runs:
        n, res, _ = person_check(r, methods)
        L.append(f'| {r["name"]} | {n} | ' + ' | '.join(f'{res[m][1]}/{res[m][0]} ({fmt(res[m][2])})' for m in methods) + ' |')

    L.append('\n## 4. 시간 안정성 — 같은 track_id 연속 평가 사이 |Δd|/d (쌍 수, 중앙 / 75% / 90%)\n')
    L.append('연속 영상만. 실제 거리도 0.2초 사이에 변하므로 0 이 정답은 아니다. 방식끼리 상대 비교용.\n')
    L.append('| 영상 | 방식 | 쌍 | 중앙 | 75% | 90% |')
    L.append('|---|---|---|---|---|---|')
    for r in runs:
        if r['name'] not in continuous:
            continue
        for m, (n, p50, p75, p90) in temporal(r, methods).items():
            L.append(f'| {r["name"]} | {m} | {n} | {fmt(p50)} | {fmt(p75)} | {fmt(p90)} |')

    L.append('\n## 5. 의심 사례\n')
    L.append('`fused` 는 그 실행에서 판정에 쓴 값. `final_locked` 는 검출되지 않은 박스를 engine 이 '
             '구간 고정 초점 + 무효 성분 제외로 계산했다면 낼 값.\n')
    pm = methods + ['final_locked']
    L.append('| 사례 | 박스 | f_px | ' + ' | '.join(pm) + ' |')
    L.append('|---|---|---|' + '---|' * len(pm))
    for label, bbox, f_px, vals in probes:
        L.append(f'| {label} | {bbox} | {fmt(f_px, 0)} | ' + ' | '.join(fmt(vals.get(m), 1) for m in pm) + ' |')

    L.append('\n## 6. 판정 시뮬레이션 — 객체 거리를 방식별 값으로 바꿔 경고 단계만 다시 계산\n')
    L.append('`fused` 행은 실제 frames.jsonl 과 같아야 한다 (재현 확인). LDW 는 거리와 무관해 실제 값을 그대로 쓴다.\n')
    L.append('| 영상 | 방식 | 등급 (안전/주의/위험/치명) | FCW L1/L2/L3 | BSW L1/L2/L3 | HEAD L2/L3 | 실제와 등급 다른 프레임 |')
    L.append('|---|---|---|---|---|---|---|')
    for r in obj_runs:
        for m in SIM_METHODS:
            s = simulate(r, m)
            diff = sum(1 for f, x in zip(r['frames'], s['sims']) if x['bin'] != f['bin'])
            g = lambda c, ks: '/'.join(str(c.get(k, 0)) for k in ks)
            L.append(f'| {r["name"]} | {m} | {g(s["bin"], [0, 1, 2, 3])} | {g(s["FCW"], [1, 2, 3])} '
                     f'| {g(s["BSW"], [1, 2, 3])} | {g(s["HEAD"], [2, 3])} | {diff} |')

    L.append('\n### 6-1. 연속 영상에서 위험 이상 (L2+) 이 나오는 시각 (초, 괄호는 등급)\n')
    L.append('| 영상 | 방식 | 시각 |')
    L.append('|---|---|---|')
    for r in obj_runs:
        if r['name'] not in continuous:
            continue
        for m in SIM_METHODS:
            s = simulate(r, m)
            ts = [f'{f["t_sec"]:.1f}({x["bin"]})' for f, x in zip(r['frames'], s['sims']) if x['bin'] >= 2]
            L.append(f'| {r["name"]} | {m} | {", ".join(ts) or "—"} |')
    if baselines:
        L.extend(baseline_section(runs, baselines, continuous))
    out_path.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs_out' / 'distance_eval.md')
    ap.add_argument('--continuous', nargs='*', default=['accident_0029_v6_hud'])
    ap.add_argument('--probe', nargs=5, action='append', default=[],
                    metavar=('VIDEO', 'FRAME', 'BBOX', 'CLS', 'LABEL'))
    ap.add_argument('--case', nargs=3, action='append', default=[],
                    metavar=('RUN', 'FRAME', 'LABEL'),
                    help='estimates.jsonl 안의 사례 (그 프레임 객체 전부)')
    ap.add_argument('--baseline', nargs='*', default=[], type=Path,
                    help='비교 기준 실행 폴더 (frames.jsonl + estimates.jsonl + meta.json). 이름으로 짝지음')
    ap.add_argument('--title', default='거리 추정 방식 비교')
    args = ap.parse_args(argv)

    runs = [load_run(p) for p in args.runs]
    by_name = {r['name']: r for r in runs}
    probes = []
    for run_name, frame, label in args.case:
        r = by_name[run_name]
        for f in r['frames']:
            if f['frame_idx'] == int(frame):
                for o in f['objects']:
                    probes.append((f'{label} ({o["cls"]})', o['bbox'], f['f_px'], o))
    for video, frame, bbox, cls, label in args.probe:
        bb = tuple(int(v) for v in bbox.split(','))
        r = by_name.get(Path(video).stem)
        vals = probe(video, int(frame), bb, cls, f_med=r['f_med'] if r else None,
                     f_fixed=segment_f_fixed(r, int(frame)) if r else None)
        if vals:
            probes.append((label + ' (검출 없음 → 박스 직접 측정)', list(bb), vals['f_px'], vals))
    baselines = {}
    for p in args.baseline:
        b = load_run(p)
        b['records'] = [json.loads(l) for l in open(Path(p) / 'frames.jsonl', encoding='utf-8')]
        baselines[b['name']] = b
    for r in runs:
        r['records'] = [json.loads(l) for l in open(Path(r['dir']) / 'frames.jsonl', encoding='utf-8')]
    write_report(runs, set(args.continuous), probes, args.out, baselines, title=args.title)
    print(f'보고서: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
