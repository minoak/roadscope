"""t36 증류 정답 만들기 — 엔진 사전 점검 (engine.calibrate_car) 의 내 차 영역 마스크를 영상마다 구해 프레임에 붙인다.

엔진 코드를 그대로 import 해서 쓴다 (다시 짜지 않음):
  - engine.calibrate_car → 구간별 CarCalib.interior_mask (Mask2Former 5장 투표 ∪ 보닛 띠 ∪ 테두리 정지 영역 ∪ 반복 검출 박스)
  - 바꾸는 것은 세 가지뿐이고 마스크에는 영향이 없다:
      * engine.load_depth_pro 를 (None, None) 으로 — Depth Pro 는 사전 점검에 쓰이지 않는다 (GPU·시간 절약)
      * car_models.load_mask2former 를 한 번만 올리도록 — 같은 모델, 영상마다 다시 올리는 시간만 줄임
      * car_models.m2f_ego_mask 를 감싸 표본 프레임별 Mask2Former 마스크를 기록 — 투표 안정도 (품질 표시) 계산용
카메라가 고정이라 구간 하나에 마스크 하나. 영상마다 FRAMES_PER_VIDEO 장을 고르게 뽑아 (장면 전환 근처 제외) 그 구간 마스크를 붙인다.

    python tools/distill/make_labels.py --set all     # 시험 30 (runs_in/nexar, 학습에 안 씀) 다음 학습 270 (manifest)
    python tools/distill/make_labels.py --merge       # → runs_out/distill/labels.csv, labels.md
결과: runs_out/distill/data/{frames,masks,meta}/. 이미 끝난 영상 (meta json 있음) 은 건너뛴다.
여러 프로세스를 같이 띄우면 영상마다 data/claim/<id> 파일을 먼저 만든 프로세스가 맡는다 (GPU 8GB 에 3개, 영상당 1.7GB).
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

OUT = ROOT / 'runs_out' / 'distill'
DATA = OUT / 'data'
FRAMES_PER_VIDEO = 12
SAVE_W, SAVE_H = 640, 360          # 저장 해상도 (원본 1280×720 의 1/2). 학습은 512×288 로 줄여 쓴다
EDGE_SKIP_SEC = 0.5                # 영상 처음·끝은 뽑지 않는다
CUT_SKIP_SEC = 0.3                 # 장면 전환 앞뒤
OUTLIERS = {'pos_00905', 'neg_01960', 'pos_00754'}   # 시험 세트 이상치 (결과에 따로 표시)

# 제외·표시 기준 (보고서 labels.md 에 그대로 적힌다)
EXCLUDE_MIN_M2F_FRAMES = 3         # 첫 구간 Mask2Former 표본이 이보다 적으면 제외 (짧은 구간·로드 실패)
EXCLUDE_MIN_M2F_AGREE = 0.75       # 표본별 Mask2Former 마스크와 투표 결과의 IoU 평균이 이보다 낮으면 제외 (선생이 흔들림)
EXCLUDE_FRAC = (0.02, 0.70)        # 마스크 면적 비율이 이 밖이면 제외 (내 차가 거의 없거나 화면 대부분 — 보정 이상)


def video_list(which):
    if which == 'train':
        rows = list(csv.DictReader(open(ROOT / 'runs_in/distill/manifest.csv', encoding='utf-8')))
        out = [{'id': ('pos_' if r['split'] == 'positive' else 'neg_') + Path(r['file_name']).stem,
                'path': r['path'], 'light': r['light'], 'scene': r['scene'], 'set': 'train'} for r in rows]
    else:
        rows = list(csv.DictReader(open(ROOT / 'runs_in/nexar/labels.csv', encoding='utf-8')))
        out = [{'id': Path(r['file']).stem, 'path': str(ROOT / 'runs_in/nexar' / r['file']), 'light': r['light'],
                'scene': r['scene'], 'set': 'test'} for r in rows]
    return out


def iou(a, b, min_union=0.0):
    """합집합이 화면의 min_union 비율보다 작으면 (둘 다 거의 빈 마스크) 일치로 본다."""
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u > min_union * a.size else 1.0


def setup_engine():
    import torch
    from roadscope import engine as E
    from roadscope import car_models
    E.VEHICLE = 'car'
    E.load_depth_pro = lambda device: (None, None)          # 사전 점검에 쓰지 않는 모델
    # Mask2Former 는 영상마다 다시 올리지 않고 한 번 올린 것을 돌려준다 (같은 가중치, 로드 시간만 줄임)
    m2f_cache = {}
    orig_load = car_models.load_mask2former

    def load_once(device):
        if device not in m2f_cache:
            m2f_cache[device] = orig_load(device)
        return m2f_cache[device]
    car_models.load_mask2former = load_once
    calls = []
    orig = car_models.m2f_ego_mask

    def recording(m2f, bgr, device):
        m = orig(m2f, bgr, device)
        calls.append(m)
        return m
    car_models.m2f_ego_mask = recording
    eng = E.Engine(fps_eval=E.CAR_FPS_DEFAULT, log=lambda *a: None)
    return E, car_models, eng, calls, torch


def label_video(v, E, cm, eng, calls):
    (DATA / 'frames').mkdir(parents=True, exist_ok=True)
    (DATA / 'masks').mkdir(parents=True, exist_ok=True)
    (DATA / 'meta').mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(v['path'])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    calls.clear()
    t0 = time.time()
    cuts, calibs, cache, n_eval, seg_info = E.calibrate_car(eng, v['path'], fps, E.CAR_FPS_DEFAULT, W, H,
                                                            log=lambda *a: None)
    del cache
    t_cal = time.time() - t0
    segs = []
    for k, (si, cal) in enumerate(zip(seg_info, calibs)):
        m = cv2.resize(cal.interior_mask.astype(np.uint8) * 255, (SAVE_W, SAVE_H), interpolation=cv2.INTER_NEAREST)
        name = f"{v['id']}_seg{k}.png"
        cv2.imwrite(str(DATA / 'masks' / name), m)
        ii = cal.interior_info
        segs.append({'segment': k, 'start_frame': si['start_frame'], 'end_frame': si['end_frame'], 'mask': name,
                     'frac': ii['frac'], 'status': ii['status'], 'm2f': ii['m2f'], 'n_moving_pairs': ii['n_moving_pairs'],
                     'hood_src': cal.hood_src, 'hood_row': int(cal.hood_row), 'horizon_conf': cal.horizon_conf,
                     'n_static_boxes': sum(1 for b in ii.get('static_boxes', []) if 'rejected' not in b)})
    # 첫 구간 Mask2Former 투표 안정도: 표본 마스크 (기록) 와 투표 결과 (car_models.vote_ego) 의 IoU 평균.
    # 구간이 여럿이면 기록된 표본을 구간별로 나눌 수 없어 첫 구간 표본 수 (≤ 5) 만큼만 쓴다
    n0 = segs[0]['m2f'].get('n_frames', 0)
    agree = None
    if n0:
        first = calls[:n0]
        vote = cm.vote_ego(first)
        if vote is not None:
            agree = float(np.mean([iou(c, vote, 0.005) for c in first]))
    # 프레임 고르기
    cut_set = sorted(cuts)
    dur = n_src / fps
    picks = []
    for t in np.linspace(EDGE_SKIP_SEC, max(EDGE_SKIP_SEC, dur - EDGE_SKIP_SEC), FRAMES_PER_VIDEO):
        fi = int(round(t * fps))
        if any(abs(fi - c) <= CUT_SKIP_SEC * fps for c in cut_set):
            continue
        seg = sum(1 for c in cut_set if c <= fi)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im = cap.read()
        if not ok:
            continue
        fname = f"{v['id']}_{fi:05d}.jpg"
        cv2.imwrite(str(DATA / 'frames' / fname), cv2.resize(im, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        picks.append({'frame': fname, 'frame_idx': fi, 't_sec': round(fi / fps, 3), 'segment': seg, 'mask': segs[seg]['mask']})
    cap.release()
    s0 = segs[0]
    reasons = []
    if s0['m2f'].get('status') != 'ok' or n0 < EXCLUDE_MIN_M2F_FRAMES:
        reasons.append(f"m2f_frames<{EXCLUDE_MIN_M2F_FRAMES}")
    if agree is not None and agree < EXCLUDE_MIN_M2F_AGREE:
        reasons.append(f'm2f_agree<{EXCLUDE_MIN_M2F_AGREE}')
    if not (EXCLUDE_FRAC[0] <= s0['frac'] <= EXCLUDE_FRAC[1]):
        reasons.append('mask_frac_out_of_range')
    flags = []
    if s0['status'] == 'no_motion':
        flags.append('no_motion')          # 움직임 기반 (정지 영역·반복 검출) 이 빠진 마스크
    if s0['hood_src'] == 'default':
        flags.append('hood_default')
    if len(segs) > 1:
        flags.append('multi_segment')
    rec = {**v, 'W': W, 'H': H, 'fps': fps, 'n_src': n_src, 'calib_sec': round(t_cal, 1), 'segments': segs,
           'm2f_agree': None if agree is None else round(agree, 4), 'frames': picks,
           'exclude': reasons, 'flags': flags, 'outlier': v['id'] in OUTLIERS}
    (DATA / 'meta' / f"{v['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    return rec


def merge():
    recs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((DATA / 'meta').glob('*.json'))]
    with open(OUT / 'labels.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'set', 'light', 'scene', 'frame', 'mask', 'segment', 't_sec', 'excluded', 'flags', 'outlier'])
        for r in recs:
            for p in r['frames']:
                w.writerow([r['id'], r['set'], r['light'], r['scene'], p['frame'], p['mask'], p['segment'], p['t_sec'],
                            ';'.join(r['exclude']), ';'.join(r['flags']), int(r['outlier'])])
    lines = ['# t36 증류 정답 (엔진 사전 점검 내 차 영역)', '',
             f'- 영상마다 engine.calibrate_car 로 구간별 내 차 마스크 → 프레임 {FRAMES_PER_VIDEO}장 (처음·끝 {EDGE_SKIP_SEC}초, '
             f'장면 전환 ±{CUT_SKIP_SEC}초 제외) 에 붙임. 저장 {SAVE_W}×{SAVE_H}',
             f'- 제외 기준: 첫 구간 Mask2Former 표본 {EXCLUDE_MIN_M2F_FRAMES}장 미만 또는 로드 실패, 표본별 마스크와 투표 결과의 '
             f'IoU 평균 {EXCLUDE_MIN_M2F_AGREE} 미만 (선생 흔들림), 마스크 면적 비율 {EXCLUDE_FRAC[0]}~{EXCLUDE_FRAC[1]} 밖',
             '- 표시 (학습에 씀): no_motion = 내 차 움직임 판정 실패라 정지 영역·반복 검출이 빠짐, hood_default = 보닛 행 추정 실패, '
             'multi_segment = 장면 전환 있음', '']
    for s in ('train', 'test'):
        rs = [r for r in recs if r['set'] == s]
        if not rs:
            continue
        ex = [r for r in rs if r['exclude']]
        from collections import Counter
        why = Counter(x for r in ex for x in r['exclude'])
        fl = Counter(x for r in rs if not r['exclude'] for x in r['flags'])
        ag = [r['m2f_agree'] for r in rs if r['m2f_agree'] is not None]
        lines += [f'## {s} ({len(rs)}개 영상)', '',
                  f"- 제외 {len(ex)}개: {dict(why)}" + (f" — {', '.join(r['id'] for r in ex)}" if len(ex) <= 20 else ''),
                  f"- 남은 {len(rs) - len(ex)}개, 프레임 {sum(len(r['frames']) for r in rs if not r['exclude'])}장, 표시 {dict(fl)}",
                  f"- Mask2Former 투표 안정도 (IoU 평균) 중앙값 {np.median(ag):.3f}, 10% {np.percentile(ag, 10):.3f}" if ag else '',
                  f"- 마스크 면적 비율 중앙값 {np.median([r['segments'][0]['frac'] for r in rs]) * 100:.1f}%, "
                  f"사전 점검 시간 중앙값 {np.median([r['calib_sec'] for r in rs]):.0f}초", '']
    (OUT / 'labels.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.merge:
        merge()
        return
    test_ids = {v['id'] for v in video_list('test')}
    train = video_list('train')
    assert not test_ids & {v['id'] for v in train}, test_ids & {v['id'] for v in train}   # 시험 30개와 겹치지 않는지
    vids = {'train': train, 'test': video_list('test'), 'all': video_list('test') + train}[args.set]
    if args.limit:
        vids = vids[:args.limit]
    E, cm, eng, calls, torch = setup_engine()
    (DATA / 'claim').mkdir(parents=True, exist_ok=True)
    log = open(OUT / f'make_labels_{os.getpid()}.log', 'a', encoding='utf-8')
    for k, v in enumerate(vids):
        if (DATA / 'meta' / f"{v['id']}.json").exists():
            continue
        try:
            open(DATA / 'claim' / v['id'], 'x').close()
        except FileExistsError:     # 다른 프로세스가 맡음
            continue
        t0 = time.time()
        try:
            r = label_video(v, E, cm, eng, calls)
            msg = (f"{v['id']} {time.time() - t0:.0f}s frac {r['segments'][0]['frac']:.3f} agree {r['m2f_agree']} "
                   f"exclude {r['exclude']} flags {r['flags']}")
        except Exception:
            msg = f"{v['id']} ERROR {traceback.format_exc()[-400:]}"
        print(f'[{k + 1}/{len(vids)}] {msg}', flush=True)
        log.write(msg + '\n'); log.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
