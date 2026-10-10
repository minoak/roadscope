"""t36 학생 평가 — 선생 (엔진 사전 점검 마스크) 대비 IoU, 처리 시간, 확인 그림.

  - val (학습 때 뺀 30개 영상) 과 시험 (runs_in/nexar 30개, 학습·검증에 안 쓴 영상) 각각
    프레임별 IoU → 영상별 평균 → 전체 중앙값·평균, 밝기별 (낮 = Normal/Bright, 어두움 = Twilight/Dark).
    시험 이상치 3개 (pos_00905, neg_01960, pos_00754) 와 선생 제외 영상은 본 집계에서 빼고 따로 적는다.
  - IoU 는 저장 해상도 640×360 에서 잰다 (학생 출력 512×288 확률을 640×360 으로 늘려 0.5 로 자름).
  - 시간: 이미지 한 장 (batch 1) — 전처리 (640×360 BGR → 512×288 정규화) + 모델 + 출력 확대·자르기.
    GPU (fp16 autocast) 와 CPU (fp32, torch 스레드 수 기록) 각각 warmup 뒤 반복 측정.
  - 확인 그림: 시험에서 IoU 가장 낮은 5개 + neg_01711 + pos_00413 → runs_out/distill/check/<영상>.jpg
    (프레임 3장: 원본 | 선생 | 학생. 초록 = 둘 다, 빨강 = 선생만, 파랑 = 학생만)

    python tools/distill/eval_ego.py
결과: runs_out/distill/eval.md, eval_per_video.csv, check/*.jpg
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ego as T

OUT, DATA = T.OUT, T.DATA
SAVE_W, SAVE_H = 640, 360
OUTLIERS = ['pos_00905', 'neg_01960', 'pos_00754']
MUST_CHECK = ['neg_01711', 'pos_00413']
DAY = {'Normal', 'Bright'}


def load_set(recs, ids=None, keep_excluded=True):
    """(영상, 프레임 이름, 640×360 BGR, 선생 마스크 bool 640×360)"""
    out = []
    for r in recs:
        if ids is not None and r['id'] not in ids:
            continue
        if r['exclude'] and not keep_excluded:
            continue
        for p in r['frames']:
            im = cv2.imread(str(DATA / 'frames' / p['frame']))
            m = cv2.imread(str(DATA / 'masks' / p['mask']), cv2.IMREAD_GRAYSCALE) > 127
            out.append((r['id'], p['frame'], im, m))
    return out


def student(model, im640, device):
    """640×360 BGR → 640×360 확률 (전처리·후처리 포함, 시간 측정에도 같은 함수)."""
    x = T.to_tensor(cv2.resize(im640, (T.IN_W, T.IN_H), interpolation=cv2.INTER_AREA))[None].to(device)
    with torch.no_grad(), torch.autocast('cuda', enabled=device == 'cuda'):
        lo = model(x)
    p = F.interpolate(lo.float(), size=(SAVE_H, SAVE_W), mode='bilinear', align_corners=False)
    return torch.sigmoid(p)[0, 0].cpu().numpy()


def iou(p, g):
    u = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / u) if u else 1.0


def stats(vals):
    v = np.array(vals)
    return f'{np.median(v):.3f} / {v.mean():.3f} / {v.min():.3f} (n={len(v)})' if len(v) else '-'


def overlay(im, m_t, m_s):
    o = im.copy().astype(np.float32)
    col = np.zeros_like(o)
    col[m_t & m_s] = (0, 200, 0)
    col[m_t & ~m_s] = (0, 0, 255)
    col[~m_t & m_s] = (255, 80, 0)
    a = (m_t | m_s)[..., None] * 0.45
    return (o * (1 - a) + col * a).astype(np.uint8)


def contour(im, m, color):
    o = im.copy()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cs, -1, color, 2)
    return o


def check_image(vid, rows, rec, path):
    """영상의 프레임 3장 (IoU 최저·중간·최고) — 원본 | 선생 윤곽 | 학생 윤곽 | 겹침."""
    rows = sorted(rows, key=lambda r: r['iou'])
    pick = [rows[0], rows[len(rows) // 2], rows[-1]] if len(rows) >= 3 else rows
    lines = []
    for r in pick:
        im, mt, ms = r['im'], r['mt'], r['ms']
        tiles = [im, contour(im, mt, (0, 0, 255)), contour(im, ms, (255, 80, 0)), overlay(im, mt, ms)]
        tiles = [cv2.resize(t, (480, 270)) for t in tiles]
        for t, s in zip(tiles, ['frame', 'teacher', 'student', f"IoU {r['iou']:.3f}"]):
            cv2.putText(t, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        lines.append(np.hstack(tiles))
    head = np.zeros((34, lines[0].shape[1], 3), np.uint8)
    s0 = rec['segments'][0]
    txt = (f"{vid} ({rec['light']}) teacher frac {s0['frac']:.3f} {s0['status']} hood {s0['hood_src']} "
           f"m2f {s0['m2f'].get('frac', 0):.3f} boxes {s0['n_static_boxes']}  green=both red=teacher-only blue=student-only")
    cv2.putText(head, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head] + lines), [cv2.IMWRITE_JPEG_QUALITY, 88])


def timing(model, im, device, n):
    for _ in range(5):
        student(model, im, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        student(model, im, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - a)
    return float(np.median(ts) * 1000), float(np.percentile(ts, 90) * 1000)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ck = torch.load(OUT / 'ego_unet_r34.pt', map_location='cpu', weights_only=True)
    model = T.build_model(pretrained=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    device = 'cuda'
    model.to(device)
    split = json.loads((OUT / 'split.json').read_text(encoding='utf-8'))
    tr_recs = {r['id']: r for r in T.load_meta('train')}
    te_recs = {r['id']: r for r in T.load_meta('test')}
    res, per_video = {}, []
    for name, recs, ids in (('val', tr_recs, set(split['val'])), ('test', te_recs, None)):
        data = load_set(list(recs.values()), ids)
        rows = {}
        for vid, fr, im, mt in data:
            ms = student(model, im, device) > 0.5
            rows.setdefault(vid, []).append({'frame': fr, 'iou': iou(ms, mt), 'im': im, 'mt': mt, 'ms': ms,
                                             'frac_s': float(ms.mean()), 'frac_t': float(mt.mean())})
        res[name] = rows
        for vid, rs in rows.items():
            r = recs[vid]
            per_video.append({'set': name, 'video': vid, 'light': r['light'], 'iou': round(float(np.mean([x['iou'] for x in rs])), 4),
                              'iou_min_frame': round(min(x['iou'] for x in rs), 4),
                              'teacher_frac': round(r['segments'][0]['frac'], 4),
                              'student_frac': round(float(np.mean([x['frac_s'] for x in rs])), 4),
                              'teacher_status': r['segments'][0]['status'], 'hood_src': r['segments'][0]['hood_src'],
                              'teacher_excluded': ';'.join(r['exclude']), 'flags': ';'.join(r['flags']),
                              'outlier': int(vid in OUTLIERS)})
    with open(OUT / 'eval_per_video.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_video[0]))
        w.writeheader(); w.writerows(per_video)

    # 시간
    im = load_set([te_recs['neg_01711']])[0][2]
    g_med, g_p90 = timing(model, im, 'cuda', 200)
    model_cpu = T.build_model(pretrained=False)
    model_cpu.load_state_dict(ck['state_dict']); model_cpu.eval()
    c_med, c_p90 = timing(model_cpu, im, 'cpu', 30)
    n_thr = torch.get_num_threads()
    torch.set_num_threads(4)
    c4_med, c4_p90 = timing(model_cpu, im, 'cpu', 30)
    torch.set_num_threads(n_thr)

    # 확인 그림
    (OUT / 'check').mkdir(exist_ok=True)
    te_pv = sorted([p for p in per_video if p['set'] == 'test'], key=lambda p: p['iou'])
    worst = [p['video'] for p in te_pv[:5]]                     # 시험 30개 전체에서 최저 5개
    worst_main = [p['video'] for p in te_pv if not p['outlier'] and not p['teacher_excluded']][:5]   # 본 집계에서 최저 5개
    for vid in dict.fromkeys(worst + worst_main + MUST_CHECK):
        check_image(vid, res['test'][vid], te_recs[vid], OUT / 'check' / f'{vid}.jpg')

    # eval.md
    def block(name, recs):
        pv = [p for p in per_video if p['set'] == name]
        main_ = [p for p in pv if not p['outlier'] and not p['teacher_excluded']]
        L = [f"- 전체 (이상치·선생 제외 영상 뺌) 중앙값 / 평균 / 최저: {stats([p['iou'] for p in main_])}",
             f"- 낮 (Normal/Bright): {stats([p['iou'] for p in main_ if p['light'] in DAY])}",
             f"- 어두움 (Twilight/Dark): {stats([p['iou'] for p in main_ if p['light'] not in DAY])}",
             f"- 움직임 판정 실패 (no_motion) 선생: {stats([p['iou'] for p in main_ if p['teacher_status'] == 'no_motion'])}, "
             f"성공: {stats([p['iou'] for p in main_ if p['teacher_status'] != 'no_motion'])}"]
        ex = [p for p in pv if p['teacher_excluded'] and not p['outlier']]
        if ex:
            L.append('- 선생 제외 기준에 걸린 영상 (본 집계 밖): ' + ', '.join(f"{p['video']} {p['iou']:.3f} ({p['teacher_excluded']})" for p in ex))
        ol = [p for p in pv if p['outlier']]
        if ol:
            L.append('- 이상치 (본 집계 밖): ' + ', '.join(f"{p['video']} {p['iou']:.3f}" for p in ol))
        return L

    tl = list(csv.DictReader(open(OUT / 'train_log.csv', encoding='utf-8')))
    md = ['# t36 내 차 영역 증류 평가', '',
          f"- 학생: smp U-Net ResNet34, 입력 {T.IN_W}×{T.IN_H}, encoder 사전학습 {ck['encoder_pretrained']}. "
          f"저장 epoch {ck['epoch']} / {len(tl)} (val 영상별 IoU 평균 최고)",
          f"- 영상: train {len(split['train'])} / val {len(split['val'])} (정답 제외 {len(split['excluded'])}개는 양쪽에서 뺌), "
          f"시험 runs_in/nexar {len(te_recs)}개 (학습·검증에 안 씀). 영상마다 프레임 12장",
          '- IoU = 선생 (엔진 사전 점검 마스크) 대비, 640×360 에서 프레임별 → 영상별 평균. 표기: 중앙값 / 평균 / 최저 (영상 수)', '',
          '## val (학습 영상과 다른 Nexar 영상)', ''] + block('val', tr_recs) + [
          '', '## 시험 (runs_in/nexar 30)', ''] + block('test', te_recs) + [
          '', '## 처리 시간 (이미지 한 장, 전처리·후처리 포함)', '',
          f'- GPU (RTX, fp16 autocast, batch 1): 중앙 {g_med:.1f} ms, 90% {g_p90:.1f} ms',
          f'- CPU (fp32, torch 스레드 {n_thr}): 중앙 {c_med:.0f} ms, 90% {c_p90:.0f} ms / 스레드 4: 중앙 {c4_med:.0f} ms, 90% {c4_p90:.0f} ms',
          '- 참고: 선생 = 영상 전체를 훑는 사전 점검 (Mask2Former 5장 + 광류 + YOLO 반복 검출), 영상당 수십 초',
          '', '## 시험 영상별', '', '| 영상 | 밝기 | IoU | 최저 프레임 | 선생 면적 | 학생 면적 | 선생 상태 | 보닛 출처 | 비고 |',
          '|---|---|---|---|---|---|---|---|---|']
    for p in te_pv:
        md.append(f"| {p['video']} | {p['light']} | {p['iou']:.3f} | {p['iou_min_frame']:.3f} | {p['teacher_frac']:.3f} | "
                  f"{p['student_frac']:.3f} | {p['teacher_status']} | {p['hood_src']} | "
                  f"{'이상치 ' if p['outlier'] else ''}{p['teacher_excluded']} {p['flags']} |")
    md += ['', f"## 확인 그림 (runs_out/distill/check/)", '',
           f"- IoU 최저 5개 (30개 전체): {', '.join(worst)} / 본 집계 (이상치·선생 제외 영상 뺌) 최저 5개: {', '.join(worst_main)}"
           f" + 지정 {', '.join(MUST_CHECK)}",
           '- 그림마다 프레임 3장 (그 영상에서 IoU 최저·중간·최고): 원본 | 선생 윤곽 (빨강) | 학생 윤곽 (파랑) | 겹침 '
           '(초록 = 둘 다, 빨강 = 선생만, 파랑 = 학생만)']
    (OUT / 'eval.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
