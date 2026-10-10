"""t36 내 차 영역 학생 모델 학습 — smp U-Net ResNet34, 프레임 한 장 → 내 차 마스크 (이진).

정답 = make_labels.py 가 엔진 사전 점검으로 만든 영상별 마스크 (runs_out/distill/data/meta/*.json, set == 'train').
  - 나누기: 영상 단위 240 / 30 (밝기 비율 맞춤, seed 고정). 같은 영상의 프레임은 한쪽에만 → 프레임 누출 없음.
    정답 제외 영상 (meta exclude) 은 나눈 뒤 양쪽에서 뺀다. 시험 30개 (runs_in/nexar) 는 여기서 읽지 않는다.
  - 입력 512×288: 원본 16:9 를 유지하면서 32 배수 (U-Net 다운샘플 5단). 내 차 경계 (보닛선·필러) 는 큰 구조라
    1280 의 2/5 해상도로 충분하고, CPU 에서도 돌릴 수 있는 크기.
  - 증강: 밝기·대비·감마·색 (낮/밤·노출 차이), 약한 이동·확대 (±4%, 0.93~1.07 — 카메라 거치 위치 차이),
    좌우 뒤집기 0.5 (내 차 여부는 좌우와 무관. 카메라 240대뿐이라 필러·거치대 모양 다양성을 늘림), 약한 흐림·JPEG.
    큰 회전·자르기는 안 함 — 내 차는 늘 화면 가장자리에 붙어 있다는 위치 단서를 지키려고.
  - 손실: BCE + Dice. 내 차는 화면의 10~20% 라 BCE 만 쓰면 배경 쪽으로 쏠리고, Dice 는 면적 비율과 무관하게 겹침을 본다.
  - AdamW 3e-4 (encoder 1e-4, 사전학습 특징을 덜 흔들게), 처음 5% warmup 뒤 cosine 감소 (OneCycle), 40 epoch, batch 16, AMP.
    영상 240개 × 12장 ≈ 2,900장이라 epoch 당 180 step, 40 epoch ≈ 7,000 step. val 영상별 IoU 평균이 가장 높은 epoch 를 저장.
  - 사전학습 encoder: ImageNet ResNet34 (smp 0.5 기본 'imagenet' = HF smp-hub/resnet34.imagenet, 로컬 캐시에서 오프라인 로드).

    python tools/distill/train_ego.py [--epochs 40]
결과: runs_out/distill/ego_unet_r34.pt, split.json, train_log.csv, learning_curve.png
"""
from __future__ import annotations
import argparse, csv, json, os, random, sys, time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs_out' / 'distill'
DATA = OUT / 'data'
IN_W, IN_H = 512, 288
SEED = 36
N_VAL = 30
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_meta(which='train'):
    return [r for r in (json.loads(p.read_text(encoding='utf-8')) for p in sorted((DATA / 'meta').glob('*.json')))
            if r['set'] == which]


def split_videos(recs):
    """영상 단위로 val N_VAL 개를 밝기 비율대로 뽑는다 (제외 여부와 무관하게 먼저 나눈 뒤 제외)."""
    rng = random.Random(SEED)
    by = {}
    for r in recs:
        by.setdefault(r['light'], []).append(r['id'])
    val = []
    for light, ids in sorted(by.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        val += ids[:round(N_VAL * len(ids) / len(recs))]
    rest = sorted(set(r['id'] for r in recs) - set(val))
    rng.shuffle(rest)
    while len(val) < N_VAL:
        val.append(rest.pop())
    val = val[:N_VAL]          # 밝기별 반올림으로 넘친 몫은 학습으로 (어느 쪽에도 안 들어가는 영상이 없게)
    return sorted(set(r['id'] for r in recs) - set(val)), sorted(val)


def load_pairs(recs, ids):
    """(영상 id, 프레임 이름, 이미지 uint8 IN_H×IN_W×3 BGR, 마스크 uint8 0/1)"""
    ids = set(ids)
    out = []
    for r in recs:
        if r['id'] not in ids or r['exclude']:
            continue
        masks = {}
        for p in r['frames']:
            if p['mask'] not in masks:
                m = cv2.imread(str(DATA / 'masks' / p['mask']), cv2.IMREAD_GRAYSCALE)
                masks[p['mask']] = (cv2.resize(m, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
            im = cv2.resize(cv2.imread(str(DATA / 'frames' / p['frame'])), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
            out.append((r['id'], p['frame'], im, masks[p['mask']]))
    return out


def to_tensor(im):
    return torch.from_numpy(((im[..., ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.93, 1.07), translate_percent=(-0.04, 0.04), rotate=(-2, 2), border_mode=cv2.BORDER_REPLICATE, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.RandomGamma(gamma_limit=(60, 160), p=0.4),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=0, p=0.4),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.ImageCompression(quality_range=(40, 90))], p=0.3),
])


def build_model(pretrained=True):
    return smp.Unet('resnet34', encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=1)


def per_video_iou(vids, preds, gts):
    """프레임별 IoU → 영상별 평균. preds/gts: bool 배열 리스트. 합집합이 0 이면 1."""
    acc = {}
    for v, p, g in zip(vids, preds, gts):
        u = np.logical_or(p, g).sum()
        acc.setdefault(v, []).append(np.logical_and(p, g).sum() / u if u else 1.0)
    return {v: float(np.mean(x)) for v, x in acc.items()}


@torch.no_grad()
def predict(model, ims, device, bs=32):
    model.eval()
    out = []
    for k in range(0, len(ims), bs):
        x = torch.stack([to_tensor(im) for im in ims[k:k + bs]]).to(device)
        with torch.autocast('cuda', enabled=device == 'cuda'):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy()[:, 0])
    return np.concatenate(out) if out else np.zeros((0, IN_H, IN_W), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=16)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda'
    recs = load_meta('train')
    tr_ids, va_ids = split_videos(recs)
    excl = sorted(r['id'] for r in recs if r['exclude'])
    (OUT / 'split.json').write_text(json.dumps({'seed': SEED, 'train': tr_ids, 'val': va_ids, 'excluded': excl},
                                               ensure_ascii=False, indent=1), encoding='utf-8')
    tr, va = load_pairs(recs, tr_ids), load_pairs(recs, va_ids)
    assert not {p[0] for p in tr} & {p[0] for p in va}
    print(f'영상 train {len(tr_ids)} / val {len(va_ids)} (제외 {len(excl)}) → 프레임 train {len(tr)} / val {len(va)}', flush=True)
    model = build_model().to(device)
    opt = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 1e-4},
                             {'params': list(model.decoder.parameters()) + list(model.segmentation_head.parameters()),
                              'lr': 3e-4}], weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.bs)          # 마지막 덜 찬 배치는 버리므로 내림
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 3e-4], total_steps=steps, pct_start=0.05,
                                                anneal_strategy='cos', div_factor=10, final_div_factor=100)
    dice = smp.losses.DiceLoss('binary', from_logits=True)
    scaler = torch.amp.GradScaler('cuda')
    va_x = torch.stack([to_tensor(p[2]) for p in va])
    va_y = torch.from_numpy(np.stack([p[3] for p in va])).float()[:, None]
    log_rows, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = np.random.permutation(len(tr))
        tl, nb = 0.0, 0
        for k in range(0, len(order) - args.bs + 1, args.bs):   # 마지막 덜 찬 배치는 버림 (BatchNorm 안정)
            xs, ys = [], []
            for j in order[k:k + args.bs]:
                a = AUG(image=tr[j][2], mask=tr[j][3])
                xs.append(to_tensor(a['image'])); ys.append(torch.from_numpy(a['mask']).float()[None])
            x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
            with torch.autocast('cuda'):
                lo = model(x)
            lo = lo.float()
            loss = F.binary_cross_entropy_with_logits(lo, y) + dice(lo, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            tl += loss.item(); nb += 1
        # val
        model.eval()
        vl, probs = 0.0, []
        with torch.no_grad():
            for k in range(0, len(va), 32):
                x, y = va_x[k:k + 32].to(device), va_y[k:k + 32].to(device)
                with torch.autocast('cuda'):
                    lo = model(x)
                lo = lo.float()
                vl += (F.binary_cross_entropy_with_logits(lo, y) + dice(lo, y)).item() * len(x)
                probs.append(torch.sigmoid(lo).cpu().numpy()[:, 0])
        probs = np.concatenate(probs)
        ious = per_video_iou([p[0] for p in va], list(probs > 0.5), [p[3] > 0 for p in va])
        vi = np.array(list(ious.values()))
        row = {'epoch': ep, 'train_loss': round(tl / nb, 5), 'val_loss': round(vl / len(va), 5),
               'val_iou_mean': round(float(vi.mean()), 4), 'val_iou_median': round(float(np.median(vi)), 4),
               'val_iou_min': round(float(vi.min()), 4), 'lr_dec': round(opt.param_groups[1]['lr'], 7),
               'sec': round(time.time() - t0, 1)}
        log_rows.append(row)
        if row['val_iou_mean'] > best:
            best = row['val_iou_mean']
            torch.save({'state_dict': model.state_dict(), 'arch': 'smp.Unet resnet34', 'in_wh': (IN_W, IN_H),
                        'epoch': ep, 'val_iou_mean': best, 'val_iou_median': row['val_iou_median'],
                        'encoder_pretrained': 'ImageNet (HF smp-hub/resnet34.imagenet 스냅샷 7a57b34, 로컬 캐시. '
                                              'torchvision resnet34-b627a593 과 같은 값 확인, smp 0.5.0)',
                        'normalize': 'RGB, ImageNet mean/std'}, OUT / 'ego_unet_r34.pt')
        print(f"ep {ep:2d} train {row['train_loss']:.4f} val {row['val_loss']:.4f} IoU 평균 {row['val_iou_mean']:.4f} "
              f"중앙 {row['val_iou_median']:.4f} 최저 {row['val_iou_min']:.4f} {row['sec']}s{' *' if best == row['val_iou_mean'] else ''}",
              flush=True)
        with open(OUT / 'train_log.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader(); w.writerows(log_rows)
    plot(log_rows)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r['train_loss'] for r in rows], label='train loss (BCE+Dice)')
    ax[0].plot(ep, [r['val_loss'] for r in rows], label='val loss')
    ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [r['val_iou_mean'] for r in rows], label='val IoU mean (per video)')
    ax[1].plot(ep, [r['val_iou_median'] for r in rows], label='val IoU median')
    ax[1].plot(ep, [r['val_iou_min'] for r in rows], label='val IoU min', alpha=.6)
    b = max(rows, key=lambda r: r['val_iou_mean'])
    ax[1].axvline(b['epoch'], color='gray', ls='--', lw=1)
    ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle('t36 ego-mask student (U-Net R34, 512x288) vs engine pre-check teacher')
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curve.png', dpi=110)


if __name__ == '__main__':
    main()
