"""t36 보조 평가: 선생 마스크를 '보닛 띠' (선생 보닛 행 아래) 와 '그 위 실내' (A필러·룸미러·실내 물체·워터마크 등) 로
나눠 부분별로 학생이 얼마나 잡는지 (재현율) 와, 학생이 선생 밖에 더 칠한 면적이 어디에 있는지를 잰다.
IoU 하나로는 '어느 부분을 놓치는지' 가 안 보여서 (확인 그림 판독 결과를 숫자로 확인).

    python tools/distill/eval_parts.py   → runs_out/distill/eval_parts.md
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ego as T
import eval_ego as EV

OUT, DATA = T.OUT, T.DATA


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ck = torch.load(OUT / 'ego_unet_r34.pt', map_location='cpu', weights_only=True)
    model = T.build_model(pretrained=False)
    model.load_state_dict(ck['state_dict'])
    model.eval().to('cuda')
    split = json.loads((OUT / 'split.json').read_text(encoding='utf-8'))
    lines = ['# t36 부분별 평가 (선생 보닛 띠 / 그 위 실내)', '',
             '- 보닛 띠 = 선생 보닛 행 (meta segments[].hood_row) 아래 행. 그 위 실내 = 보닛 행 − 8px 위의 선생 마스크 '
             '(A필러·룸미러·실내 물체·워터마크·반복 검출 박스). 640×360, 프레임 픽셀 합산',
             '- 재현율 = 학생이 잡은 선생 픽셀 / 선생 픽셀. 넘침 = 학생만 칠한 픽셀 / 화면 (보닛 행 위 ±0.1H 띠 / 그 밖)', '',
             '| 세트 | 영상 | 보닛 띠 재현율 | 그 위 실내 재현율 (선생 면적 비율) | 넘침: 보닛 행 바로 위 0.1H | 넘침: 그 밖 |',
             '|---|---|---|---|---|---|']
    per = []
    for name, which, ids in (('val', 'train', set(split['val'])), ('test', 'test', None)):
        recs = [r for r in T.load_meta(which) if (ids is None or r['id'] in ids)
                and not r['exclude'] and not r['outlier']]
        acc = np.zeros(8)
        for r in recs:
            a = np.zeros(8)
            for p in r['frames']:
                seg = r['segments'][p['segment']]
                hr = int(round(seg['hood_row'] * EV.SAVE_H / r['H']))
                im = cv2.imread(str(DATA / 'frames' / p['frame']))
                mt = cv2.imread(str(DATA / 'masks' / p['mask']), 0) > 127
                ms = EV.student(model, im, 'cuda') > 0.5
                rows = np.arange(EV.SAVE_H)[:, None]
                hood = rows >= hr
                upper = rows < hr - 8
                near = (rows >= hr - int(0.1 * EV.SAVE_H)) & (rows < hr)
                a += [(mt & hood).sum(), (mt & ms & hood).sum(), (mt & upper).sum(), (mt & ms & upper).sum(),
                      (ms & ~mt & near).sum(), (ms & ~mt & ~near).sum(), mt.size, 1]
            acc += a
            per.append((name, r['id'], a[3] / a[2] if a[2] else None, a[2] / (a[6])))
        f = lambda x, y: f'{x / y:.3f}' if y else '-'
        lines.append(f"| {name} | {len(recs)} | {f(acc[1], acc[0])} | {f(acc[3], acc[2])} ({acc[2] / acc[6] * 100:.1f}%) | "
                     f"{acc[4] / acc[6] * 100:.2f}% | {acc[5] / acc[6] * 100:.2f}% |")
    lines += ['', '## 시험 영상별 그 위 실내 재현율 (선생 면적이 화면 0.5% 이상인 영상)', '',
              '| 영상 | 재현율 | 선생 면적 |', '|---|---|---|']
    for name, vid, rec, frac in sorted([p for p in per if p[0] == 'test' and p[3] >= 0.005], key=lambda p: p[2]):
        lines.append(f'| {vid} | {rec:.3f} | {frac * 100:.1f}% |')
    (OUT / 'eval_parts.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
