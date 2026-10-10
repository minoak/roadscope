"""frames_to_video.py — 데이터셋 원본 이미지들을 번호 순으로 이어 붙여 임시 mp4 를 만든다.

원본 블랙박스 영상이 없던 초기에 엔진 테스트용으로 썼다.
`night ride (N).png` 만 사용하고 `___fuse`, `___save` 가 붙은 파생 파일은 제외한다.

사용:
    python tools/render/frames_to_video.py <이미지폴더> <출력.mp4> [--fps 10] [--repeat 2] [--limit 0]

    --repeat: 이미지 1장을 몇 프레임 동안 보여줄지. 기본 fps 10 × repeat 2 로 만들면
              python -m roadscope.engine --fps 5 가 이미지마다 정확히 1번씩 평가한다 (샘플링 로직 확인용).
    --limit:  앞에서부터 N장만 사용 (0 = 전부).

출력은 H.264 (yuv420p, faststart) — 브라우저에서 바로 재생된다 (imageio-ffmpeg 의 ffmpeg 사용).

주의: 서로 다른 장면 사진을 이어 붙인 것이라 프레임 간 연속성이 없다.
      트래커의 track_id 와 engine 의 차선 시간 축 보정은 같은 장면이 반복되는 구간에서만 의미가 있다.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

import cv2
import imageio_ffmpeg

NAME_RE = re.compile(r'^night ride \((\d+)\)\.png$')


def collect_frames(img_dir: Path) -> list[Path]:
    """`night ride (N).png` 원본만 골라 N 오름차순 정렬."""
    found = []
    for p in img_dir.iterdir():
        m = NAME_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('img_dir', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)

    paths = collect_frames(args.img_dir)
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        print(f'이미지 없음: {args.img_dir}')
        return 1

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    H, W = first.shape[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # H.264 — crf 18 (육안 무손실에 가까움), 홀수 크기는 yuv420p 가 못 받으므로 2 의 배수로 맞춤
    writer = imageio_ffmpeg.write_frames(
        str(args.out), (W, H), fps=args.fps, codec='libx264',
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=None, macro_block_size=2,
        output_params=['-crf', '18', '-preset', 'veryfast', '-movflags', '+faststart'])
    writer.send(None)   # 제너레이터 시작

    n_resized = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)   # RGBA PNG → BGR (alpha 버림)
        if img is None:
            print(f'[skip] 읽기 실패: {p.name}')
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            n_resized += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for _ in range(args.repeat):
            writer.send(rgb)
    writer.close()

    n_frames = len(paths) * args.repeat
    print(f'{len(paths)}장 × {args.repeat} = {n_frames}프레임, {W}x{H} @ {args.fps}fps '
          f'({n_frames / args.fps:.1f}s), 크기 맞춤 {n_resized}장 → {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
