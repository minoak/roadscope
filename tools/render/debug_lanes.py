"""debug_lanes.py — 특정 프레임의 차선 검출 과정을 이미지로 확인.

    python tools/render/debug_lanes.py <영상> <frame_idx> [<frame_idx> ...] [--out runs_out/debug]

그림: 위 = 원본 로직 (mvp), 아래 = engine.find_lanes 검증 후. 각 줄 좌: 오버레이, 우: 마스크
    (아래 오른쪽은 바이크 영역을 지운 차선 검출용 lane 마스크)
    초록 = lane 마스크 (class 3), 파랑 = ego 마스크 (내 바이크+라이더)
    노랑 얇은 선 = Hough 선분, 하늘색 = 클러스터 통합 선 (mvp.detect_lane_lines)
    빨강 굵은 선 = 최종 ego left/right, 흰 원 = VP
    engine 의 차선 검증 결과 (통과/탈락 사유) 를 콘솔에 함께 출력한다.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
from roadscope import rules as mvp
from roadscope import engine


def hough_segments(mask):
    """mvp.detect_lane_lines 의 Hough 단계까지만 (선분 원본)."""
    lane_bin = (mask == 3).astype(np.uint8) * 255
    if lane_bin.sum() < 100:
        return []
    lane_clean = cv2.morphologyEx(lane_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    raw = cv2.HoughLinesP(lane_clean, rho=1, theta=np.pi / 180,
                          threshold=40, minLineLength=30, maxLineGap=25)
    return [] if raw is None else np.asarray(raw).reshape(-1, 4).tolist()


def draw_line(img, line, y0, y1, color, thick):
    p0 = (int(round(mvp.line_x_at(line, y0))), int(y0))
    p1 = (int(round(mvp.line_x_at(line, y1))), int(y1))
    cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('frames', type=int, nargs='+')
    ap.add_argument('--out', type=Path, default=ROOT / 'runs_out' / 'debug')
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    models = engine.load_lane_models(engine.LANE_CKPTS, 'cuda')
    cap = cv2.VideoCapture(str(args.video))
    for fi in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            print(f'[skip] frame {fi} 읽기 실패'); continue
        H, W = bgr.shape[:2]
        mask = engine.predict_lane_mask(models, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 'cuda')
        ego_mask = mvp.get_ego_mask(mask)
        lane_mask = engine.lane_input_mask(mask, ego_mask)
        # 원본 로직 (노트북/mvp) 결과
        lines = mvp.detect_lane_lines(mask)
        vp_x, vp_y, vp_conf = mvp.vp_from_lanes(lines, W, H)
        ego = mvp.identify_ego_lane(lines, W, H)
        # engine 결과 (차선 검증 + LDW + 차선 위 주행)
        jl = engine.judge_lane(mask, ego_mask)
        chk, ldw = jl['lane'], jl['ldw']
        cx, cy, cconf = chk['vp']

        def overlay(segs_, lines_, ego_, vp_, title):
            img = (bgr * 0.55).astype(np.uint8)
            img[mask == 3] = (0, 220, 0)
            img[ego_mask > 0] = (img[ego_mask > 0] * 0.5 + np.array([200, 80, 0]) * 0.5).astype(np.uint8)
            for x1, y1, x2, y2 in segs_:
                cv2.line(img, (x1, y1), (x2, y2), (0, 230, 255), 2)
            for l in lines_:
                draw_line(img, l, l['y_top'], l['y_bot'], (255, 200, 0), 3)
            for side in ('left', 'right'):
                if ego_[side] is not None:
                    draw_line(img, ego_[side], max(vp_[1], 0), H - 1, (0, 0, 255), 5)
            cv2.circle(img, (int(vp_[0]), int(vp_[1])), 14, (255, 255, 255), 3)
            cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            return img

        over = overlay(hough_segments(mask), lines, ego, (vp_x, vp_y),
                       f'[before] frame {fi}  lines={len(lines)}  vp_conf={vp_conf:.2f}')
        # after 의 Hough 선분은 바이크 영역을 지운 mask 기준 (원래 mask 선분과 섞지 않는다)
        after = overlay(hough_segments(lane_mask), chk['lines'], chk['ego'], (cx, cy),
                        f'[after] lines={len(chk["lines"])}  vp_conf={cconf:.2f}  '
                        f'ok={chk["ok"]}  LDW L{ldw.level}'
                        + ('  STRADDLE' if ldw.extra.get('straddle') else ''))
        # 사유(한글)는 콘솔에 출력 — cv2 putText 는 한글 불가
        # 차선 위 주행 근거: 근거리 행 범위 (분홍 가로선) + 판정에 쓴 선 주변 허용 폭 (분홍 띠)
        y_near = int(engine.STRADDLE_NEAR_Y * H)
        cv2.line(after, (0, y_near), (W - 1, y_near), (255, 0, 255), 1)
        if ldw.extra.get('straddle'):
            side_line = chk['ego']['left' if '좌측' in ldw.reason else 'right']
            tol = engine.STRADDLE_TOL * W
            for y in range(y_near, H, 4):
                x = mvp.line_x_at(side_line, y)
                cv2.line(after, (int(x - tol), y), (int(x + tol), y), (255, 0, 255), 1)
        check = {k: chk[k] for k in ('ok', 'reason', 'n_raw')}

        mask_vis = np.zeros_like(bgr)
        mask_vis[mask == 3] = (0, 220, 0)
        mask_vis[ego_mask > 0] = (200, 80, 0)
        mask_vis[(mask == 4)] = (0, 0, 180)
        out = np.vstack([np.hstack([over, mask_vis]),
                         np.hstack([after, (lane_mask == 3)[..., None]
                                    * np.array([0, 220, 0], np.uint8)])])
        out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
        path = args.out / f'lanes_frame{fi:04d}.png'
        cv2.imwrite(str(path), out)

        lane_px = int((mask == 3).sum())
        print(f'\n=== frame {fi}: lane px {lane_px}, ego px {int(ego_mask.sum())}, '
              f'lines before {len(lines)} / after {len(chk["lines"])}, '
              f'vp ({cx},{cy}) conf {cconf:.2f} → {path.name}')
        for side in ('left', 'right'):
            l = chk['ego'][side]
            if l is not None:
                u = engine.line_u(l, jl['ego_cx'] or W / 2, cy, H)
                print(f'  after {side}: dydx={l["dydx"]:+.2f} u={u:+.2f} '
                      f'y {l["y_top"]:.0f}~{l["y_bot"]:.0f} len={l["length"]:.0f}')
        print(f'  검증: {check}  ego conf={chk["ego"]["conf"]}')
        print(f'  LDW L{ldw.level} "{ldw.reason}"  extra={ldw.extra}')
    cap.release()


if __name__ == '__main__':
    sys.exit(main())
