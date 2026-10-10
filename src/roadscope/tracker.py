# SPDX-License-Identifier: AGPL-3.0-only
"""IoU-based simple tracker — silence blackout 해결용.

매 프레임 Moveable bbox 리스트를 받아서:
- 직전 프레임 bbox 와 IoU > threshold 매칭 → 같은 ID
- 매칭 안 된 신규 객체 → 새 ID
- 매칭 안 된 기존 트랙 → "ghost" (max_ghost_frames 동안 유지)

Ghost 트랙은 마지막 알려진 bbox/severity 를 유지하므로,
seg 가 잠깐 객체를 잃어도 프레임 bin 을 유지할 수 있음.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    severity: int          # 0~3 (zone_sev 등)
    last_seen_frame: int
    ghost_count: int = 0   # 0 = 현재 프레임에서 검출됨, >0 = ghost


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    aa = max(1, (ax1 - ax0) * (ay1 - ay0))
    bb = max(1, (bx1 - bx0) * (by1 - by0))
    union = aa + bb - inter
    return inter / max(union, 1)


class SimpleIouTracker:
    def __init__(self, iou_threshold: float = 0.3, max_ghost_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_ghost_frames = max_ghost_frames
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def update(
        self,
        frame_idx: int,
        detections: List[Tuple[Tuple[int, int, int, int], int]],
    ) -> List[Track]:
        """detections: [(bbox, severity), ...] for current frame.
        반환: 현재 활성 트랙 리스트 (real + ghost).
        """
        # 1. 매칭 — Hungarian 안 쓰고 greedy IoU
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches = []  # (track_id, det_idx, iou_val)

        candidates = []
        for tid in unmatched_tracks:
            for di in unmatched_dets:
                v = iou(self.tracks[tid].bbox, detections[di][0])
                if v >= self.iou_threshold:
                    candidates.append((v, tid, di))
        # 큰 IoU 부터
        candidates.sort(reverse=True)
        used_tracks, used_dets = set(), set()
        for v, tid, di in candidates:
            if tid in used_tracks or di in used_dets:
                continue
            matches.append((tid, di, v))
            used_tracks.add(tid); used_dets.add(di)

        # 2. 매칭된 트랙 업데이트
        for tid, di, _ in matches:
            bbox, sev = detections[di]
            self.tracks[tid].bbox = bbox
            self.tracks[tid].severity = sev
            self.tracks[tid].last_seen_frame = frame_idx
            self.tracks[tid].ghost_count = 0

        # 3. 매칭 안 된 detection → 신규 트랙
        for di in range(len(detections)):
            if di in used_dets:
                continue
            bbox, sev = detections[di]
            self.tracks[self._next_id] = Track(
                track_id=self._next_id, bbox=bbox, severity=sev,
                last_seen_frame=frame_idx, ghost_count=0)
            self._next_id += 1

        # 4. 매칭 안 된 트랙 → ghost 카운트 증가
        to_remove = []
        for tid in unmatched_tracks:
            if tid in used_tracks:
                continue
            self.tracks[tid].ghost_count += 1
            if self.tracks[tid].ghost_count > self.max_ghost_frames:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self._next_id = 0

    def ghost_max_severity(self) -> int:
        """현재 ghost (검출 실패) 트랙의 최대 severity. 0 이면 ghost 없음."""
        return max(
            (t.severity for t in self.tracks.values() if t.ghost_count > 0),
            default=0,
        )

    def n_real(self) -> int:
        return sum(1 for t in self.tracks.values() if t.ghost_count == 0)

    def n_ghost(self) -> int:
        return sum(1 for t in self.tracks.values() if t.ghost_count > 0)
