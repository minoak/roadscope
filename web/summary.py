# SPDX-License-Identifier: AGPL-3.0-only
"""Summaries of saved engine records; never imports or runs the engine."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median

LABELS = ("안전", "주의", "위험", "치명")
WARNING_NAMES = {"FCW": "정면 충돌", "LDW": "차선 편향", "BSW": "측면 접근", "HEAD": "역주행"}
DISPLAY_MERGE_GAP_SECONDS = 0.5


def display_records(records):
    """표시 전용: 같은 종류·방향의 짧은 끊김을 높은 등급으로 잇는다."""
    rows = [{**r, "warnings": {k: dict(w) for k, w in r.get("warnings", {}).items()}}
            for r in records]
    for name in WARNING_NAMES:
        # 높은 등급부터 메워 짧은 위험·치명이 낮은 등급에 묻히지 않게 한다.
        for threshold in (3, 2, 1):
            previous = None
            for i, row in enumerate(rows):
                warning = row["warnings"].get(name, {})
                level = warning.get("level", 0)
                side = warning.get("side")
                if previous is not None:
                    prior = rows[previous]["warnings"][name]
                    if level > 0 and side != prior.get("side"):
                        previous = None  # 반대 방향이 끼면 짧아도 묶지 않는다.
                if level < threshold:
                    continue
                if previous is not None and i > previous + 1:
                    gap = row["t_sec"] - rows[previous + 1]["t_sec"]
                    if gap < DISPLAY_MERGE_GAP_SECONDS and not math.isclose(
                            gap, DISPLAY_MERGE_GAP_SECONDS, rel_tol=0, abs_tol=1e-9):
                        prior = rows[previous]["warnings"][name]
                        representative = prior if prior["level"] >= level else warning
                        for between in rows[previous + 1:i]:
                            between["warnings"][name] = dict(representative)
                previous = i
    for row in rows:
        row["bin"] = max([row["bin"], *(w.get("level", 0) for w in row["warnings"].values())])
    return rows


def warning_events(records):
    """표시 기록에서 구간당 한 행과 실제 해제 행을 만든다."""
    events, previous = [], {}
    for r in records:
        current = {}
        for name, label in WARNING_NAMES.items():
            w = r.get("warnings", {}).get(name, {})
            level = w.get("level", 0)
            if level > 0:
                key = (level, w.get("side"))
                current[name] = key
                if previous.get(name) != key:
                    events.append({"time": r["t_sec"], "grade": level, "label": LABELS[level],
                                   "warning": name, "reason": w.get("reason") or label,
                                   "overall_grade": r["bin"]})
            elif name in previous:
                events.append({"time": r["t_sec"], "grade": r["bin"], "label": LABELS[r["bin"]],
                               "warning": name, "reason": label + " 경고 해제", "overall_grade": r["bin"]})
        previous = current
    return events


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def camera_summary(meta, records):
    cameras = []
    width = (meta.get("size") or [None])[0]
    fps = meta.get("fps_src")
    for cal in meta.get("car_calib", []):
        interior = cal.get("interior") or {}
        baseline = cal.get("lane_baseline") or {}
        seg = cal.get("segment", 0)
        excluded = sum(obj.get("interior") is True for r in records if r.get("calib_seg", 0) == seg
                       for obj in r.get("objects", []))
        cameras.append({
            "segment": seg,
            "start": cal.get("start_frame", 0) / fps if finite(fps) and fps > 0 else None,
            "end": (cal["end_frame"] + 1) / fps if finite(fps) and fps > 0 and "end_frame" in cal else None,
            "horizon_ratio": cal.get("horizon_ratio"), "horizon_y": cal.get("horizon_y"),
            "horizon_source": cal.get("horizon_src"), "horizon_conf": cal.get("horizon_conf"),
            "vp_x": cal.get("vp_x"),
            "vp_offset_px": cal["vp_x"] - width / 2 if finite(cal.get("vp_x")) and finite(width) else None,
            "camera_height_m": cal.get("cam_h_m"), "camera_height_source": cal.get("cam_h_src"),
            "focal_length_px": cal.get("f_px"), "focal_length_source": cal.get("f_src"),
            "ego_area_percent": round(interior["frac"] * 100, 2) if finite(interior.get("frac")) else None,
            "excluded_boxes": excluded, "lane_baseline_m": baseline.get("baseline_m"),
            "ldw_measurable": baseline.get("ldw_measurable"),
            "lane_baseline_status": ("failed" if baseline.get("ldw_measurable") is False else
                                     "applied" if baseline.get("baseline_m") is not None else
                                     "not_applied" if baseline else "not_recorded"),
            "ldw_shift_px": baseline.get("ldw_shift_px"),
        })
    confs = [c["horizon_conf"] for c in cameras]
    low = meta.get("vehicle") == "car" and "failed" in confs
    if low:
        confidence = {"level": "low", "label": "낮음", "note": "카메라 각도 측정 실패", "low": True}
    elif meta.get("vehicle") != "car" or not confs:
        confidence = {"level": "unknown", "label": "미제공", "note": "영상 단위 카메라 측정 기록 없음", "low": False}
    elif all(c == "high" for c in confs):
        confidence = {"level": "high", "label": "높음", "note": "수평선 측정 신뢰도 높음", "low": False}
    else:
        notes = {"high": "높음", "medium": "보통", "low": "낮음"}
        confidence = {"level": "limited", "label": "제한적", "note": "수평선 측정 신뢰도 " +
                      ", ".join(sorted({notes.get(c, "미제공") for c in confs})), "low": False}
    return cameras, confidence


def build_summary(run_dir, duration=None):
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "meta.json")
    with (run_dir / "frames.jsonl").open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("판정 기록이 비어 있습니다.")
    previous_time = -1
    for r in records:
        t = r.get("t_sec")
        if not finite(t) or t < 0 or t <= previous_time or r.get("bin") not in range(4):
            raise ValueError("판정 기록의 시각 또는 등급이 올바르지 않습니다.")
        previous_time = t
    duration_source = "video"
    if not finite(duration) or duration <= records[-1]["t_sec"]:
        # end_frame is inclusive and refers to ORIGINAL frames, unlike n_frames.
        ends = [c.get("end_frame") for c in meta.get("car_calib", []) if finite(c.get("end_frame"))]
        fps = meta.get("fps_src")
        duration = (max(ends) + 1) / fps if ends and finite(fps) and fps > 0 else None
        duration_source = "car_calib.end_frame"
        if not finite(duration) or duration <= records[-1]["t_sec"]:
            steps = [b["t_sec"] - a["t_sec"] for a, b in zip(records, records[1:])]
            fps_eval = meta.get("fps_eval")
            step = median(steps) if steps else 1 / fps_eval if finite(fps_eval) and fps_eval > 0 else 1
            duration = records[-1]["t_sec"] + step
            duration_source = "estimated_last_interval"
    segments = []
    for i, r in enumerate(records):
        end = records[i + 1]["t_sec"] if i + 1 < len(records) else duration
        if segments and segments[-1]["grade"] == r["bin"]:
            segments[-1]["end"] = end
        else:
            segments.append({"start": r["t_sec"], "end": end, "grade": r["bin"], "label": LABELS[r["bin"]]})

    displayed = display_records(records)

    def intervals(predicate):
        merged = []
        for i, r in enumerate(displayed):
            end = displayed[i + 1]["t_sec"] if i + 1 < len(displayed) else duration
            if predicate(r["bin"]):
                if merged and merged[-1]["end"] == r["t_sec"]:
                    merged[-1]["end"] = end
                else:
                    merged.append({"start": r["t_sec"], "end": end})
        # 시간은 원자료의 실제 등급별 누적 시간, 구간 수·목록만 표시용이다.
        elapsed = sum(s["end"] - s["start"] for s in segments if predicate(s["grade"]))
        return {"count": len(merged), "duration": elapsed, "intervals": merged}

    events = warning_events(displayed)
    peak = max(records, key=lambda r: r["bin"])
    cameras, confidence = camera_summary(meta, records)
    return {"duration": duration, "duration_source": duration_source, "frame_count": len(records),
            "vehicle": meta.get("vehicle", "moto"), "fps_eval": meta.get("fps_eval"),
            "segments": segments, "events": events,
            "peak": {"grade": peak["bin"], "label": LABELS[peak["bin"]], "time": peak["t_sec"],
                     "reason": peak.get("top_reason") or "활성 경고 없음"},
            "danger": intervals(lambda grade: grade >= 2), "caution": intervals(lambda grade: grade == 1),
            "cameras": cameras, "confidence": confidence,
            "unevaluated_duration": records[0]["t_sec"]}
