"""Nexar 라벨과 최종 frames.jsonl 평가. 모델 추론은 실행하지 않는다.

python tools/evaluate/eval_nexar.py [--labels runs_in/nexar/labels.csv] [--runs runs_out/nexar]
엔진이 마지막에 쓰는 meta.json의 시각/행 수를 확인한다. 미완료는 미탐 분모에서 제외.
roadscope 패키지가 import 가능해야 한다 (pip install -e . 또는 PYTHONPATH=src).
라벨·실행 결과(runs_in/, runs_out/)는 리포에 없다. 직접 만든 결과에 대해 실행한다.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roadscope.viz import load_records, render_timeline

# 출처 해시 대상. 키는 실행 기록(run_log.md)에 남은 원래 파일 이름, 값은 이 리포의 위치
CODE_FILES = {"engine.py": "src/roadscope/engine.py", "geom_car.py": "src/roadscope/geom_car.py",
              "car_models.py": "src/roadscope/car_models.py", "mvp.py": "src/roadscope/rules.py"}

# t21 이전의 고정 비교 코호트. 새 결과가 기준 폴더에 생겨도 10개만 비교한다.
ORIGINAL_FILES = {f"{stem}.mp4" for stem in (
    "pos_00411", "pos_00936", "pos_00283", "pos_00413", "pos_00348",
    "pos_00077", "pos_00905", "neg_01988", "neg_01960", "neg_02117")}


def normalized_reason(reason):
    return re.sub(r"-?\d+(?:\.\d+)?", "#", str(reason or "사유 없음"))


def reason_category(reason):
    if "차선 침범" in reason:
        return "BSW 차선 침범"
    if "측면 나란히" in reason:
        return "BSW 측면 나란히"
    return {"BSW": "BSW 기타 측면 거리", "FCW": "FCW 정면", "LDW": "LDW 차선 위 주행"}.get(
        reason.split(":", 1)[0], "기타")


def l1_reasons(records):
    """bin==1 프레임의 대표 사유(배타적)와 활성 채널 사유(중복 가능)."""
    selected = [r for r in records if r["bin"] == 1]
    top, active = Counter(), Counter()
    for record in selected:
        top[f"{record.get('top_warning') or 'unknown'}: {normalized_reason(record.get('top_reason'))}"] += 1
        active.update({f"{channel}: {normalized_reason(warning.get('reason'))}"
                       for channel, warning in (record.get("warnings") or {}).items()
                       if isinstance(warning, dict) and warning.get("level") == 1})
    return {"frames": len(selected), "top_reason_frames": dict(sorted(top.items())),
            "active_reason_frames": dict(sorted(active.items()))}


def combine_l1_reasons(rows):
    selected = [r for r in rows if r["status"] == "evaluated" and r["label"] == 0]
    result = {"frames": sum(r.get("negative_l1_reasons", {}).get("frames", 0) for r in selected)}
    for key in ("top_reason_frames", "active_reason_frames"):
        counter = Counter()
        for row in selected:
            counter.update(row.get("negative_l1_reasons", {}).get(key, {}))
        result[key] = dict(sorted(counter.items()))
        categories = Counter()
        for reason, count in counter.items():
            categories[reason_category(reason)] += count
        result[key.replace("reason", "category")] = dict(sorted(categories.items()))
    return result


def warning_diagnostics(records):
    """새 필드는 경로별 존재 프레임 수만 기록한다. 값 존재를 경고 승격으로 추정하지 않는다."""
    known = {
        "": set("frame_idx t_sec bin bin_label top_warning top_reason warnings lane_measurable lanes vp corridor objects d_front_m d_left_m d_right_m".split()),
        "objects[]": set("track_id cls conf bbox lateral_m distance_m zone_sev direction warning source ghost".split()),
        "warnings.*": set("level reason track_id distance_m offset_ratio side".split()),
    }
    fields, reasons, channels = Counter(), Counter(), Counter()
    ttc_reason_frames = 0
    for record in records:
        present, frame_reasons = set(), set()
        def extra_keys(value, prefix, allowed):
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                if key not in allowed and item is not None:
                    present.add(f"{prefix}.{key}".lstrip("."))
        extra_keys(record, "", known[""])
        for obj in record.get("objects") or []:
            extra_keys(obj, "objects[]", known["objects[]"])
        for channel, warning in (record.get("warnings") or {}).items():
            extra_keys(warning, f"warnings.{channel}", known["warnings.*"])
            if not isinstance(warning, dict) or not isinstance(warning.get("level"), (int, float)):
                continue
            if warning["level"] > 0:
                channels[channel] += 1
                # 거리/시간 숫자가 다른 같은 사유를 묶는다. 한 프레임에서는 중복 제거.
                reason = re.sub(r"-?\d+(?:\.\d+)?", "#", str(warning.get("reason") or "사유 없음"))
                frame_reasons.add(f"{channel}: {reason}")
        fields.update(present)
        reasons.update(frame_reasons)
        ttc_reason_frames += any("ttc" in reason.lower() for reason in frame_reasons)
    return {"extra_field_frames": dict(sorted(fields.items())),
            "optional_status": "present" if fields else "skipped: 새 필드 없음/계약 미정",
            "active_warning_frames": dict(sorted(channels.items())),
            "reason_frames": dict(sorted(reasons.items())), "ttc_reason_frames": ttc_reason_frames}


def combine_diagnostics(rows):
    diagnostics = [r["warning_diagnostics"] for r in rows
                   if r["status"] == "evaluated" and "warning_diagnostics" in r]
    result = {}
    for key in ("extra_field_frames", "active_warning_frames", "reason_frames"):
        counter = Counter()
        for item in diagnostics:
            counter.update(item[key])
        result[key] = dict(sorted(counter.items()))
    result["ttc_reason_frames"] = sum(item["ttc_reason_frames"] for item in diagnostics)
    result["optional_status"] = "present" if result["extra_field_frames"] else "skipped: 새 필드 없음/계약 미정"
    return result


def read_labels(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"file", "label", "time_of_event", "time_of_alert", "light"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("labels.csv 필수 열 누락: " + ", ".join(sorted(required)))
        rows, stems = [], set()
        for line, raw in enumerate(reader, 2):
            name = raw["file"].strip()
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"라벨 {line}행: file은 파일명만 허용합니다.")
            if Path(name).stem in stems or raw["label"] not in ("0", "1"):
                raise ValueError(f"라벨 {line}행: 중복 파일명 또는 잘못된 label")
            stems.add(Path(name).stem)
            row = {**raw, "file": name, "label": int(raw["label"]), "light": raw["light"].strip()}
            for key in ("time_of_event", "time_of_alert"):
                value = float(raw[key]) if raw[key].strip() else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f"라벨 {line}행: {key}는 유한한 0 이상 값이어야 합니다.")
                if row["label"] and value is None:
                    raise ValueError(f"라벨 {line}행: 사고 영상의 {key} 누락")
                row[key] = value
            if row["label"] and row["time_of_alert"] > row["time_of_event"]:
                raise ValueError(f"라벨 {line}행: 경고 기준이 사고 시각보다 늦습니다.")
            rows.append(row)
    if not rows:
        raise ValueError("labels.csv가 비었습니다.")
    return rows


def frame_metrics(label, records):
    """샘플 시각만 사용하며, 중간 프레임을 보간하거나 경고를 연장하지 않는다."""
    event = label["time_of_event"] if label["label"] else None
    alert = label["time_of_alert"]
    result = {}
    for threshold, name in ((2, "danger"), (1, "caution")):
        first = next((r["t_sec"] for r in records
                      if event is not None and r["t_sec"] < event and r["bin"] >= threshold), None)
        result[f"first_{name}_sec"] = first
        result[f"{name}_lead_sec"] = event - first if first is not None else None
        # 양수는 사람의 경고 기준보다 빠름, 음수는 늦음.
        result[f"{name}_alert_delta_sec"] = alert - first if first is not None else None
    window = [r for r in records if event is not None and max(0, event - 3) <= r["t_sec"] < event]
    result["pre_event_window_frames"] = len(window) if event is not None else None
    result["pre_event_danger_frames"] = sum(r["bin"] >= 2 for r in window) if event is not None else None
    result["missed"] = not any(r["bin"] >= 2 for r in window) if event is not None else None
    first_window = next((r["t_sec"] for r in window if r["bin"] >= 2), None)
    result["window_danger_lead_sec"] = event - first_window if first_window is not None else None
    baseline = [r for r in records if event is None or r["t_sec"] < event - 3]
    count = sum(r["bin"] >= 2 for r in baseline)
    caution = sum(r["bin"] == 1 for r in baseline)
    result.update(fp_danger_frames=count, fp_total_frames=len(baseline),
                  fp_ratio=count / len(baseline) if baseline else None,
                  l1_frames=caution, l1_total_frames=len(baseline),
                  l1_ratio=caution / len(baseline) if baseline else None,
                  evaluated_frames=len(records))
    return result


def summarize(rows):
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    positive = [r for r in evaluated if r["label"] == 1]
    leads = [r["danger_lead_sec"] for r in positive if r["danger_lead_sec"] is not None]
    caution_leads = [r["caution_lead_sec"] for r in positive if r["caution_lead_sec"] is not None]
    danger = sum(r["fp_danger_frames"] for r in evaluated)
    frames = sum(r["fp_total_frames"] for r in evaluated)
    result = {"videos_total": len(rows), "videos_evaluated": len(evaluated),
            "positive_total": sum(r["label"] == 1 for r in rows), "positive_evaluated": len(positive),
            "missed_count": sum(r["missed"] for r in positive),
            "miss_rate": sum(r["missed"] for r in positive) / len(positive) if positive else None,
            "mean_danger_lead_sec": mean(leads) if leads else None, "danger_lead_count": len(leads),
            "mean_caution_lead_sec": mean(caution_leads) if caution_leads else None,
            "caution_lead_count": len(caution_leads),
            "fp_danger_frames": danger, "fp_total_frames": frames,
            "fp_ratio": danger / frames if frames else None}
    for name, label in (("negative", 0), ("positive_outside", 1)):
        selected = [r for r in evaluated if r["label"] == label]
        count = sum(r["l1_frames"] for r in selected)
        total = sum(r["l1_total_frames"] for r in selected)
        result[f"l1_{name}_frames"] = count
        result[f"l1_{name}_total_frames"] = total
        result[f"l1_{name}_ratio"] = count / total if total else None
    result["warning_diagnostics"] = combine_diagnostics(evaluated)
    result["negative_l1_reasons"] = combine_l1_reasons(evaluated)
    negatives = [r for r in evaluated if r["label"] == 0]
    result["negative_danger_frames"] = sum(r["fp_danger_frames"] for r in negatives)
    result["negative_total_frames"] = sum(r["fp_total_frames"] for r in negatives)
    result["negative_danger_ratio"] = (result["negative_danger_frames"] / result["negative_total_frames"]
                                        if result["negative_total_frames"] else None)
    window_leads = [r["window_danger_lead_sec"] for r in positive
                    if r.get("window_danger_lead_sec") is not None]
    result["mean_window_danger_lead_sec"] = mean(window_leads) if window_leads else None
    result["window_danger_lead_count"] = len(window_leads)
    return result


def _stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def evaluate(labels_path, runs, *, timelines=True, baseline_runs=None, middle_runs=None, detailed=False,
             compare_runs=None, t46_metrics=False):
    rows = []
    snapshots = {}
    for label in read_labels(labels_path):
        row = {**label, "status": "missing", "detail": "frames.jsonl 없음", "timeline": None}
        directory = Path(runs) / Path(label["file"]).stem
        frames, meta_file = directory / "frames.jsonl", directory / "meta.json"
        try:
            if not frames.is_file():
                rows.append(row)
                continue
            row.update(status="pending", detail="최종 meta.json 대기")
            if not meta_file.is_file() or (directory / "frames.jsonl.tmp").exists():
                if (directory / "frames.jsonl.tmp").exists():
                    row["detail"] = "frames.jsonl.tmp 존재: 최종 결과로 확정하지 않음"
                rows.append(row)
                continue
            before = (_stamp(frames), _stamp(meta_file))
            if before[0][0] > before[1][0]:
                row["detail"] = "프레임이 meta.json보다 새로움: 실행/교체 중"
                rows.append(row)
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
            records = load_records(directory)
            if meta.get("n_frames") != len(records):
                row["detail"] = "meta.json의 n_frames와 레코드 수 불일치"
                rows.append(row)
                continue
            fps = meta.get("fps_eval")
            if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("meta.json의 fps_eval 누락/오류")
            if label["label"] and records[-1]["t_sec"] + 1 / fps + 0.001 < label["time_of_event"]:
                raise ValueError("사고 시각까지 평가된 결과가 아님")
            metrics = frame_metrics(label, records)
            metrics["warning_diagnostics"] = warning_diagnostics(records)
            if label["label"] == 0:
                metrics["negative_l1_reasons"] = l1_reasons(records)
            metrics["interior_evidence"] = {
                "frames": sum(any(o.get("interior") is True for o in r.get("objects", [])) for r in records),
                "box_count": sum(o.get("interior") is True for r in records for o in r.get("objects", [])),
                "class_box_counts": dict(Counter(o.get("cls", "unknown") for r in records
                                                   for o in r.get("objects", []) if o.get("interior") is True))}
            timeline = None
            if timelines:
                timeline = render_timeline(directory, directory / "timeline_eval.png",
                                           time_of_event=label["time_of_event"] if label["label"] else None,
                                           time_of_alert=label["time_of_alert"] if label["label"] else None)
            if before != (_stamp(frames), _stamp(meta_file)):
                row.update(status="pending", detail="읽는 동안 결과 변경: 다시 실행 필요")
            else:
                row.update(metrics, status="evaluated", detail="", timeline=timeline)
                row["car_calib"] = meta.get("car_calib") or []
                confidences = {c.get("horizon_conf", "unknown") for c in row["car_calib"]}
                row["calib_confidence"] = next(iter(confidences)) if len(confidences) == 1 else ("mixed" if confidences else "unknown")
                row["fps_eval"] = fps
                if t46_metrics:
                    row.update(t46_video_metrics(row, records, meta))
                if detailed:
                    snapshots[row["file"]] = (records, meta)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            row.update(status="invalid", detail=str(exc))
        rows.append(row)
    groups = {"all": rows, "night": [r for r in rows if r["light"].lower() == "dark"],
              "day": [r for r in rows if r["light"].lower() == "normal"]}
    other = [r for r in rows if r["light"].lower() not in ("dark", "normal")]
    if other:
        groups["unknown"] = other
    for confidence in sorted({r.get("calib_confidence", "unknown") for r in rows if r["status"] == "evaluated"}):
        groups[f"calib:{confidence}"] = [r for r in rows if r.get("calib_confidence") == confidence]
    for scene in sorted({r.get("scene", "").strip() or "unknown" for r in rows}):
        groups[f"scene:{scene}"] = [r for r in rows if (r.get("scene", "").strip() or "unknown") == scene]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "labels": str(labels_path),
            "runs": str(runs), "complete": all(r["status"] == "evaluated" for r in rows),
            "definitions": {"miss_window": "max(0, event-3) <= t < event; bin >= 2 없으면 미탐",
                            "first_warning": "t < event에서 첫 bin>=2 / bin>=1 샘플",
                            "lead_sec": "event - first; 경고 없는 영상은 null, 평균에서 제외",
                            "alert_delta_sec": "alert - first; 양수=빠름, 음수=늦음",
                            "false_positive": "정상 전체 + 사고 t < event-3; bin>=2 프레임 합 / 전체 프레임 합",
                            "l1": "주의 비율은 bin == 1 (L2/L3 제외). 정상 전체 / 사고 t < event-3을 각각 프레임 가중 집계; 사고 이후 제외",
                            "diagnostics": "완료 결과 전체 프레임의 활성 경고 사유(숫자는 #)별 집계. TTC 사유 프레임은 승격 프레임과 다름. 새 필드는 경로별 존재 수만 집계; 계약에 없는 의미/보정 효과 추정 안 함",
                            "missing": "미완료/누락/잘못된 결과는 지표에서 제외; 미탐으로 간주하지 않음",
            "light": "Dark=밤, Normal=낮, 나머지(예: Twilight)=별도 집계; 낮/밤에 임의 배정하지 않음"},
            "summary": {name: summarize(group) for name, group in groups.items()}, "videos": rows}
    report["definitions"].update(
        window_lead="보조 선행 시간: 사고 전 3초 창 안 첫 L2 이상부터 사고까지. 창 안 검출 영상만 평균",
        calibration="meta.car_calib[].horizon_conf 원값별 영상 집계. 모든 구간이 같은 값이면 해당 그룹, 다르면 mixed; 없음은 unknown. 수평선 신뢰도이며 전체 거리 정확도 인증이 아님",
        current_policy="저장된 bin을 그대로 평가. TTC 값으로 등급을 재판정하지 않음; 이전 엔진/TTC 판정 포함 t20은 당시 출력 비교")
    if baseline_runs is not None:
        baseline = (report if Path(runs).resolve() == Path(baseline_runs).resolve()
                    else evaluate(labels_path, baseline_runs, timelines=False))
        previous = {r["file"]: r for r in baseline["videos"] if r["file"] in ORIGINAL_FILES}
        current = {r["file"]: r for r in rows if r["file"] in ORIGINAL_FILES}
        paired = [name for name in sorted(current) if current[name]["status"] == "evaluated"
                  and previous[name]["status"] == "evaluated"]
        report["comparison"] = {"baseline_runs": str(baseline_runs),
            "baseline": summarize(list(previous.values())), "current": summarize(list(current.values())),
            "paired_count": len(paired), "paired_baseline": summarize([previous[n] for n in paired]),
            "paired_current": summarize([current[n] for n in paired]),
            "videos": [{"file": n, "baseline": previous[n], "current": current[n]} for n in sorted(current)]}
        if middle_runs is not None:
            middle = evaluate(labels_path, middle_runs, timelines=False)
            maps = {"baseline": previous,
                    "t20": {r["file"]: r for r in middle["videos"] if r["file"] in ORIGINAL_FILES},
                    "current": current}
            common = [n for n in sorted(current) if all(m[n]["status"] == "evaluated" for m in maps.values())]
            report["comparison_three"] = {
                "runs": {"baseline": str(baseline_runs), "t20": str(middle_runs), "current": str(runs)},
                "paired_count": len(common), "paired_files": common,
                "summary": {key: summarize([m[n] for n in common]) for key, m in maps.items()},
                "videos": [{"file": n, **{key: m[n] for key, m in maps.items()}} for n in sorted(current)]}
    if detailed:
        ranked = sorted((r for r in rows if r["status"] == "evaluated" and r["fp_ratio"] is not None),
                        key=lambda r: (-r["fp_ratio"], r["file"]))[:5]
        report["case_analysis"] = {
            "ranking": "사고 밖 L2 이상 비율 내림차순, 동률은 파일명 오름차순; 정상은 전체, 사고는 t < event-3",
            "misses": [case_evidence(r, *snapshots[r["file"]], missed=True) for r in rows if r.get("missed")],
            "false_positive_top5": [case_evidence(r, *snapshots[r["file"]], missed=False) for r in ranked]}
    if compare_runs is not None:
        baseline = evaluate(labels_path, compare_runs, timelines=False)
        report["comparison_all"] = compare_all(baseline, report)
    report["definitions"].update(
        interior="interior:true는 엔진이 제외한 박스. 실외 객체의 존재/정면/거리/FCW 근거에서 제외하고 interior_objects로 별도 보존; 실제 실내인지 정답 검증은 아님. 누락/false는 기존 객체로 취급",
        negative_l1_reasons="정상 영상 bin==1만 집계. top_reason은 프레임당 1개(합계=L1); active_reason은 채널별 중복 가능. 숫자를 #로 묶고 방향·문구는 유지")
    return report


def compare_all(baseline, current):
    """모든 라벨의 완료 교집합 비교. 신뢰도 이동과 고정 코호트를 함께 남긴다."""
    old = {r["file"]: r for r in baseline["videos"]}
    new = {r["file"]: r for r in current["videos"]}
    paired = sorted(n for n in old.keys() & new.keys()
                    if old[n]["status"] == new[n]["status"] == "evaluated")
    groups = {"all": paired, "day": [n for n in paired if new[n]["light"].lower() == "normal"],
              "night": [n for n in paired if new[n]["light"].lower() == "dark"]}
    other = [n for n in paired if new[n]["light"].lower() not in ("normal", "dark")]
    if other:
        groups["other_light"] = other
    summary = {key: {"baseline": summarize([old[n] for n in names]),
                     "current": summarize([new[n] for n in names])} for key, names in groups.items()}
    confidences = sorted({r["calib_confidence"] for n in paired for r in (old[n], new[n])})
    dynamic, fixed = {}, {}
    for confidence in confidences:
        dynamic[confidence] = {key: summarize([rows[n] for n in paired if rows[n]["calib_confidence"] == confidence])
                               for key, rows in (("baseline", old), ("current", new))}
        names = [n for n in paired if old[n]["calib_confidence"] == confidence]
        fixed[confidence] = {"files": names, "baseline": summarize([old[n] for n in names]),
                             "current": summarize([new[n] for n in names])}
    common_hits = [n for n in paired if new[n]["label"] and not old[n]["missed"] and not new[n]["missed"]]
    return {"baseline_runs": baseline["runs"], "current_runs": current["runs"],
            "paired_count": len(paired), "paired_files": paired,
            "unpaired_files": sorted((old.keys() | new.keys()) - set(paired)), "summary": summary,
            "calibration_dynamic": dynamic, "calibration_baseline_cohort": fixed,
            "calibration_transitions": dict(Counter(f"{old[n]['calib_confidence']} -> {new[n]['calib_confidence']}" for n in paired)),
            "common_window_hits": {"files": common_hits,
                "baseline_mean_sec": mean(old[n]["window_danger_lead_sec"] for n in common_hits) if common_hits else None,
                "current_mean_sec": mean(new[n]["window_danger_lead_sec"] for n in common_hits) if common_hits else None},
            "new_misses": [n for n in paired if new[n]["missed"] and not old[n]["missed"]],
            "recovered_misses": [n for n in paired if old[n]["missed"] and not new[n]["missed"]],
            "videos": [{"file": n, "baseline": old[n], "current": new[n]} for n in paired]}


def _span(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return [min(values), max(values)] if values else None


def case_evidence(row, records, meta, *, missed):
    """저장된 프레임만 추출. 박스가 사고 당사자인지/실제 오경보인지는 단정하지 않는다."""
    event = row["time_of_event"] if row["label"] else None
    selected = [r for r in records if (
        max(0, event - 3) <= r["t_sec"] < event if missed
        else r["bin"] >= 2 and (event is None or r["t_sec"] < event - 3))]
    evidence, fronts, targets, close_side, road_objects, danger_targets = [], [], [], [], [], []
    for r in selected:
        interior = [o for o in r.get("objects") or [] if o.get("interior") is True]
        objects = [o for o in r.get("objects") or [] if o.get("interior") is not True]
        road_objects.extend(objects)
        front = [o for o in objects if o.get("direction") == "front"]
        fcw = (r.get("warnings") or {}).get("FCW") or {}
        target = [o for o in objects if fcw.get("track_id") is not None and o.get("track_id") == fcw["track_id"]]
        nearby = [o for o in objects if o.get("direction") != "front"
                  and o.get("distance_m") is not None and o["distance_m"] < 5]
        fronts.extend(front)
        targets.extend(target)
        if fcw.get("level", 0) >= 2:
            danger_targets.extend(target)
        close_side.extend(nearby)
        warning_ids = {w.get("track_id") for w in (r.get("warnings") or {}).values()
                       if isinstance(w, dict) and w.get("level", 0) >= 2 and w.get("track_id") is not None}
        chosen = objects if missed else [o for o in objects if o in target or o.get("track_id") in warning_ids]
        evidence.append({"frame_idx": r["frame_idx"], "t_sec": r["t_sec"], "bin": r["bin"],
                         "calib_seg": r.get("calib_seg"), "front_box_count": len(front),
                         "front_real_box_count": sum(not o.get("ghost", False) for o in front),
                         "FCW": fcw, "warnings": r.get("warnings"),
                         "interior_objects": interior,
                         "objects": [{k: o.get(k) for k in (
                             "track_id", "cls", "bbox", "distance_m", "lateral_m", "direction",
                             "in_lane", "ghost", "source", "dist_geo", "ttc", "interior")} for o in chosen]})
    cause_counts = Counter(f"L{e['FCW'].get('level', 0)}:{e['FCW'].get('cause')}" for e in evidence)
    target_geometry = Counter((o.get("dist_geo") or {}).get("used", "unknown") for o in targets)
    width = meta.get("size", [None])[0]
    size = meta.get("size") or [None, None]
    bottom_wide = sum(bool(o.get("bbox") and size[0] and size[1]
                           and o["bbox"][1] > size[1] * 0.7
                           and o["bbox"][2] - o["bbox"][0] > size[0] * 0.6) for o in targets)
    danger_channels = Counter(channel for r in selected for channel, warning in (r.get("warnings") or {}).items()
                              if isinstance(warning, dict) and warning.get("level", 0) >= 2)
    invalid_vp = [c.get("vp_x") for c in row["car_calib"] if width is not None
                  and c.get("vp_x") is not None and not 0 <= c["vp_x"] < width]
    unavailable = [o for o in road_objects if o.get("distance_m") is None]
    stats = {"selected_frames": len(selected),
             "interior_box_count": sum(len(e["interior_objects"]) for e in evidence),
             "interior_frames": sum(bool(e["interior_objects"]) for e in evidence),
             "interior_class_box_counts": dict(Counter(o.get("cls") for e in evidence for o in e["interior_objects"])),
             "road_box_count": len(road_objects), "distance_unavailable_box_count": len(unavailable),
             "distance_unavailable_gate_counts": dict(Counter(g for o in unavailable for g in (o.get("dist_geo") or {}).get("gated", []))),
             "distance_unavailable_directions": dict(Counter(o.get("direction") for o in unavailable)),
             "front_box_frames": sum(e["front_box_count"] > 0 for e in evidence),
             "front_real_box_frames": sum(e["front_real_box_count"] > 0 for e in evidence),
             "front_distance_m": _span(o.get("distance_m") for o in fronts),
             "front_lateral_m": _span(o.get("lateral_m") for o in fronts),
             "fcw_target_distance_m": _span(o.get("distance_m") for o in targets),
             "fcw_target_lateral_m": _span(o.get("lateral_m") for o in targets),
             "fcw_cause_frames": dict(cause_counts), "fcw_target_geometry": dict(target_geometry),
             "fcw_target_class_counts": dict(Counter(o.get("cls") for o in targets)),
             "fcw_danger_target_class_counts": dict(Counter(o.get("cls") for o in danger_targets)),
             "fcw_danger_target_distance_m": _span(o.get("distance_m") for o in danger_targets),
             "fcw_danger_car_3_to_5m_frames": sum(o.get("cls") == "car" and o.get("distance_m") is not None
                                                   and 3 <= o["distance_m"] < 5 for o in danger_targets),
             "fcw_target_ghost_frames": sum(o.get("ghost", False) for o in targets),
             "fcw_target_bottom_wide_frames": bottom_wide,
             "danger_channel_frames": dict(danger_channels),
             "near_side_distance_m": _span(o.get("distance_m") for o in close_side),
             "near_side_lateral_m": _span(o.get("lateral_m") for o in close_side),
             "vp_x_outside_image": invalid_vp}
    def span_text(key):
        value = stats[key]
        return "없음" if value is None else f"{value[0]:.1f}~{value[1]:.1f}"
    dkey, xkey = ("front_distance_m", "front_lateral_m") if missed else ("fcw_target_distance_m", "fcw_target_lateral_m")
    text = (f"정면 박스 {stats['front_box_frames']}/{len(selected)}프레임(실검출 있음 {stats['front_real_box_frames']}, "
            f"유령만 {stats['front_box_frames'] - stats['front_real_box_frames']}); "
            f"{'정면 객체' if missed else 'FCW 대상'} 거리 {span_text(dkey)} m, X {span_text(xkey)} m. "
            f"FCW {dict(cause_counts)}, 보정 {row['calib_confidence']}. "
            f"실내 제외 {stats['interior_box_count']}박스/{stats['interior_frames']}프레임(정면 근거에서 제외).")
    if missed:
        if unavailable:
            text += (f" 거리 없음 {len(unavailable)}/{len(road_objects)}박스, "
                     f"거리 제외 사유 {stats['distance_unavailable_gate_counts']}.")
        if stats["front_distance_m"] is not None and stats["front_distance_m"][0] >= 5:
            text += " 정면 기록 거리가 L2 기준 5 m 미만에 진입하지 않음."
        if close_side:
            text += f" 5 m 미만 측면 객체는 X {span_text('near_side_lateral_m')} m로 정면에서 제외됨."
        active = [e for e in evidence if e["FCW"].get("level", 0) > 0]
        if active:
            first, last = active[0], active[-1]
            text += (f" FCW 활성 구간 {first['t_sec']:.3f}~{last['t_sec']:.3f}초의 시작/끝 거리는 "
                     f"{first['FCW'].get('distance_m')}→{last['FCW'].get('distance_m')} m.")
        text += " TTC는 기록만이며, 사고 당사자/실제 거리 오차/실내 제외의 정당성은 로그만으로 확정 불가."
    else:
        text += (f" 거리 선택 {dict(target_geometry)}, FCW 대상 유령 {stats['fcw_target_ghost_frames']}프레임; "
                 f"대상 클래스 {stats['fcw_target_class_counts']}, 위험 이상 채널 {dict(danger_channels)}(중복 가능).")
        if bottom_wide:
            text += f" 하단 30%에서 시작하며 폭이 화면 60%를 넘는 박스 {bottom_wide}프레임: 하단 영역 오검출 가능성, 실물 확인 필요."
        if invalid_vp:
            text += f" 소실점 x={invalid_vp}가 영상 폭 {width}px 밖이라 정면 분류 왜곡이 의심됨."
        elif row["calib_confidence"] in ("low", "failed"):
            text += " 낮은 수평선 신뢰도에서 근접 거리 판정이 반복됨."
        else:
            text += " 정면 근접 박스의 거리/면적 조건으로 경고가 반복됨."
    return {"file": row["file"], "window": ([max(0, event - 3), event] if missed else [0, event - 3 if event is not None else None]),
            "window_end_exclusive": event is not None, "fp_ratio": row["fp_ratio"],
            "fp_danger_frames": row["fp_danger_frames"], "fp_total_frames": row["fp_total_frames"],
            "calib_confidence": row["calib_confidence"], "car_calib": row["car_calib"],
            "whole_video_interior": row.get("interior_evidence", {}),
            "summary": stats, "explanation": text, "frames": evidence}


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _ratio(row):
    value = row.get("fp_ratio")
    return "—" if value is None else f"{100 * value:.2f}% ({row['fp_danger_frames']}/{row['fp_total_frames']})"


def _delta(value):
    if value is None:
        return "—"
    return f"{abs(value):.3f}초 {'빠름' if value > 0 else '늦음' if value < 0 else '동일'}"


def _l1(row, group):
    ratio = row.get(f"l1_{group}_ratio")
    return "—" if ratio is None else (f"{ratio * 100:.2f}% "
        f"({row[f'l1_{group}_frames']}/{row[f'l1_{group}_total_frames']})")


def _comparison_cell(row):
    if row["status"] != "evaluated":
        return row["status"]
    miss = str(int(row["missed"])) if row["label"] else "—"
    l1 = f"{row['l1_ratio'] * 100:.2f}%" if row["l1_ratio"] is not None else "—"
    return f"{miss} / {_number(row['danger_lead_sec'])} / {_ratio(row)} / {l1}"


def all_comparison_markdown(comparison):
    old, new = (comparison["summary"]["all"][key] for key in ("baseline", "current"))
    lines = ["", "## t24 대 t29 전체 비교", "",
             f"`{comparison['baseline_runs']}` → `{comparison['current_runs']}`. "
             f"양쪽 완료 {comparison['paired_count']}개; 미완료 {comparison['unpaired_files']}.",
             "저장된 bin을 그대로 사용하며 동일 라벨의 완료 교집합으로 계산. 각 실행의 원래 샘플을 보간하지 않음.", "",
             "| 지표 | t24 | t29 |", "|---|---:|---:|"]
    for key in ("videos_evaluated", "positive_evaluated", "missed_count", "mean_window_danger_lead_sec",
                "window_danger_lead_count", "mean_danger_lead_sec"):
        lines.append(f"| {key} | {_number(old[key])} | {_number(new[key])} |")
    lines += [f"| 사고 밖 위험 이상 | {_ratio(old)} | {_ratio(new)} |",
              f"| 정상 L1 | {_l1(old, 'negative')} | {_l1(new, 'negative')} |",
              f"| 사고 밖 L1 | {_l1(old, 'positive_outside')} | {_l1(new, 'positive_outside')} |"]
    def negative_danger(group):
        ratio = group['negative_danger_ratio']
        return "—" if ratio is None else f"{100 * ratio:.2f}% ({group['negative_danger_frames']}/{group['negative_total_frames']})"
    lines += [f"| 정상 위험 이상 | {negative_danger(old)} | {negative_danger(new)} |", "",
              f"새 미탐: {comparison['new_misses']}; 해소된 미탐: {comparison['recovered_misses']}."]
    hits = comparison["common_window_hits"]
    lines += [f"양쪽 모두 사고 창 안에서 검출된 동일 {len(hits['files'])}개만 비교한 창 내 평균 선행: "
              f"{_number(hits['baseline_mean_sec'])} → {_number(hits['current_mean_sec'])}초. "
              "전체 창 내 평균의 검출 표본 수 변화와 구별할 것."]
    def group_table(title, groups):
        result = ["", f"### {title}", "",
                  "| 구분 | t24 영상/미탐·사고 | t29 영상/미탐·사고 | 창 내 평균 선행 t24→t29 (n) | 사고 밖 위험 t24 | 사고 밖 위험 t29 | 정상 L1 t24 | 정상 L1 t29 |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, group in groups.items():
            a, b = group['baseline'], group['current']
            result.append(f"| {name} | {a['videos_evaluated']} / {a['missed_count']}·{a['positive_evaluated']} | "
                          f"{b['videos_evaluated']} / {b['missed_count']}·{b['positive_evaluated']} | "
                          f"{_number(a['mean_window_danger_lead_sec'])} ({a['window_danger_lead_count']}) → "
                          f"{_number(b['mean_window_danger_lead_sec'])} ({b['window_danger_lead_count']}) | "
                          f"{_ratio(a)} | {_ratio(b)} | {_l1(a, 'negative')} | {_l1(b, 'negative')} |")
        return result
    lines += ["", "조명 라벨: Normal=낮, Dark=밤, other_light=Twilight 등 나머지. 낮/밤 합계에서 나머지를 누락한 전체 지표를 만들지 않음."]
    lines += group_table("낮/밤 및 기타 조명", {k: v for k, v in comparison['summary'].items() if k != 'all'})
    lines += ["", "실행마다 신뢰도 등급이 바뀌므로 아래 원값별 그룹은 구성원이 다를 수 있음. 신뢰도는 거리 정확도의 정답 등급이 아님."]
    lines += group_table("각 실행의 보정 신뢰도별", comparison['calibration_dynamic'])
    lines += ["", "등급 이동: " + str(comparison['calibration_transitions']),
              "동일 영상 비교를 위해 다음 표는 t24의 등급으로 구성원을 고정함."]
    lines += group_table("t24 보정 신뢰도 코호트 고정", comparison['calibration_baseline_cohort'])
    lines += ["", "### 30개 영상별 비교", "",
              "각 실행: 미탐(1) / 사고 전 3초 창 내 선행(초) / 사고 밖 위험 이상 / L1. "
              "정상 L1은 전체, 사고 L1은 사고 밖이며 정상의 미탐·선행은 —.", "",
              "| 영상 | 낮/밤 | 보정 t24→t29 | t24 | t29 |", "|---|---|---|---|---|"]
    def cell(row):
        miss = str(int(row['missed'])) if row['label'] else '—'
        l1 = '—' if row['l1_ratio'] is None else f"{row['l1_ratio'] * 100:.2f}% ({row['l1_frames']}/{row['l1_total_frames']})"
        return f"{miss} / {_number(row['window_danger_lead_sec'])} / {_ratio(row)} / {l1}"
    for item in comparison['videos']:
        a, b = item['baseline'], item['current']
        lines.append(f"| {item['file']} | {b['light']} | {a['calib_confidence']}→{b['calib_confidence']} | {cell(a)} | {cell(b)} |")
    lines += ["", "### 정상 주의(L1) 사유별", "",
              "bin==1만 집계. 대표 사유는 배타적이며 합계가 정상 L1 프레임 수와 일치. "
              "활성 사유는 동시 경고를 각각 세므로 합계가 L1보다 클 수 있음. "
              "사유별 증감은 집계 변화이며 동일 객체가 다른 사유로 이동했다는 인과 증명은 아님."]
    for key, title in (("top_category_frames", "대표 사유 유형(방향·유령 문구 통합)"),
                       ("top_reason_frames", "대표 사유"), ("active_reason_frames", "동시 발생 포함 활성 사유")):
        a, b = old['negative_l1_reasons'][key], new['negative_l1_reasons'][key]
        lines += ["", f"| {title} | t24 프레임 | t29 프레임 | 증감 |", "|---|---:|---:|---:|"]
        for reason in sorted(a.keys() | b.keys()):
            safe = reason.replace('|', '/').replace('\n', ' ')
            lines.append(f"| {safe} | {a.get(reason, 0)} | {b.get(reason, 0)} | {b.get(reason, 0) - a.get(reason, 0):+d} |")
    return lines


def markdown(report):
    total = report["summary"]["all"]
    lines = [f"미탐 {total['missed_count']}/{total['positive_evaluated']} | "
             f"사고 전 3초 창 평균 위험 선행 {_number(total['mean_window_danger_lead_sec'])}초 (n={total['window_danger_lead_count']}) | "
             f"사고 밖 위험 이상 {_ratio(total)} | 정상 주의(L1) {_l1(total, 'negative')}", "",
             "# Nexar 충돌 경고 평가", "", f"생성: {report['created_at']}", "",
             "전체 완료" if report["complete"] else "**부분 결과 — 미완료 영상을 미탐/정상으로 계산하지 않음.**", "",
             f"전체 구간 첫 위험 평균 선행: {_number(total['mean_danger_lead_sec'])}초 "
             f"(n={total['danger_lead_count']}). 이 평균에는 사고 창 밖 경고와 미탐 영상의 이른 경고도 포함됨.", "",
             "| 구분 | 평가 영상 | 미탐/사고 영상 | 평균 위험 리드타임(표본) | 평균 주의 리드타임(표본) | 오탐 프레임 비율 |",
             "|---|---:|---|---:|---:|---:|"]
    for name, group in report["summary"].items():
        title = {"all": "전체", "night": "밤", "day": "낮", "unknown": "미상"}.get(name, name)
        missed = f"{group['missed_count']}/{group['positive_total']}"
        if group["positive_evaluated"] < group["positive_total"]:
            missed = f"미확정 (현재 미탐 {group['missed_count']}, 사고 평가 {group['positive_evaluated']}/{group['positive_total']})"
        lines.append(f"| {title} | {group['videos_evaluated']}/{group['videos_total']} | {missed} | "
                     f"{_number(group['mean_danger_lead_sec'])} ({group['danger_lead_count']}) | "
                     f"{_number(group['mean_caution_lead_sec'])} ({group['caution_lead_count']}) | {_ratio(group)} |")
    lines += ["", "| 영상 | 밤/낮 | 상태 | 사고/경고 기준(초) | 첫 위험 / 주의(초) | 위험 / 주의 리드(초) | 기준 대비 위험 / 주의 | 미탐 | 오탐 프레임 비율 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in report["videos"]:
        ok = row["status"] == "evaluated"
        miss = ("미탐" if row.get("missed") else "검출") if ok and row["label"] else "—"
        file = f"[{row['file']}]({Path(row['file']).stem}/timeline_eval.png)" if row["timeline"] else row["file"]
        state = {"evaluated": "평가 완료", "missing": "결과 없음", "pending": "미완료", "invalid": "오류"}[row["status"]]
        lines.append(f"| {file} | {row['light']} | {state} | {_number(row['time_of_event'])} / {_number(row['time_of_alert'])} | "
                     f"{_number(row.get('first_danger_sec'))} / {_number(row.get('first_caution_sec'))} | "
                     f"{_number(row.get('danger_lead_sec'))} / {_number(row.get('caution_lead_sec'))} | "
                     f"{_delta(row.get('danger_alert_delta_sec'))} / {_delta(row.get('caution_alert_delta_sec'))} | {miss} | {_ratio(row)} |")
    lines += ["", "## 주의(L1) 비율", "", "| 구분 | 정상 영상 전체 | 사고 영상 사고 밖 |", "|---|---:|---:|"]
    for name, group in report["summary"].items():
        lines.append(f"| {name} | {_l1(group, 'negative')} | {_l1(group, 'positive_outside')} |")
    if report.get("comparison_all"):
        lines += all_comparison_markdown(report['comparison_all'])
    comparison = report.get("comparison")
    if comparison and not report.get("comparison_three"):
        lines += ["", "## 기존 10개 비교", "", f"기준 폴더: `{comparison['baseline_runs']}`",
                  "동일 영상 양쪽 완료 교집합만 아래 합계에 사용. 누락/미완료는 0으로 대체하지 않음.", "",
                  "| 지표 | 이전 | 현재 |", "|---|---:|---:|"]
        old, new = comparison["paired_baseline"], comparison["paired_current"]
        for key in ("videos_evaluated", "positive_evaluated", "missed_count", "mean_danger_lead_sec", "mean_caution_lead_sec"):
            lines.append(f"| {key} | {_number(old[key])} | {_number(new[key])} |")
        lines += [f"| 사고 밖 위험 이상 | {_ratio(old)} | {_ratio(new)} |",
                  f"| 정상 L1 | {_l1(old, 'negative')} | {_l1(new, 'negative')} |",
                  f"| 사고 밖 L1 | {_l1(old, 'positive_outside')} | {_l1(new, 'positive_outside')} |", "",
                  "영상별: 미탐(1)/위험 선행 초/사고 밖 위험 이상/L1. 정상 영상의 미탐·선행은 —.", "",
                  "| 영상 | 이전 | 현재 |", "|---|---|---|"]
        for item in comparison["videos"]:
            lines.append(f"| {item['file']} | {_comparison_cell(item['baseline'])} | {_comparison_cell(item['current'])} |")
    three = report.get("comparison_three")
    if three:
        keys = ("baseline", "t20", "current")
        lines += ["", "## 기존 10개 세 결과 비교", "",
                  "이전 엔진 / 새 기하·TTC 판정 포함(t20) / 현재(TTC 기록만). 저장된 각 버전의 bin을 사용.",
                  " / ".join(f"`{three['runs'][key]}`" for key in keys),
                  f"세 결과 모두 완료인 {three['paired_count']}/10개 교집합만 합계 비교에 사용. "
                  "서로 다른 평가 FPS의 원래 샘플을 사용하며 보간하지 않음. 엔진 변경의 개별 효과를 분리하는 실험은 아님.", "",
                  "| 지표 | 이전 엔진 | t20 | 현재 |", "|---|---:|---:|---:|"]
        for key in ("videos_evaluated", "positive_evaluated", "missed_count", "mean_danger_lead_sec",
                    "mean_caution_lead_sec", "mean_window_danger_lead_sec"):
            lines.append(f"| {key} | " + " | ".join(_number(three['summary'][k][key]) for k in keys) + " |")
        for name, formatter in (("사고 밖 위험 이상", _ratio), ("정상 L1", lambda r: _l1(r, 'negative'))):
            lines.append(f"| {name} | " + " | ".join(formatter(three['summary'][k]) for k in keys) + " |")
        lines += ["", "영상별: 미탐(1) / 위험 선행 초 / 사고 밖 위험 이상 / L1. 정상의 미탐·선행은 —.", "",
                  "| 영상 | 이전 엔진 | t20 | 현재 |", "|---|---|---|---|"]
        for item in three["videos"]:
            lines.append(f"| {item['file']} | " + " | ".join(_comparison_cell(item[k]) for k in keys) + " |")
            for k in keys:
                if item[k]["status"] != "evaluated":
                    lines.append(f"\n미완료: {item['file']} ({k}): {item[k]['detail']}\n")
    cases = report.get("case_analysis")
    if cases:
        lines += ["", "## 보정 신뢰도별 지표", "",
                  "car_calib의 horizon_conf 원값으로 집계. 구간별 값이 다르면 mixed로 분리하며, 전체 거리 정확도 등급으로 해석하지 않음.", "",
                  "| 신뢰도 | 영상 | 미탐/사고 | 평균 위험 선행(초) | 사고 밖 위험 이상 | 정상 L1 |", "|---|---:|---:|---:|---:|---:|"]
        for name, group in report["summary"].items():
            if name.startswith("calib:"):
                lines.append(f"| {name[6:]} | {group['videos_evaluated']} | {group['missed_count']}/{group['positive_evaluated']} | "
                             f"{_number(group['mean_danger_lead_sec'])} | {_ratio(group)} | {_l1(group, 'negative')} |")
        lines += ["", "## 미탐 사고의 사고 전 3초", "",
                  "박스 유무·거리는 로그 기준이며 유령 박스를 구별함. interior:true는 정면/FCW 근거에서 제외하고 interior_objects로 별도 보존. "
                  "분석 구간의 프레임별 박스·거리·X·FCW·보정 구간은 eval.json의 case_analysis에 수록."]
        for item in cases["misses"]:
            lines += ["", f"- **{item['file']}** ({item['window'][0]:.3f} ≤ t < {item['window'][1]:.3f}초): {item['explanation']}"]
        lines += ["", "## 사고 밖 위험 이상 비율 상위 5개", "", cases["ranking"],
                  "실제 위험 여부/정체 여부는 영상으로 검증하지 않음. 아래는 경고를 만든 기록상의 경로이며, 보정 low만으로 실패를 단정하지 않음."]
        for item in cases["false_positive_top5"]:
            lines += ["", f"- **{item['file']}** {_ratio(item)}: {item['explanation']}"]
        lines += ["", "### 위험 상위 영상의 실내 제외와 남은 FCW", "",
                  "실내 수는 영상 전체의 제외 기록, FCW 수는 사고 밖 위험 이상 프레임 중 FCW 자체가 L2 이상인 대상만 계산. "
                  "person 클래스가 남아 있어도 실내 박스가 남았다는 뜻은 아님. 주차/주행 및 실내 제외의 옳고 그름은 영상 검증이 필요.", "",
                  "| 영상 | 전체 실내 박스 수(클래스별) | L2 이상 FCW 대상 클래스 | car 거리 3≤d<5 m 프레임 |",
                  "|---|---|---|---:|"]
        for item in cases['false_positive_top5']:
            stats = item['summary']
            lines.append(f"| {item['file']} | {item['whole_video_interior'].get('class_box_counts', {})} | "
                         f"{stats['fcw_danger_target_class_counts']} | {stats['fcw_danger_car_3_to_5m_frames']} |")
        lines += ["", "### 대표 프레임 근거", "",
                  "각 분석 구간의 첫/중간/마지막 샘플. 미탐은 FCW 대상 또는 가장 가까운 정면 박스, 오경보 후보는 FCW 대상 박스. "
                  "JSON에는 미탐 창의 전체 객체와 오경보 후보 프레임의 FCW/위험 채널 대상 객체를 수록. 실내는 별도 목록.", "",
                  "| 영상 | t(초) | 박스 [x1,y1,x2,y2] | 거리(m) | X(m) | 유령 | FCW level/cause |", "|---|---:|---|---:|---:|---|---|"]
        for item in cases["misses"] + cases["false_positive_top5"]:
            frames = item["frames"]
            for idx in sorted({0, len(frames) // 2, len(frames) - 1}) if frames else []:
                e = frames[idx]
                targets = [o for o in e["objects"] if e["FCW"].get("track_id") is not None and o["track_id"] == e["FCW"]["track_id"]]
                fronts = sorted((o for o in e["objects"] if o["direction"] == "front"),
                                key=lambda o: o["distance_m"] if o["distance_m"] is not None else math.inf)
                obj = (targets or fronts or [{}])[0]
                lines.append(f"| {item['file']} | {e['t_sec']:.3f} | {obj.get('bbox', '없음')} | {_number(obj.get('distance_m'))} | "
                             f"{_number(obj.get('lateral_m'))} | {obj.get('ghost', '—')} | {e['FCW'].get('level')}/{e['FCW'].get('cause')} |")
    diagnostics = report["summary"]["all"]["warning_diagnostics"]
    lines += ["", "## 경고 사유와 선택 필드", "", f"새 필드: {diagnostics['optional_status']}",
              f"TTC를 사유로 명시한 활성 경고 프레임: {diagnostics['ttc_reason_frames']} (승격 수 아님)", "",
              "| 경고 사유 (숫자는 #) | 프레임 수 |", "|---|---:|"]
    for reason, count in diagnostics["reason_frames"].items():
        safe_reason = reason.replace("|", "/").replace("\n", " ")
        lines.append(f"| {safe_reason} | {count} |")
    if diagnostics["extra_field_frames"]:
        lines += ["", "| 선택 필드 경로 | 값이 있는 프레임 수 |", "|---|---:|"]
        lines += [f"| {field} | {count} |" for field, count in diagnostics["extra_field_frames"].items()]
    lines += ["", "## 계산 기준", ""] + [f"- {value}" for value in report["definitions"].values()]
    lines += ["- 평균 리드타임은 첫 경고가 있었던 사고 영상만 사용. 미탐 수와 함께 해석할 것.",
              "- 오탐은 라벨에 따른 대리지표이며, 사고 3초 전에도 실제 위험이 있을 수 있음.",
              "- 초 단위 지표는 표에서만 반올림. 프레임 가중 비율이며 영상별 비율의 단순 평균이 아님."]
    problems = [r for r in report["videos"] if r["detail"]]
    if problems:
        lines += ["", "## 미평가 사유", ""]
        lines += [f"- {r['file']}: {r['detail'].replace(chr(10), ' ')}" for r in problems]
    return "\n".join(lines) + "\n"


def t46_video_metrics(row, records, meta):
    """Stored output only: clip exclusion, LDW coverage and auditable miss evidence."""
    calibration = row["car_calib"]
    if not calibration or any(c.get("horizon_conf") not in ("high", "medium", "low", "failed")
                              for c in calibration):
        raise ValueError("t46: car_calib horizon_conf 누락/알 수 없는 값")
    failed = [c["segment"] for c in calibration if c["horizon_conf"] == "failed"]
    trust = [c.get("lane_trust", {}).get("trusted") for c in calibration]
    known_trust = all(isinstance(value, bool) for value in trust)
    if any(not isinstance(r.get("lane_measurable"), bool) for r in records):
        raise ValueError("t46: lane_measurable 누락/잘못된 값")
    ldw = Counter()
    for r in records:
        warning = r["warnings"]["LDW"]
        if warning["level"] > 0:
            side = warning.get("side")
            if side not in ("left", "right"):
                reason = warning.get("reason", "")
                side = "left" if "좌측" in reason else "right" if "우측" in reason else "unknown"
            ldw[side] += 1
    result = {"outlier": bool(failed), "failed_segments": failed,
              "mixed_failure": bool(failed) and len(failed) < len(calibration),
              "lane_trusted": all(trust) if known_trust else None,
              "lane_any_trusted": any(trust) if known_trust else None,
              "ldw_measurable_frames": sum(r["lane_measurable"] for r in records),
              "ldw_warning_frames": dict(ldw), "source": meta.get("source"),
              "frame_time_range_sec": [records[0]["t_sec"], records[-1]["t_sec"]]}
    if row["missed"]:
        evidence = case_evidence(row, records, meta, missed=True)
        stats = evidence["summary"]
        def span(key):
            value = stats[key]
            return "없음" if value is None else f"{value[0]:.2f}~{value[1]:.2f}"
        window_segments = sorted({r.get("calib_seg") for r in evidence["frames"]
                                  if r.get("calib_seg") is not None})
        segment_conf = {str(c["segment"]): c["horizon_conf"] for c in calibration
                        if c["segment"] in window_segments}
        channel_levels = {channel: dict(Counter(str(e["warnings"][channel]["level"])
                                               for e in evidence["frames"]))
                          for channel in ("FCW", "BSW", "LDW", "HEAD")}
        evidence["window_segment_confidence"] = segment_conf
        evidence["channel_level_frames"] = channel_levels
        front_range = stats["front_distance_m"]
        mechanism = ("정면 객체 기록이 없어 FCW 정면 거리 근거 없음. " if not stats["front_box_frames"] else
                     "정면 기록 거리가 모두 5 m 이상으로 거리 기반 L2 범위에 진입하지 않음. "
                     if front_range is not None and front_range[0] >= 5 else
                     "정면 거리와 FCW 대상의 대응은 아래 프레임 근거 참조. ")
        evidence["explanation"] = (
            f"정면 박스 {stats['front_box_frames']}/{stats['selected_frames']}프레임"
            f"(실검출 {stats['front_real_box_frames']}); 거리 {span('front_distance_m')} m, "
            f"X {span('front_lateral_m')} m; FCW level/cause {stats['fcw_cause_frames']}, "
            f"창 내 보정 {segment_conf}. "
            f"거리 미산출 {stats['distance_unavailable_box_count']}/{stats['road_box_count']}실외 박스"
            f"(게이트 {stats['distance_unavailable_gate_counts']}), 실내 제외 {stats['interior_box_count']}박스; "
            + mechanism + "저장된 모든 경고가 L1 이하라 bin≥2 없음.")
        result["miss_evidence"] = evidence
    return result


def wilson_interval(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def video_bootstrap(pairs, repetitions, seed):
    """Resample clips, then ratio of summed numerators/denominators, never frames."""
    import numpy as np
    values = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    values = values[values[:, 1] > 0]
    if not len(values):
        return {"ci95": None, "resampling_clips": 0, "valid_repetitions": 0}
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = []
    for start in range(0, repetitions, 500):
        indices = rng.integers(0, len(values), size=(min(500, repetitions - start), len(values)))
        totals = values[indices].sum(axis=1)
        estimates.extend((totals[:, 0] / totals[:, 1]).tolist())
    return {"ci95": np.quantile(estimates, [0.025, 0.975], method="linear").tolist(),
            "resampling_clips": len(values), "valid_repetitions": len(estimates)}


def t46_summary(rows, repetitions, seed):
    completed = [r for r in rows if r["status"] == "evaluated"]
    excluded = [r for r in completed if r["outlier"]]
    selected = [r for r in completed if not r["outlier"]]
    summary = summarize(selected)
    summary.update(videos_total=len(rows), videos_completed=len(completed),
                   outliers=len(excluded), outlier_files=[r["file"] for r in excluded],
                   mixed_failure_count=sum(r["mixed_failure"] for r in excluded),
                   unevaluated_files=[r["file"] for r in rows if r["status"] != "evaluated"],
                   files=[r["file"] for r in selected])
    summary["miss_rate_ci95"] = wilson_interval(summary["missed_count"], summary["positive_evaluated"])
    intervals = {}
    for key, pool, numerator, denominator in (
        ("fp_ratio", selected, "fp_danger_frames", "fp_total_frames"),
        ("l1_negative_ratio", [r for r in selected if r["label"] == 0], "l1_frames", "l1_total_frames"),
        ("ldw_measurable_ratio", selected, "ldw_measurable_frames", "evaluated_frames"),
    ):
        n, d = sum(r[numerator] for r in pool), sum(r[denominator] for r in pool)
        summary[key] = n / d if d else None
        intervals[key] = video_bootstrap([(r[numerator], r[denominator]) for r in pool], repetitions, seed)
    leads = [r["window_danger_lead_sec"] for r in selected if r.get("window_danger_lead_sec") is not None]
    intervals["mean_window_danger_lead_sec"] = video_bootstrap([(x, 1) for x in leads], repetitions, seed)
    summary["bootstrap"] = intervals
    summary["ldw_measurable_frames"] = sum(r["ldw_measurable_frames"] for r in selected)
    summary["evaluated_frames"] = sum(r["evaluated_frames"] for r in selected)
    known = [r for r in selected if r["lane_trusted"] is not None]
    trusted = sum(r["lane_trusted"] for r in known)
    summary.update(lane_trusted_clips=trusted, lane_trust_known_clips=len(known),
                   lane_trust_unknown_clips=len(selected) - len(known),
                   lane_any_trusted_clips=sum(r["lane_any_trusted"] for r in known),
                   lane_trusted_ratio=trusted / len(known) if known else None,
                   lane_trusted_ratio_ci95=wilson_interval(trusted, len(known)))
    summary["negative_ldw_frames"] = {
        side: sum(r["ldw_warning_frames"].get(side, 0) for r in selected if r["label"] == 0)
        for side in ("left", "right", "unknown")}
    return summary


def t46_grouped(rows, repetitions, seed):
    return {name: t46_summary(group, repetitions, seed) for name, group in (
        ("all", rows), ("day", [r for r in rows if r["light"].lower() == "normal"]),
        ("night", [r for r in rows if r["light"].lower() == "dark"]),
        ("other_light", [r for r in rows if r["light"].lower() not in ("normal", "dark")]))}


def t46_table(sets, *, outlier_column="이상치(1개 이상 구간 failed)"):
    def percent(value):
        return "—" if value is None else f"{value * 100:.2f}%"
    def ci(interval, scale=100):
        return "—" if interval is None else f"[{interval[0] * scale:.2f}, {interval[1] * scale:.2f}]"
    def ratio(s, key, n, d):
        return f"{percent(s[key])} {ci(s['bootstrap'][key]['ci95'])} ({s[n]}/{s[d]})"
    lines = [f"| 세트 | 조명 | 평가/전체 영상 | {outlier_column} | 미탐/사고; 미탐률 [Wilson 95% CI] | 사고 전 3초 창 첫 위험 평균 선행 s [95% CI]; 검출 n | 사고 밖 위험 이상 % [95% CI] (프레임) | 정상 주의 % [95% CI] (프레임) | LDW 측정 가능 % [95% CI] (프레임) | 전 구간 차선 신뢰 클립/판정 가능 클립; 비율 [Wilson 95% CI] |",
             "|---|---|---:|---:|---|---|---|---|---|---|"]
    for name, groups in sets.items():
        for light, s in groups.items():
            label = {"all": "전체", "day": "낮", "night": "밤", "other_light": "기타(Twilight 등)"}[light]
            lead = s["mean_window_danger_lead_sec"]
            lead_text = "—" if lead is None else f"{lead:.3f}"
            lines.append(f"| {name} | {label} | {s['videos_evaluated']}/{s['videos_total']} | {s['outliers']} | "
                         f"{s['missed_count']}/{s['positive_evaluated']}; {percent(s['miss_rate'])} {ci(s['miss_rate_ci95'])} | "
                         f"{lead_text} {ci(s['bootstrap']['mean_window_danger_lead_sec']['ci95'], 1)}; n={s['window_danger_lead_count']} | "
                         f"{ratio(s, 'fp_ratio', 'fp_danger_frames', 'fp_total_frames')} | "
                         f"{ratio(s, 'l1_negative_ratio', 'l1_negative_frames', 'l1_negative_total_frames')} | "
                         f"{ratio(s, 'ldw_measurable_ratio', 'ldw_measurable_frames', 'evaluated_frames')} | "
                         f"{s['lane_trusted_clips']}/{s['lane_trust_known_clips']}; "
                         f"{percent(s['lane_trusted_ratio'])} {ci(s['lane_trusted_ratio_ci95'])} |")
    return "\n".join(lines) + "\n"


def t46_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_t46(args):
    """Produce the three t46 artifacts without touching source run folders."""
    import numpy as np
    if args.bootstrap_repetitions < 1000 or args.seed < 0:
        raise ValueError("bootstrap repetitions >= 1000, seed >= 0 필요")
    root_out = (ROOT / "runs_out/eval_t46").resolve()
    output = args.output.resolve()
    if output != root_out and root_out not in output.parents:
        raise ValueError("t46 출력은 runs_out/eval_t46 아래로 제한")
    versions = {"t34": ROOT / "runs_out/nexar_t34", "t40": ROOT / "runs_out/nexar_t40",
                "t44": ROOT / "runs_out/nexar_t44", "t48": args.runs, "test": args.test_runs}
    label_paths = {name: args.test_labels if name == "test" else args.labels for name in versions}
    inputs = {Path(__file__), ROOT / "src/roadscope/viz.py", ROOT / "docs/dev-log/DECISIONS.md",
              ROOT / "runs_in/testset/manifest.csv", ROOT / "runs_in/distill/manifest.csv"}
    inputs.update(ROOT / path for path in CODE_FILES.values())
    inputs.update(label_paths.values())
    for name, directory in versions.items():
        inputs.add(directory / "run_log.md")
        for label in read_labels(label_paths[name]):
            inputs.update(directory / Path(label["file"]).stem / f for f in ("meta.json", "frames.jsonl"))
    missing = sorted(str(p) for p in inputs if not p.is_file())
    if missing:
        raise ValueError(f"t46 입력 누락: {missing}")
    input_hashes = {str(p.resolve().relative_to(ROOT)).replace('\\', '/'): t46_hash(p)
                    for p in sorted(inputs)}
    def log_hashes(directory):
        text = (directory / "run_log.md").read_text(encoding="utf-8-sig")
        result = {}
        for filename in ("engine.py", "geom_car.py", "car_models.py", "mvp.py"):
            match = re.search(re.escape(filename) + r"['\"]?\s*:\s*[`'\"]?([0-9a-f]{12,64})", text)
            if not match:
                raise ValueError(f"{directory}: {filename} 실행 해시 없음")
            result[filename] = match[1]
        return result
    dev_hashes, test_hashes = log_hashes(args.runs), log_hashes(args.test_runs)
    matches = {name: a[:min(len(a), len(test_hashes[name]))] == test_hashes[name][:min(len(a), len(test_hashes[name]))]
               for name, a in dev_hashes.items()}
    if not all(matches.values()):
        raise ValueError("개발/시험 엔진 실행 해시 불일치")
    reports = {}
    for name, directory in versions.items():
        print(f"t46: {name} 저장 결과 평가", flush=True)
        reports[name] = evaluate(label_paths[name], directory, timelines=False, t46_metrics=True)
        if not reports[name]["complete"]:
            problems = [(r["file"], r["status"], r["detail"]) for r in reports[name]["videos"]
                        if r["status"] != "evaluated"]
            raise ValueError(f"t46 미완료 입력 {name}: {problems}")
    reps, seed = args.bootstrap_repetitions, args.seed
    sets = {"development": t46_grouped(reports["t48"]["videos"], reps, seed),
            "test": t46_grouped(reports["test"]["videos"], reps, seed)}
    outliers = {}
    for name, version in (("development", "t48"), ("test", "test")):
        rows = reports[version]["videos"]
        # Explicit inclusion here is for a separate outlier-only table, never the primary table.
        outliers[name] = t46_grouped([{**r, "outlier": False} for r in rows if r["outlier"]], reps, seed)
    historical = {name: t46_summary(reports[name]["videos"], reps, seed) for name in ("t34", "t40", "t44", "t48")}
    common = sorted(set.intersection(*(set(s["files"]) for s in historical.values())))
    fixed = {name: t46_summary([r for r in reports[name]["videos"] if r["file"] in common], reps, seed)
             for name in historical}
    sensitivity = {}
    for name, version in (("development", "t48"), ("test", "test")):
        sensitivity[name] = t46_summary([{**r, "outlier": r["outlier"] and not r["mixed_failure"]}
                                         for r in reports[version]["videos"]], reps, seed)
    diff_keys = ("videos_evaluated", "outliers", "positive_evaluated", "missed_count", "miss_rate",
                 "mean_window_danger_lead_sec", "window_danger_lead_count", "fp_ratio", "l1_negative_ratio",
                 "ldw_measurable_ratio", "lane_trusted_clips", "lane_trusted_ratio")
    differences = {group: {key: (sets['test'][group][key] - sets['development'][group][key]
                                if sets['test'][group][key] is not None and sets['development'][group][key] is not None
                                else None) for key in diff_keys} for group in sets["development"]}
    dev_files = {r["file"] for r in reports["t48"]["videos"]}
    test_files = {r["file"] for r in reports["test"]["videos"]}
    with (ROOT / "runs_in/distill/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        distill_files = {('pos_' if r['split'] == 'positive' else 'neg_') + r['file_name'] for r in csv.DictReader(stream)}
    with (ROOT / "runs_in/testset/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {r["file"]: r["path"] for r in csv.DictReader(stream)}
    if set(manifest) != test_files:
        raise ValueError("시험 manifest와 labels 파일 집합 불일치")
    source_mismatches = [r["file"] for r in reports["test"]["videos"]
                         if Path(r["source"]).name != r["file"]]
    provenance = {"development_run_hashes": dev_hashes, "test_run_hashes": test_hashes,
                  "hash_matches_at_recorded_precision": matches,
                  "current_code_sha256": {name: t46_hash(ROOT / CODE_FILES[name]) for name in dev_hashes},
                  "input_sha256": input_hashes, "test_original_sources": manifest,
                  "test_meta_source_filename_mismatches": source_mismatches,
                  "development_test_overlap": sorted(dev_files & test_files),
                  "test_distillation_overlap": sorted(test_files & distill_files)}
    if source_mismatches or dev_files & test_files or test_files & distill_files:
        raise ValueError("입력 출처 또는 데이터셋 중복 검증 실패")
    report = {"schema": "t46-evaluation-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "complete": True, "provenance": provenance,
              "methods": {"bootstrap_repetitions": reps, "seed": seed, "rng": "numpy.PCG64",
                          "numpy_version": np.__version__, "quantiles": [0.025, 0.975], "quantile_method": "linear",
                          "resampling": "uniform clips with replacement; n draws for n contributing clips; ratio of summed frame counts",
                          "outlier": "any car_calib segment horizon_conf == failed; entire clip excluded",
                          "lead": "mean over detected accident clips only, max(0,event-3)<=t<event; bin>=2",
                          "lane_trusted": "every car_calib segment has lane_trust.trusted == true",
                          "ldw_measurable": "sum(frame.lane_measurable == true) / all included stored frames",
                          "negative_ldw": "all normal frames with warnings.LDW.level>0, side field or Korean reason; any top channel/bin",
                          "base_definitions": reports["test"]["definitions"]},
              "sets": sets, "outlier_metrics": outliers, "test_minus_development": differences,
              "development_history": historical, "history_fixed_cohort": {"files": common, "summary": fixed},
              "all_segments_failed_only_sensitivity": sensitivity,
              "videos": {name: reports[version]["videos"] for name, version in (("development", "t48"), ("test", "test"))},
              "history_videos": {name: reports[name]["videos"] for name in ("t34", "t40", "t44")}}
    table = t46_table({"개발 t48": sets["development"], "시험 t45": sets["test"]})
    document = t46_markdown(report, table, args)
    changed = [name for name, sha in input_hashes.items() if t46_hash(ROOT / name) != sha]
    if changed:
        raise ValueError(f"평가 중 입력 변경: {changed}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "eval.md").write_text(document, encoding="utf-8")
    (output / "readme_table.md").write_text(table, encoding="utf-8")
    print(f"t46 완료: {output / 'eval.md'}", flush=True)


def t46_markdown(report, table, args):
    methods = report["methods"]
    lines = ["# t46 개발·시험 세트 평가", "", table.rstrip(), "", "## 계산 및 신뢰구간", "",
             "- 평가/전체 영상: 완료 결과 중 보정 실패 이상치를 제외한 영상 수 / 라벨 영상 수. 영상 수와 이상치 수는 관측 집계라 CI를 붙이지 않음.",
             "- 이상치: car_calib의 **어느 한 구간이라도** horizon_conf==failed이면 영상 전체를 본 지표에서 제외. 성공·실패 혼합 영상도 포함하며 아래에 별도 집계. horizon_ok나 수평선 높이만으로 제외하지 않음.",
             "- 사고 미탐: max(0, time_of_event−3) ≤ t_sec < time_of_event 창에 bin≥2가 없는 사고 영상. 미탐률 CI는 z=1.959963984540054의 Wilson score 95% 구간(연속성 보정 없음).",
             "- 선행: 같은 3초 창의 첫 bin≥2부터 사고 시각까지의 초. 검출된 사고 영상만 평균; 미탐에 0초를 대입하지 않음. CI도 검출 영상만 재표집한 평균의 percentile 구간.",
             "- 사고 밖 위험 이상: 정상 영상 전체 + 사고 영상 t_sec < time_of_event−3 중 bin≥2 프레임 수 / 해당 전체 프레임 수. 사고 후와 3초 창은 분모에서 제외. 실제 오경보 정답률이 아닌 대리지표.",
             "- 정상 주의: 정상 영상 전체에서 bin==1 프레임 수 / 정상 전체 프레임 수(L2/L3 제외).",
             "- LDW 측정 가능: 본 지표 영상의 전체 저장 프레임에서 lane_measurable==true의 비율. 차선 신뢰 클립: 모든 구간 lane_trust.trusted==true인 영상. 신뢰 필드 누락은 미상이며 거짓으로 대입하지 않음. 클립 수는 관측 집계, 그 비율은 Wilson CI.",
             f"- 영상 단위 부트스트랩: {methods['bootstrap_repetitions']:,}회, seed={methods['seed']}, numpy {methods['numpy_version']} PCG64. 각 세트·조명·지표별 같은 시드로 새 생성기를 초기화. 기여 영상 n개에서 n개를 균등 복원 추출하고 분자 합/분모 합을 다시 계산(영상별 비율 평균 아님). 분모 0인 영상은 해당 지표 재표집에서 제외. 2.5/97.5 백분위, linear 보간. 사고 밖 위험·정상 주의·LDW 비율에 적용.",
             "- 부트스트랩의 단위는 영상이며 프레임을 독립 표본으로 뽑지 않음. 선행의 단위도 영상. JSON의 bootstrap.resampling_clips와 valid_repetitions에 실제 표본 수·유효 반복 수를 저장.",
             "- 분모가 0이면 값/CI는 —(JSON null). 표본 1개 또는 관측값이 모두 같으면 부트스트랩 CI가 한 점으로 축소될 수 있으며 모집단 불확실성이 없다는 뜻은 아님. 개별 95% CI이며 다중비교 보정 없음.",
             "- Normal=낮, Dark=밤. Twilight와 다른 값은 기타로 보존. 전체에는 기타 포함. 출력 비율은 프레임 가중이며 표에서만 반올림. 저장 프레임을 보간하거나 새 판정을 하지 않음.",
             "- 영상들은 선택된 개발/시험 표본이다. CI는 이 영상들을 독립 재표집 단위로 취급하며 같은 주행·카메라의 잠재적 상관은 반영하지 않음. 과적합 해석과 규칙 변경 제안은 포함하지 않음.",
             "", "## 실행·출처 검증", "",
             "개발 t48과 시험 t45 run_log.md에 기록된 엔진 해시를 대조. 개발 기록은 앞 12자리이므로 전체 64자리 일치 증명은 아님.", "",
             "| 파일 | 개발 기록 | 시험 기록 | 기록된 정밀도 일치 |", "|---|---|---|---|"]
    p = report["provenance"]
    for name, value in p["development_run_hashes"].items():
        lines.append(f"| {name} | `{value}` | `{p['test_run_hashes'][name]}` | {p['hash_matches_at_recorded_precision'][name]} |")
    lines += ["", "meta.source는 시험 실행용 runs_out/t45_work/videos/<file> 경로. 원본 HF 캐시 경로는 runs_in/testset/manifest.csv에서 가져와 JSON에 보존. 영상 파일을 읽거나 GPU/엔진을 실행하지 않음.",
              f"라벨 파일 ID 중복: 개발∩시험 {len(p['development_test_overlap'])}개, 시험∩증류 {len(p['test_distillation_overlap'])}개. 파일 ID 대조이며 카메라/장면 독립성이나 영상 바이트 중복 검증은 아님.",
              "모든 라벨의 meta/frames 존재, 최종 기록 시각, n_frames, 시간 순서·bin, 사고 시각까지의 기록 범위를 기존 evaluator로 검증. 사용 입력 SHA256과 실행 중 불변 여부 확인. 출력 JSON provenance 참조.",
              "", "## 개발·시험 숫자 차이 (시험−개발)", "",
              "비율 차이는 퍼센트포인트(pp), 선행은 초, 나머지는 개수. 차이 자체의 CI나 통계적 유의성 검정은 계산하지 않음.", "",
              "| 지표 | 전체 차이 | 낮 차이 | 밤 차이 | 기타 차이 |", "|---|---:|---:|---:|---:|"]
    labels = {"videos_evaluated": "평가 영상", "outliers": "이상치", "positive_evaluated": "사고 영상", "missed_count": "미탐",
              "miss_rate": "미탐률 pp", "mean_window_danger_lead_sec": "창 내 첫 위험 평균 선행 s",
              "window_danger_lead_count": "창 내 검출 영상", "fp_ratio": "사고 밖 위험 이상 pp",
              "l1_negative_ratio": "정상 주의 pp", "ldw_measurable_ratio": "LDW 측정 가능 pp",
              "lane_trusted_clips": "전 구간 차선 신뢰 클립", "lane_trusted_ratio": "전 구간 차선 신뢰 pp"}
    for key, title in labels.items():
        values = []
        for group in ("all", "day", "night", "other_light"):
            value = report["test_minus_development"][group][key]
            values.append("—" if value is None else f"{value * (100 if title.endswith('pp') else 1):+.3f}")
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    lines += ["", "## 개발 세트 버전 변화", "",
              "각 실행에서 같은 보정 실패 기준을 적용. 정상 LDW는 warnings.LDW.level>0인 전체 정상 프레임(대표 경고/최종 bin과 무관); side가 없으면 저장된 좌측/우측 사유를 사용.", ""]
    def history_table(history):
        result = ["| 실행 | 평가/전체 | 이상치 | 미탐/사고; Wilson 95% CI % | 사고 밖 위험 이상 % [95% CI] (프레임) | 정상 LDW 좌/우/미상 |", "|---|---:|---:|---|---|---|"]
        for version, s in history.items():
            interval, fpci = s["miss_rate_ci95"], s["bootstrap"]["fp_ratio"]["ci95"]
            ldw = s["negative_ldw_frames"]
            result.append(f"| {version} | {s['videos_evaluated']}/{s['videos_total']} | {s['outliers']} | "
                          f"{s['missed_count']}/{s['positive_evaluated']}; [{interval[0]*100:.2f}, {interval[1]*100:.2f}] | "
                          f"{s['fp_ratio']*100:.2f} [{fpci[0]*100:.2f}, {fpci[1]*100:.2f}] ({s['fp_danger_frames']}/{s['fp_total_frames']}) | "
                          f"{ldw['left']}/{ldw['right']}/{ldw['unknown']} |")
        return result
    lines += history_table(report["development_history"])
    cohort = report["history_fixed_cohort"]
    same = all(set(s["files"]) == set(cohort["files"]) for s in report["development_history"].values())
    lines += ["", f"네 실행 모두 평가 가능한 고정 교집합: {len(cohort['files'])}개. 각 실행 평가 집합과 동일: {same}. 고정 교집합 수치는 JSON history_fixed_cohort에 별도 저장."]
    if not same:
        lines += ["", "고정 교집합 비교:", ""] + history_table(cohort["summary"])
    lines += ["", "## 이상치 별도 집계", "",
              "아래는 본 지표에서 제외된 영상만 집계한 참고 수치. 이 표의 평가/전체는 이상치 집합 내 수이며 중복 제외는 하지 않음.", "",
              t46_table({"개발 이상치만": report["outlier_metrics"]["development"], "시험 이상치만": report["outlier_metrics"]["test"]}, outlier_column="추가 제외").rstrip(),
              "", "| 세트 | 영상 | 조명 | 사고 라벨 | 실패 구간 | 성공·실패 혼합 | 창 내 미탐 |", "|---|---|---|---:|---|---|---|"]
    for name, rows in report["videos"].items():
        for r in rows:
            if r["outlier"]:
                lines.append(f"| {name} | {r['file']} | {r['light']} | {r['label']} | {r['failed_segments']} | {r['mixed_failure']} | {r['missed']} |")
    lines += ["", "### 혼합 구간 처리에 따른 숫자 차이", "",
              "주 지표는 어느 구간이든 failed인 영상을 제외. 아래는 모든 구간이 failed인 영상만 제외했을 때의 참고 집계로, 본 지표를 대체하지 않음.", "",
              t46_table({"개발: 전 구간 failed만 제외": {"all": report["all_segments_failed_only_sensitivity"]["development"]},
                         "시험: 전 구간 failed만 제외": {"all": report["all_segments_failed_only_sensitivity"]["test"]}}, outlier_column="이상치(전 구간 failed)").rstrip(),
              "", "## 시험 세트 미탐의 사고 전 3초 기록", "",
              "본 지표 미탐과 제외된 이상치의 미탐을 모두 구별해 기록. t25 형식의 박스·거리·X·FCW cause·보정 근거이며 실제 사고 당사자나 거리 오차의 원인은 로그만으로 확정하지 않음. 프레임별 객체(실내 별도 보존)와 채널 등급은 JSON miss_evidence 참조. 규칙 수정 제안 없음.", ""]
    for r in report["videos"]["test"]:
        if r["missed"]:
            e = r["miss_evidence"]
            kind = "이상치 제외" if r["outlier"] else "본 지표"
            lines.append(f"- **{r['file']} ({kind}, {r['light']}, [{e['window'][0]:.3f}, {e['window'][1]:.3f}) s)**: {e['explanation']}")
    lines += ["", "## 재현", "", "```powershell", "$env:CUDA_VISIBLE_DEVICES='-1'",
              f"python -B tools/evaluate/eval_nexar.py --t46 --labels {args.labels.as_posix()} --runs {args.runs.as_posix()} --test-labels {args.test_labels.as_posix()} --test-runs {args.test_runs.as_posix()} --output {args.output.as_posix()} --bootstrap-repetitions {args.bootstrap_repetitions} --seed {args.seed}", "```",
              "", "--t46은 기존 실행 폴더에 timeline/eval을 쓰지 않고 지정된 eval_t46 폴더에 세 파일만 생성. 모델 추론·GPU·학습·엔진 규칙 변경 없음."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "runs_in/nexar/labels.csv")
    parser.add_argument("--runs", type=Path, help="기본 nexar; --t46에서는 nexar_t48")
    parser.add_argument("--baseline-runs", type=Path, default=ROOT / "runs_out/nexar", help="기존 10개 비교 결과 폴더")
    parser.add_argument("--middle-runs", type=Path, default=ROOT / "runs_out/nexar_t20", help="TTC 판정 포함 중간 결과 폴더")
    parser.add_argument("--compare-runs", type=Path, help="30개 전체 비교 폴더; nexar_t29 평가 시 기본 nexar_t24")
    parser.add_argument("--no-timelines", action="store_true", help="수치만 계산하고 PNG 생성 생략")
    parser.add_argument("--stdout", action="store_true", help="파일/PNG를 쓰지 않고 Markdown만 출력")
    parser.add_argument("--t46", action="store_true", help="개발/시험 CI 및 버전 비교를 eval_t46에 출력")
    parser.add_argument("--test-labels", type=Path, default=ROOT / "runs_in/testset/labels.csv")
    parser.add_argument("--test-runs", type=Path, default=ROOT / "runs_out/testset_t45")
    parser.add_argument("--output", type=Path, default=ROOT / "runs_out/eval_t46")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.runs is None:
        args.runs = ROOT / ("runs_out/nexar_t48" if args.t46 else "runs_out/nexar")
    if args.t46:
        if args.stdout:
            parser.error("--t46은 세 파일 출력 모드이며 --stdout과 병용 불가")
        evaluate_t46(args)
        return
    compare_runs = args.compare_runs
    if compare_runs is None and args.runs.name == "nexar_t29":
        compare_runs = args.runs.parent / "nexar_t24"
    report = evaluate(args.labels, args.runs, timelines=not (args.no_timelines or args.stdout),
                      baseline_runs=None if compare_runs else args.baseline_runs,
                      middle_runs=None if compare_runs else args.middle_runs, detailed=True, compare_runs=compare_runs)
    if args.stdout:
        print(markdown(report))
        return
    args.runs.mkdir(parents=True, exist_ok=True)
    (args.runs / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.runs / "eval.md").write_text(markdown(report), encoding="utf-8")
    print(f"{'완료' if report['complete'] else '부분 결과'}: {args.runs / 'eval.md'}")


if __name__ == "__main__":
    main()
