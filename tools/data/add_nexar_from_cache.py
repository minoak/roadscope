"""Add the t21 Nexar selection from local Hugging Face snapshots, without network access.

Existing labels/videos are preserved. Missing cache entries are reported before any writes.
Run again safely: python tools/data/add_nexar_from_cache.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SELECTION = {
    "negative": "01211 01489 01539 01584 01711 01811 01966".split(),
    "positive": "00004 00071 00142 00310 00602 00655 00754 00845 00864 00872 00877 00896 01019".split(),
}
FIELDS = ["file", "label", "time_of_event", "time_of_alert", "light", "weather", "scene"]


def add_from_cache(cache, output):
    snapshots = Path(cache) / "snapshots"
    roots = sorted(snapshots.iterdir()) if snapshots.is_dir() else []
    labels = Path(output) / "labels.csv"
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected labels.csv columns; existing schema must be preserved")
        rows = list(reader)
    existing = {row["file"]: row for row in rows}
    if len(existing) != len(rows):
        raise ValueError("Duplicate filenames in labels.csv")

    planned, missing = [], []
    for split, ids in SELECTION.items():
        candidates = []
        for root in roots:
            directory = root / "train" / split
            metadata = directory / "metadata.csv"
            if metadata.is_file():
                with metadata.open(encoding="utf-8-sig", newline="") as stream:
                    candidates.append((directory, {r["file_name"]: r for r in csv.DictReader(stream)}))
        for video_id in ids:
            filename = f"{video_id}.mp4"
            match = next(((directory / filename, meta[filename]) for directory, meta in candidates
                          if filename in meta and (directory / filename).is_file()), None)
            if match is None:
                missing.append(f"train/{split}/{filename} (video or metadata row missing)")
                continue
            source, meta = match
            row = {"file": f"{split[:3]}_{filename}", "label": str(int(split == "positive")),
                   "time_of_event": meta["time_of_event"], "time_of_alert": meta["time_of_alert"],
                   "light": meta["light_conditions"], "weather": meta["weather"], "scene": meta["scene"]}
            if row["file"] in existing and existing[row["file"]] != row:
                raise ValueError(f"Existing label differs from cache: {row['file']}")
            destination = Path(output) / row["file"]
            if destination.exists() and destination.stat().st_size != source.stat().st_size:
                raise ValueError(f"Existing video has a different size: {destination}")
            planned.append((source, destination, row))
    if missing:
        return {"missing": missing}

    copied = added = 0
    for source, destination, row in planned:
        if not destination.exists():
            # Never expose an interrupted copy as a complete video.
            temporary = destination.with_suffix(".mp4.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
            copied += 1
        if row["file"] not in existing:
            rows.append(row)
            added += 1
    if added:
        temporary = labels.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(labels)
    return {"copied": copied, "labels_added": added, "labels_total": len(rows), "missing": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/huggingface/hub/datasets--nexar-ai--nexar_collision_prediction")
    parser.add_argument("--out", type=Path, default=ROOT / "runs_in/nexar")
    args = parser.parse_args()
    result = add_from_cache(args.cache, args.out)
    if result["missing"]:
        print("\n".join(result["missing"]))
        return 1
    print(f"Copied {result['copied']}; added {result['labels_added']} labels; total {result['labels_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
