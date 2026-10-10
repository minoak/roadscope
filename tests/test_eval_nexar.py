"""실제 엔진 없이 평가 경계·부분 결과·PNG를 검증한다.

python -m unittest tests.test_eval_nexar -v
임시 파일은 작업 폴더 안에 만들고 테스트 후 정리한다.
"""
import csv
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

from tools.evaluate import eval_nexar as evaluation
from roadscope.viz import render_timeline


def records(points):
    return [{"frame_idx": index, "t_sec": t, "bin": level, "warnings": {}}
            for index, (t, level) in enumerate(points)]


def label(name="pos.mp4", event=10, alert=8, positive=1, light="Dark"):
    return {"file": name, "label": positive, "time_of_event": event,
            "time_of_alert": alert, "light": light}


class MetricsTest(unittest.TestCase):
    def test_boundaries_and_first_warning(self):
        value = evaluation.frame_metrics(label(), records([(0, 0), (1, 1), (6.999, 2), (7, 3), (9.999, 2), (10, 3)]))
        self.assertEqual(value["first_danger_sec"], 6.999)
        self.assertEqual(value["first_caution_sec"], 1)
        self.assertAlmostEqual(value["danger_lead_sec"], 3.001)
        self.assertAlmostEqual(value["danger_alert_delta_sec"], 1.001)
        self.assertFalse(value["missed"])
        self.assertEqual(value["pre_event_danger_frames"], 2)
        self.assertEqual(value["fp_total_frames"], 3)
        self.assertAlmostEqual(value["fp_ratio"], 1 / 3)

    def test_event_exact_and_after_do_not_rescue_miss(self):
        value = evaluation.frame_metrics(label(), records([(0, 0), (7, 1), (10, 3), (11, 3)]))
        self.assertTrue(value["missed"])
        self.assertIsNone(value["first_danger_sec"])
        self.assertEqual(value["first_caution_sec"], 7)

    def test_early_warning_can_still_be_miss(self):
        value = evaluation.frame_metrics(label(), records([(1, 2), (7, 0), (9, 1)]))
        self.assertTrue(value["missed"])
        self.assertEqual(value["danger_lead_sec"], 9)
        self.assertEqual(value["fp_ratio"], 1)

    def test_late_alert_and_window_start(self):
        value = evaluation.frame_metrics(label(alert=6), records([(0, 0), (7, 2), (9, 0)]))
        self.assertFalse(value["missed"])
        self.assertEqual(value["danger_alert_delta_sec"], -1)
        self.assertEqual(value["fp_ratio"], 0)

    def test_short_positive_has_no_baseline(self):
        value = evaluation.frame_metrics(label(event=2, alert=1), records([(0, 2), (1, 0), (2, 3)]))
        self.assertIsNone(value["fp_ratio"])
        self.assertFalse(value["missed"])

    def test_normal_all_frames_in_denominator(self):
        value = evaluation.frame_metrics(label(event=None, alert=None, positive=0), records([(0, 3), (1, 0), (2, 2)]))
        self.assertAlmostEqual(value["fp_ratio"], 2 / 3)
        self.assertIsNone(value["missed"])
        self.assertIsNone(value["first_danger_sec"])

    def test_weighted_summary_and_null_average(self):
        rows = [{**label(), **evaluation.frame_metrics(label(), records([(0, 2), (5, 0), (8, 0)])), "status": "evaluated"}]
        normal = label("neg.mp4", None, None, 0)
        rows.append({**normal, **evaluation.frame_metrics(normal, records([(t, 0) for t in range(8)])), "status": "evaluated"})
        rows.append({**label("missing.mp4"), "status": "missing"})
        value = evaluation.summarize(rows)
        self.assertAlmostEqual(value["fp_ratio"], 0.1)
        self.assertEqual(value["positive_evaluated"], 1)
        self.assertEqual(value["positive_total"], 2)
        self.assertEqual(value["mean_danger_lead_sec"], 10)
        empty = evaluation.summarize([])
        self.assertIsNone(empty["fp_ratio"])
        self.assertIsNone(empty["mean_danger_lead_sec"])


class FilesTest(unittest.TestCase):
    def setUp(self):
        base = evaluation.ROOT / "runs_out"
        base.mkdir(exist_ok=True)
        # Windows 제한 토큰에서 tempfile의 mode=0700 ACL 대신 부모 권한을 상속한다.
        self.directory = (base / ("nexar-test-" + uuid.uuid4().hex)).resolve()
        assert evaluation.ROOT in self.directory.parents
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)
        self.labels = self.directory / "labels.csv"
        self.runs = self.directory / "runs"
        self.runs.mkdir()

    def cleanup_directory(self):
        target = self.directory.resolve()
        assert target.parent == (evaluation.ROOT / "runs_out").resolve()
        assert target.name.startswith("nexar-test-")
        shutil.rmtree(target)

    def write_labels(self, rows):
        with self.labels.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_run(self, name, points, meta=True):
        directory = self.runs / Path(name).stem
        directory.mkdir(exist_ok=True)
        values = records(points)
        (directory / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values), encoding="utf-8")
        if meta:
            (directory / "meta.json").write_text(json.dumps({"n_frames": len(values), "fps_eval": 1}), encoding="utf-8")
        return directory

    def test_missing_pending_invalid_are_not_misses(self):
        names = ["absent.mp4", "pending.mp4", "count.mp4", "short.mp4", "bad.mp4", "stale.mp4", "pass2.mp4"]
        self.write_labels([label(name) for name in names])
        self.write_run("pending.mp4", [(0, 0)], meta=False)
        count = self.write_run("count.mp4", [(0, 0), (10, 0)])
        (count / "meta.json").write_text('{"n_frames": 9, "fps_eval": 1}')
        self.write_run("short.mp4", [(0, 0)])
        self.write_run("bad.mp4", [(0, 0), (0, 0)])
        stale = self.write_run("stale.mp4", [(0, 0), (10, 0)])
        old = (stale / "frames.jsonl").stat().st_mtime_ns - 1_000_000_000
        os.utime(stale / "meta.json", ns=(old, old))
        pass2 = self.write_run("pass2.mp4", [(0, 0), (10, 0)])
        (pass2 / "frames.jsonl.tmp").write_text("second pass")
        result = evaluation.evaluate(self.labels, self.runs, timelines=False)
        self.assertEqual([r["status"] for r in result["videos"]],
                         ["missing", "pending", "pending", "invalid", "invalid", "pending", "pending"])
        self.assertEqual(result["summary"]["all"]["positive_evaluated"], 0)
        self.assertIsNone(result["summary"]["all"]["miss_rate"])
        self.assertIn("미확정", evaluation.markdown(result))

    def test_labels_validation(self):
        for rows in ([label("../escape.mp4")], [label(), label()],
                     [label(event=float("nan"))], [label(alert=None)], [label(alert=11)]):
            with self.subTest(rows=rows):
                self.write_labels(rows)
                with self.assertRaises(ValueError):
                    evaluation.read_labels(self.labels)

    def test_changed_snapshot_is_pending(self):
        self.write_labels([label()])
        directory = self.write_run("pos.mp4", [(0, 0), (7, 2), (10, 0)])
        original = evaluation.load_records
        def changed(path):
            values = original(path)
            (directory / "meta.json").write_text('{"n_frames": 3, "fps_eval": 1, "changed": true}')
            return values
        with patch.object(evaluation, "load_records", side_effect=changed):
            report = evaluation.evaluate(self.labels, self.runs, timelines=False)
        self.assertEqual(report["videos"][0]["status"], "pending")
        self.assertNotIn("missed", report["videos"][0])

    def test_timeline_markers_and_default_compatibility(self):
        directory = self.write_run("pos.mp4", [(t, 2 if t >= 8 else 0) for t in range(12)])
        from matplotlib.axes import Axes
        original = Axes.axvline
        calls = []
        def record_line(ax, x=0, *args, **kwargs):
            calls.append((x, kwargs.get("color")))
            return original(ax, x, *args, **kwargs)
        with patch.object(Axes, "axvline", record_line):
            render_timeline(directory, directory / "default.png")
            self.assertEqual(calls, [])
            render_timeline(directory, directory / "marked.png", time_of_event=10, time_of_alert=8)
        self.assertEqual(calls, [(10, "#FF3838")] * 2 + [(8, "#FFA500")] * 2)
        with Image.open(directory / "marked.png") as image:
            pixels = np.asarray(image.convert("RGB"))
            self.assertGreater(np.all(pixels == [255, 56, 56], axis=2).sum(), 100)
            self.assertGreater(np.all(pixels == [255, 165, 0], axis=2).sum(), 100)

    def test_ten_video_report_and_pngs(self):
        rows = [label(f"pos_{i}.mp4", light="Dark" if i < 3 else "Normal") for i in range(7)]
        rows += [label(f"neg_{i}.mp4", None, None, 0, "Dark" if i == 0 else "Normal") for i in range(3)]
        self.write_labels(rows)
        for i, row in enumerate(rows):
            points = [(t, 2 if (i in (0, 3, 4, 5) and t == 8) or (i >= 7 and t == 0) else 0) for t in range(12)]
            self.write_run(row["file"], points)
        with patch("sys.argv", ["eval_nexar.py", "--labels", str(self.labels), "--runs", str(self.runs)]):
            evaluation.main()
        report = json.loads((self.runs / "eval.json").read_text(encoding="utf-8"))
        self.assertTrue(report["complete"])
        self.assertEqual(report["summary"]["all"]["missed_count"], 3)
        self.assertEqual(report["summary"]["night"]["missed_count"], 2)
        self.assertEqual(report["summary"]["day"]["missed_count"], 1)
        self.assertEqual(report["summary"]["all"]["mean_danger_lead_sec"], 2)
        self.assertAlmostEqual(report["summary"]["all"]["fp_ratio"], 3 / 85)
        self.assertIn("3/7", (self.runs / "eval.md").read_text(encoding="utf-8"))
        self.assertEqual(len(list(self.runs.glob("*/timeline_eval.png"))), 10)


if __name__ == "__main__":
    unittest.main()
