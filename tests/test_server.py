"""CPU-only contract checks. No test invokes an engine or loads a model."""
import json
import os
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ROADSCOPE_DISABLE_ENGINE"] = "1"

from fastapi.testclient import TestClient
from web import server
from web.summary import build_summary, camera_summary, display_records, warning_events


class WebTests(unittest.TestCase):
    def setUp(self):
        # Normal inherited ACLs: mkdtemp's Windows 0700 directory is inaccessible
        # to the managed sandbox token on this workstation.
        self.root = Path(__file__).resolve().parents[1] / "web" / ".data" / "tests" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.data_patch = patch.object(server, "DATA", self.root / "data")
        self.data_patch.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close()
        self.data_patch.stop()
        shutil.rmtree(self.root)

    def fixture(self):
        run = self.root / "run"
        run.mkdir()
        meta = {"vehicle": "car", "fps_src": 10, "size": [1280, 720], "n_frames": 6,
                "car_calib": [{"segment": 0, "end_frame": 59, "horizon_conf": "high",
                               "lane_baseline": {"baseline_m": 0.0, "ldw_shift_px": 0}},
                              {"segment": 1, "horizon_conf": "failed"}]}
        records = []
        for t, level in enumerate([0, 2, 3, 1, 1, 0]):
            records.append({"t_sec": float(t), "bin": level, "top_reason": f"peak {t}",
                            "warnings": {"FCW": {"level": level, "track_id": 1, "reason": f"distance {t}"}}})
        (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (run / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return run

    def test_intervals_peak_and_failed_segment(self):
        summary = build_summary(self.fixture())
        self.assertEqual(summary["duration"], 6)
        self.assertEqual(summary["danger"], {"count": 1, "duration": 2, "intervals": [{"start": 1, "end": 3}]})
        self.assertEqual(summary["caution"]["duration"], 2)
        self.assertEqual(summary["peak"]["time"], 2)
        self.assertTrue(summary["confidence"]["low"])
        self.assertEqual(summary["cameras"][0]["lane_baseline_status"], "applied")
        self.assertEqual(summary["cameras"][1]["lane_baseline_status"], "not_recorded")
        self.assertEqual([e["time"] for e in summary["events"]], [1, 2, 3, 5])

    def test_duration_and_unmeasured_start(self):
        run = self.fixture()
        records = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()][1:]
        (run / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        summary = build_summary(run, 6.25)
        self.assertEqual(summary["unevaluated_duration"], 1)
        self.assertEqual(summary["duration"], 6.25)
        self.assertEqual(summary["duration_source"], "video")

    def test_lane_zero_failure_is_not_legacy_not_applied(self):
        baselines = [
            {"baseline_m": None, "ldw_measurable": False},
            {"baseline_m": 0., "ldw_measurable": True},
            {"baseline_m": None, "reason": "legacy insufficient evidence"},
            {},
        ]
        cameras, _ = camera_summary({"car_calib": [{"lane_baseline": b} for b in baselines]}, [])
        self.assertEqual([c['lane_baseline_status'] for c in cameras],
                         ['failed', 'applied', 'not_applied', 'not_recorded'])
        self.assertEqual([c['ldw_measurable'] for c in cameras], [False, True, None, None])

    def test_web_scratch_is_writable_and_cleans_after_failure(self):
        from web.media import job_scratch, prepare_upload
        import tempfile
        from roadscope import viz
        with self.assertRaisesRegex(ValueError, 'simulated conversion failure'):
            with job_scratch(prefix='upload-', dir=self.root) as name:
                scratch = Path(name)
                (scratch / 'prepared.mp4').write_bytes(b'test')
                self.assertEqual(scratch.parent, self.root.resolve())
                raise ValueError('simulated conversion failure')
        self.assertFalse(scratch.exists())
        self.assertIs(viz.tempfile, tempfile)
        if os.name == 'nt':
            self.assertIsNot(prepare_upload.__globals__['tempfile'], tempfile)
            self.assertIs(viz.prepare_upload.__globals__['tempfile'], tempfile)

    @staticmethod
    def warning_row(t, level, side=None, name="FCW"):
        return {"t_sec": t, "bin": level, "warnings": {
            name: {"level": level, "side": side, "reason": f"grade {level}"}}}

    def test_display_gap_boundary_and_raw_preservation(self):
        for gap, expected in [(0.499, 3), (0.5, 1), (0.501, 1)]:
            with self.subTest(gap=gap):
                rows = [self.warning_row(0, 3), self.warning_row(0.1, 1),
                        self.warning_row(0.1 + gap, 3)]
                before = json.dumps(rows)
                displayed = display_records(rows)
                self.assertEqual(displayed[1]["bin"], expected)
                self.assertEqual(json.dumps(rows), before)
        # 소수 뺄셈 오차로 정확히 0.5초인 간격을 합치면 안 된다.
        rows = [self.warning_row(0, 2), self.warning_row(0.3, 0), self.warning_row(0.8, 2)]
        self.assertEqual(display_records(rows)[1]["bin"], 0)

    def test_display_keeps_direction_type_and_sustained_changes(self):
        rows = [self.warning_row(0, 2, "left", "BSW"),
                self.warning_row(0.1, 1, "right", "BSW"),
                self.warning_row(0.2, 2, "left", "BSW")]
        self.assertEqual(display_records(rows)[1]["bin"], 1)
        rows = [self.warning_row(0, 2), self.warning_row(0.1, 1), self.warning_row(0.8, 2)]
        self.assertEqual([e["grade"] for e in warning_events(display_records(rows))], [2, 1, 2])
        rows = [self.warning_row(0, 2), self.warning_row(0.1, 0),
                self.warning_row(0.2, 2, "right", "BSW")]
        self.assertEqual(display_records(rows)[1]["bin"], 0)

    def test_display_repeated_flicker_keeps_critical_and_real_release(self):
        rows = [self.warning_row(t, grade) for t, grade in
                [(0, 1), (1, 3), (1.1, 1), (1.2, 3), (1.3, 0), (1.6, 3), (2, 0), (3, 1)]]
        for i, row in enumerate(rows):
            row["warnings"]["FCW"]["track_id"] = i
        displayed = display_records(rows)
        self.assertEqual([(e["time"], e["grade"]) for e in warning_events(displayed)],
                         [(0, 1), (1, 3), (2, 0), (3, 1)])
        isolated = [self.warning_row(0, 1), self.warning_row(1, 3), self.warning_row(1.1, 1)]
        self.assertEqual([r["bin"] for r in display_records(isolated)], [1, 3, 1])
        self.assertEqual(display_records([{"t_sec": 0, "bin": 0}])[0]["bin"], 0)

    def test_summary_display_counts_keep_raw_timeline_and_duration(self):
        run = self.fixture()
        rows = [self.warning_row(t, grade) for t, grade in
                [(0, 1), (1, 2), (1.1, 1), (1.2, 3), (1.3, 0), (1.6, 2), (2, 0)]]
        path = run / "frames.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        before = path.read_bytes()
        summary = build_summary(run, 3)
        self.assertEqual(summary["danger"]["count"], 1)
        self.assertEqual(summary["caution"]["count"], 1)
        self.assertAlmostEqual(summary["danger"]["duration"], 0.6)
        self.assertAlmostEqual(summary["caution"]["duration"], 1.1)
        self.assertEqual([s["grade"] for s in summary["segments"]], [r["bin"] for r in rows])
        self.assertEqual(summary["peak"]["time"], 1.2)
        self.assertEqual(path.read_bytes(), before)

    def test_cpu_guard_and_dev_path(self):
        with patch.object(server.EXECUTOR, "submit") as submit:
            response = self.client.post('/api/jobs', data={"vehicle": "car"}, files={"video": ("x.mp4", b"video")})
            self.assertEqual(response.status_code, 503)
            submit.assert_not_called()
        with patch.dict(os.environ, {"ROADSCOPE_DEV": "1"}):
            self.assertEqual(self.client.get('/dev/load?run=../../app.py').status_code, 404)
        self.assertEqual(self.client.get('/api/jobs/invalid').status_code, 404)
        with self.assertRaises(RuntimeError):
            server.run_engine("x.mp4", "car", self.root)

    def test_upload_validation_and_queue_only(self):
        with patch.dict(os.environ, {"ROADSCOPE_DISABLE_ENGINE": "0"}), patch.object(server.EXECUTOR, "submit") as submit:
            def post(name, content, vehicle="car"):
                return self.client.post('/api/jobs', data={"vehicle": vehicle}, files={"video": (name, content)})
            self.assertEqual(post("x.exe", b"x").status_code, 415)
            self.assertEqual(post("x.mp4", b"").status_code, 422)
            self.assertEqual(post("x.mp4", b"x", "bus").status_code, 422)
            with patch.object(server, "MAX_UPLOAD", 3):
                self.assertEqual(post("x.mp4", b"1234").status_code, 413)
            submit.assert_not_called()
            response = post("../clip.mp4", b"queue-only fixture")
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["id"]
            submit.assert_called_once_with(server.run_job, job_id)
            job = self.client.get('/api/jobs/' + job_id).json()
            self.assertEqual(job["filename"], "clip.mp4")
            self.assertNotIn("upload", job)
            self.assertEqual(job["status"], "queued")
            server.update_job(job_id, session="old-session")
            self.assertEqual(self.client.get('/api/jobs/' + job_id).json()["status"], "interrupted")

    def test_incremental_progress_ignores_partial_line(self):
        path = self.root / "frames.jsonl"
        path.write_bytes(b'{"t_sec":0}\n{"t_sec":')
        self.assertEqual(server.frame_progress(path), (1, 0))
        with path.open('ab') as stream:
            stream.write(b'1}\n')
        self.assertEqual(server.frame_progress(path), (2, 1))
        self.assertEqual(server.frame_progress(path), (2, 1))

    def test_done_reload_download_and_ranges(self):
        job_id = "a" * 32
        directory = server.job_dir(job_id)
        directory.mkdir(parents=True)
        media = directory / "hud.mp4"
        media.write_bytes(b"0123456789")
        server.write_json(directory / "result.json", {"duration": 6})
        server.write_json(directory / "job.json", {"id": job_id, "session": "old", "status": "done", "kind": "saved",
                          "filename": "clip.mp4", "vehicle": "car", "files": {"hud": str(media)}})
        self.assertEqual(self.client.get('/api/jobs/' + job_id).json()["status"], "done")
        url = f'/api/jobs/{job_id}/files/hud'
        response = self.client.get(url, headers={"Range": "bytes=2-5"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, b"2345")
        self.assertIn('attachment', self.client.get(url + '?attachment=true').headers['content-disposition'])
        self.assertEqual(self.client.get(f'/api/jobs/{job_id}/files/worker.log').status_code, 404)

    def test_engine_cli_contract_without_execution(self):
        car = server.engine_command(Path("job_abc.mp4"), "car")
        moto = server.engine_command(Path("job_abc.mp4"), "moto")
        self.assertNotIn("--fps", car)
        self.assertEqual(moto[-2:], ["--fps", "5"])
        self.assertEqual(Path(car[car.index("--out") + 1]) / "job_abc", server.DATA / "results" / "job_abc")


if __name__ == "__main__":
    unittest.main()
