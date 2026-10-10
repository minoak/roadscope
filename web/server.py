# SPDX-License-Identifier: AGPL-3.0-only
"""Run from the project root: python -m uvicorn web.server:app --host 127.0.0.1 --port 7861.

The worker keeps the durable, single-worker pipeline of the earlier Gradio app
(not included in this repository) and reuses roadscope.viz's upload/HUD/H.264
helpers. Importing this module never imports an engine/model. The engine runs as
a subprocess (python -m roadscope.engine), so roadscope must be importable:
pip install -e . or PYTHONPATH=src.
ROADSCOPE_DEV=1 enables local saved-run loading; ROADSCOPE_DISABLE_ENGINE=1
rejects uploads and engine invocation (use both for CPU-only UI verification).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .summary import build_summary, read_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / ".data"
STATIC = ROOT / "web" / "static"
SESSION = uuid.uuid4().hex
MAX_UPLOAD = 500 * 1024 * 1024
STAGES = ("영상 확인", "카메라 측정", "장면 판정", "결과 영상 만들기")
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-analysis")
DEV_LOCK = threading.Lock()
COUNTS = {}
COUNT_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    yield
    EXECUTOR.shutdown(wait=False)


app = FastAPI(title="RoadScope", lifespan=lifespan)


def job_dir(job_id):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return DATA / "jobs" / job_id


def write_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_job(job_id):
    try:
        job = read_json(job_dir(job_id) / "job.json")
    except FileNotFoundError:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    if job["status"] not in ("done", "error") and job["session"] != SESSION:
        job.update(status="interrupted", error="서버가 재시작되어 분석이 중단됐습니다. 영상을 다시 올려 주세요.")
    return job


def update_job(job_id, **changes):
    path = job_dir(job_id) / "job.json"
    job = read_json(path)
    job.update(changes, updated_at=time.time())
    write_json(path, job)
    return job


def frame_progress(path):
    # Bound polling I/O and count only complete records.
    with COUNT_LOCK:
        if not path.is_file():
            return 0, 0
        offset, count, last_t, pending = COUNTS.get(str(path), (0, 0, 0, b""))
        if path.stat().st_size < offset:
            offset, count, last_t, pending = 0, 0, 0, b""
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(1024 * 1024)
        lines = (pending + chunk).split(b"\n")
        count += len(lines) - 1
        if len(lines) > 1:
            last_t = json.loads(lines[-2])["t_sec"]
        COUNTS[str(path)] = (offset + len(chunk), count, last_t, lines[-1])
        return count, last_t


def engine_command(source, vehicle):
    command = [sys.executable, "-u", "-m", "roadscope.engine", str(source), "--vehicle", vehicle,
               "--out", str(DATA / "results")]
    if vehicle == "moto":
        command += ["--fps", "5"]
    return command


def run_engine(source, vehicle, run_dir):
    if os.environ.get("ROADSCOPE_DISABLE_ENGINE") == "1":
        raise RuntimeError("현재 서버는 결과 조회 전용입니다. 엔진 실행이 차단돼 있습니다.")
    with (run_dir / "engine.log").open("w", encoding="utf-8") as log:
        subprocess.run(engine_command(source, vehicle), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def run_job(job_id):
    directory = job_dir(job_id)
    try:
        # Lazy imports: result browsing cannot initialize torch, CUDA or an engine.
        import cv2
        from .media import prepare_upload, render_timeline, render_video

        job = update_job(job_id, status="preparing")
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        target = directory / "input" / f"job_{job_id}.mp4"
        source, conversion = prepare_upload(job["upload"], target)
        cap = cv2.VideoCapture(source)
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
                raise ValueError("영상 길이와 FPS를 확인하지 못했습니다.")
            video = {"duration": count / fps, "fps": fps,
                     "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        finally:
            cap.release()
        estimate = math.ceil(count / fps * min(5, fps)) if job["vehicle"] == "moto" else None
        update_job(job_id, status="evaluating", video=video, conversion=conversion, expected_frames=estimate)
        run_engine(source, job["vehicle"], run_dir)
        update_job(job_id, status="rendering")
        hud = render_video(source, run_dir, directory / "hud.mp4")
        timeline = render_timeline(run_dir, directory / "timeline.png")
        summary = build_summary(run_dir, video["duration"])
        write_json(directory / "result.json", summary)
        update_job(job_id, status="done", completed_frames=summary["frame_count"],
                   files={"hud": hud, "frames": str(run_dir / "frames.jsonl"), "timeline": timeline})
    except Exception as exc:
        (directory / "worker.log").write_text(traceback.format_exc(), encoding="utf-8")
        message = ("평가 엔진이 종료됐습니다. 관리자에게 작업 주소를 전달해 주세요." if isinstance(exc, subprocess.CalledProcessError)
                   else str(exc) if isinstance(exc, ValueError)
                   else "분석을 완료하지 못했습니다. 관리자에게 작업 주소를 전달해 주세요.")
        update_job(job_id, status="error", error=message)


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith(("/api/", "/dev/")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/config")
def config():
    return {"uploads_enabled": os.environ.get("ROADSCOPE_DISABLE_ENGINE") != "1", "max_upload_bytes": MAX_UPLOAD}


@app.post("/api/jobs", status_code=202)
def create_job(video: UploadFile = File(...), vehicle: str = Form(...)):
    directory = None
    try:
        if os.environ.get("ROADSCOPE_DISABLE_ENGINE") == "1":
            raise HTTPException(503, "현재 서버는 결과 조회 전용입니다. 새 분석은 실행하지 않습니다.")
        if vehicle not in ("car", "moto"):
            raise HTTPException(422, "차량은 car 또는 moto여야 합니다.")
        filename = (video.filename or "video.mp4").replace("\\", "/").split("/")[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in (".mp4", ".mov"):
            raise HTTPException(415, "MP4 또는 MOV 영상을 골라 주세요.")
        if video.size is not None and video.size > MAX_UPLOAD:
            raise HTTPException(413, "영상은 최대 500MB까지 올릴 수 있습니다.")
        job_id = uuid.uuid4().hex
        directory = job_dir(job_id)
        directory.mkdir(parents=True)
        upload = directory / ("upload" + suffix)
        size = 0
        with upload.open("wb") as output:
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "영상은 최대 500MB까지 올릴 수 있습니다.")
                output.write(chunk)
        if not size:
            raise HTTPException(422, "영상 파일이 비어 있습니다.")
        write_json(directory / "job.json", {"id": job_id, "session": SESSION, "status": "queued", "kind": "real",
                   "filename": filename, "vehicle": vehicle, "upload": str(upload),
                   "run_dir": str(DATA / "results" / f"job_{job_id}"), "created_at": time.time()})
        EXECUTOR.submit(run_job, job_id)
        return {"id": job_id, "url": f"/#/job/{job_id}"}
    except Exception:
        if directory is not None and directory.is_dir():
            # Only this request's generated job directory, never user-supplied paths.
            shutil.rmtree(directory)
        raise
    finally:
        video.file.close()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = read_job(job_id)
    state = job["status"]
    count = job.get("completed_frames", 0)
    stage_index, progress = 0, 0
    if state == "preparing":
        progress = 5
    elif state == "evaluating":
        count, last_t = frame_progress(Path(job["run_dir"]) / "frames.jsonl")
        stage_index = 2 if count else 1
        duration = (job.get("video") or {}).get("duration")
        progress = 10 if not count else 30 + min(59, int(last_t / duration * 59)) if duration else 30
    elif state == "rendering":
        stage_index, progress = 3, 90
    elif state == "done":
        stage_index, progress = 4, 100
    result = {"id": job_id, "status": state, "progress": progress, "stage_index": stage_index,
              "stage": STAGES[min(stage_index, 3)], "filename": job["filename"], "vehicle": job["vehicle"],
              "completed_frames": count, "video": job.get("video"), "error": job.get("error"),
              "progress_estimated": state == "evaluating", "kind": job["kind"]}
    if state == "done":
        result["summary"] = read_json(job_dir(job_id) / "result.json")
        result["files"] = {name: f"/api/jobs/{job_id}/files/{name}" for name in job.get("files", {})}
    return result


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str, attachment: bool = False):
    job = read_job(job_id)
    types = {"hud": ("video/mp4", "hud.mp4"), "frames": ("application/x-ndjson", "frames.jsonl"),
             "timeline": ("image/png", "timeline.png")}
    if job["status"] != "done" or name not in types or name not in job.get("files", {}):
        raise HTTPException(404, "결과 파일이 없습니다.")
    path = Path(job["files"][name]).resolve()
    if not path.is_file() or not (path.is_relative_to(DATA.resolve()) or path.is_relative_to((ROOT / "runs_out").resolve())):
        raise HTTPException(404, "결과 파일이 없습니다.")
    mime, filename = types[name]
    return FileResponse(path, media_type=mime, filename=filename,
                        content_disposition_type="attachment" if attachment or name != "hud" else "inline")


@app.get("/dev/load")
def dev_load(request: Request, run: str):
    if os.environ.get("ROADSCOPE_DEV") != "1" or request.client.host not in ("127.0.0.1", "::1", "testclient"):
        raise HTTPException(404, "경로를 찾을 수 없습니다.")
    runs = (ROOT / "runs_out").resolve()
    target = (runs / run).resolve()
    if not target.is_relative_to(runs) or not (target / "meta.json").is_file() or not (target / "frames.jsonl").is_file():
        raise HTTPException(404, "저장된 결과 폴더를 찾을 수 없습니다.")
    with DEV_LOCK:
        job_id = uuid.uuid5(uuid.NAMESPACE_URL, str(target)).hex
        directory = job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            summary = build_summary(target)
            from .media import render_timeline, prepare_upload
            timeline = render_timeline(target, directory / "timeline.png")
            files = {"frames": str(target / "frames.jsonl"), "timeline": timeline}
            if (target / "hud.mp4").is_file():
                files["hud"], _ = prepare_upload(target / "hud.mp4", directory / "hud.mp4")
            write_json(directory / "result.json", summary)
            write_json(directory / "job.json", {"id": job_id, "session": SESSION, "status": "done", "kind": "saved",
                       "filename": target.name, "vehicle": summary["vehicle"], "run_dir": str(target),
                       "completed_frames": summary["frame_count"], "files": files, "created_at": time.time()})
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(422, "저장된 결과를 읽을 수 없습니다.") from exc
    return RedirectResponse(f"/#/job/{job_id}", status_code=303)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
