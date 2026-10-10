# SPDX-License-Identifier: AGPL-3.0-only
"""프레임 계약만 사용하는 HUD, 영상, 타임라인 렌더러.

엔진/모델을 import하지 않는다. 좌표는 원본 영상 픽셀, 영상 입력/출력은 BGR.
CLI: python -m roadscope.viz VIDEO RUN_DIR --out RUN_DIR/hud.mp4
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BIN_LABELS = ("안전", "주의", "위험", "치명")
BIN_COLORS = ("#35C49B", "#F4C65C", "#FF865E", "#DE83EE")
WARNING_NAMES = {"FCW": "정면 충돌", "LDW": "차선 편향", "BSW": "측면 접근", "HEAD": "역주행"}
WARNING_COLORS = {"FCW": "#FF865E", "LDW": "#F4C65C", "BSW": "#68B8FA", "HEAD": "#DE83EE"}
# 현재 계약에는 도로/자차 마스크가 없다. 하단 바이크를 피하는 보수적 표시 한계.
# 도로를 추론하거나 차선 좌표를 고치지 않으며, 원근상 가까운 차선은 생략한다.
LANE_ROAD_BOTTOM_RATIO = 0.50


@lru_cache(maxsize=1)
def korean_font_path() -> str:
    """한글 미지원 기본 폰트로 조용히 대체하지 않는다."""
    candidates = [os.environ.get("HUD_FONT", ""), "C:/Windows/Fonts/malgun.ttf",
                  "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("한글 폰트를 찾지 못했습니다. HUD_FONT에 한글 TTF/TTC 경로를 지정하세요.")


@lru_cache(maxsize=32)
def _font(size: int):
    return ImageFont.truetype(korean_font_path(), size)


def _distance(value) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
        return f"{number:.1f} m" if math.isfinite(number) and number >= 0 else "—"
    except (TypeError, ValueError):
        return "—"


def _fit(draw, text, font, width):
    text = " ".join(str(text).split())
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…" if text else ""


def _lane_overlay(size, record, header_h, footer_h):
    """차선 전용 투명 레이어: UI 밖의 상단 도로 후보 영역에만 래스터화한다."""
    width, height = size
    overlay = Image.new("RGBA", size)
    if not record or record.get("lane_measurable") is not True:
        return overlay
    top = header_h + 1
    bottom = min(int(height * LANE_ROAD_BOTTOM_RATIO), height - footer_h - 1)
    if bottom <= top:
        return overlay
    # 타일 자체가 clip rect이므로 선 두께까지 상태 바와 바이크 제외 영역을 넘지 않는다.
    tile = Image.new("RGBA", (width, bottom - top))
    painter = ImageDraw.Draw(tile)
    for name in ("left", "right"):
        line = (record.get("lanes") or {}).get(name)
        if line is None or len(line) != 4:
            continue
        coords = tuple(float(v) for v in line)
        if not all(math.isfinite(v) for v in coords):
            continue
        x0, y0, x1, y1 = coords
        painter.line((x0, y0 - top, x1, y1 - top), fill="#86D6BF",
                     width=max(1, round(2 * width / 960)))
    # 검출된 차량/바이크 박스 내부에도 선을 얹지 않는다.
    for obj in record.get("objects", []):
        bbox = obj.get("bbox")
        if bbox is not None and len(bbox) == 4:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            if all(math.isfinite(v) for v in (x0, y0, x1, y1)) and x1 > x0 and y1 > y0:
                painter.rectangle((x0, y0 - top, x1, y1 - top), fill=(0, 0, 0, 0))
    overlay.paste(tile, (0, top))
    return overlay


def render_frame(bgr: np.ndarray, record: dict | None) -> np.ndarray:
    """입력 배열/레코드를 변경하지 않고 동일 크기의 HUD BGR 배열을 반환한다.

    record=None은 첫 평가 이전의 미평가 프레임용이며 스키마 확장이 아니다.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("bgr는 (H, W, 3) uint8 배열이어야 합니다.")
    h, w = bgr.shape[:2]
    # 원본 픽셀 좌표에서 그린다. 축소는 render_video의 명시적 옵션만 담당한다.
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    width, height = image.size
    unit = width / 960
    pad = round(16 * unit)
    small, medium, large = (_font(max(12, round(v * unit))) for v in (15, 18, 25))
    header_h, footer_h = round(68 * unit), round(46 * unit)
    layer = _lane_overlay(image.size, record, header_h, footer_h)
    draw = ImageDraw.Draw(layer)
    # 상태 패널은 항상 도형보다 나중에 그린다.
    draw.rectangle((0, 0, width, header_h), fill=(12, 20, 32, 235))
    draw.rectangle((0, height - footer_h, width, height), fill=(12, 20, 32, 235))

    if record is None:
        draw.text((pad, pad), "미평가 · 첫 평가 결과 대기", font=large, fill="white")
    else:
        level = int(record["bin"])
        if level not in range(4):
            raise ValueError("record.bin은 0~3이어야 합니다.")
        color = BIN_COLORS[level]
        badge_w = round(116 * unit)
        draw.rounded_rectangle((pad, round(12 * unit), pad + badge_w, round(55 * unit)),
                               radius=round(9 * unit), fill=color)
        draw.text((pad + round(12 * unit), round(18 * unit)),
                  f"L{level} {BIN_LABELS[level]}", font=large, fill="#101A27")
        text_x = pad * 2 + badge_w
        title = record.get("top_reason") or "활성 경고 없음"
        draw.text((text_x, round(14 * unit)), _fit(draw, title, medium, width - text_x - pad),
                  font=medium, fill="white")
        stamp = f"평가 시점 {float(record['t_sec']):.2f}s · 거리 추정값"
        draw.text((text_x, round(39 * unit)), stamp, font=small, fill="#B5C4D7")

        measurable = record.get("lane_measurable") is True

        warnings = record.get("warnings") or {}
        for obj in record.get("objects", []):
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            x0, x1 = max(0, min(width - 1, x0)), max(0, min(width - 1, x1))
            y0, y1 = max(header_h + 2, y0), min(height - footer_h - 2, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            # track_id 없는 객체끼리 None == None으로 잘못 연결하지 않는다.
            matches = [(name, warning) for name, warning in warnings.items()
                       if name in WARNING_NAMES and warning.get("level", 0) > 0
                       and ((obj.get("track_id") is not None
                             and warning.get("track_id") == obj["track_id"])
                            or obj.get("warning") == name)]
            matches.sort(key=lambda item: -item[1]["level"])
            box_color = WARNING_COLORS[matches[0][0]] if matches else "#A3ADBA"
            draw.rectangle((x0, y0, x1, y1), outline=box_color,
                           width=max(1, round((3 if matches else 1) * unit)))
            if matches:
                label = " / ".join(f"{name} L{warning['level']}" for name, warning in matches)
                label += f" · {_distance(obj.get('distance_m'))}"
                label = _fit(draw, label, small, width - pad * 2)
                tw = draw.textlength(label, font=small) + pad
                tx = max(pad, min(x0, width - tw - pad))
                ty = max(header_h + 4, y0 - round(29 * unit))
                draw.rounded_rectangle((tx, ty, tx + tw, ty + round(26 * unit)),
                                       radius=4, fill=(12, 20, 32, 240), outline=box_color)
                draw.text((tx + pad / 2, ty + round(3 * unit)), label, font=small, fill="white")

        # 전경 텍스트는 도형 다음에 그려 박스가 글을 덮지 않는다.
        active = [(name, warnings[name]) for name in WARNING_NAMES
                  if warnings.get(name, {}).get("level", 0) > 0]
        active.sort(key=lambda item: -item[1]["level"])
        row_h = round(29 * unit)
        start_y = height - footer_h - pad - row_h * len(active)
        for i, (name, warning) in enumerate(active):
            label = f"{name} L{warning['level']} · {warning.get('reason') or WARNING_NAMES[name]}"
            label = _fit(draw, label, small, width - pad * 3)
            tw = draw.textlength(label, font=small) + pad
            y = max(header_h + pad, start_y + i * row_h)
            draw.rounded_rectangle((pad, y, pad + tw, y + row_h - 3), radius=4,
                                   fill=(12, 20, 32, 225))
            draw.text((pad + 6, y + 2), label, font=small, fill=WARNING_COLORS[name])

        footer_y = height - footer_h + round(13 * unit)
        distances = "     ".join(f"{name} {_distance(record.get(key))}" for name, key in
                                 (("정면", "d_front_m"), ("좌", "d_left_m"), ("우", "d_right_m")))
        draw.text((pad, footer_y), distances, font=small, fill="#E6EDF7")
        notice = "차선 측정 가능" if measurable else "차선 측정 불가"
        nx = width - pad - draw.textlength(notice, font=small)
        draw.text((nx, footer_y), notice, font=small, fill="#B5C4D7" if measurable else "#F4C65C")

    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def load_records(run_dir) -> list[dict]:
    records = []
    with (Path(run_dir) / "frames.jsonl").open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (not isinstance(record.get("frame_idx"), int) or record["frame_idx"] < 0
                    or not isinstance(record.get("t_sec"), (int, float))
                    or not math.isfinite(record["t_sec"]) or record["t_sec"] < 0
                    or record.get("bin") not in range(4)):
                raise ValueError(f"frames.jsonl {line_no}행: 프레임 번호/시각/등급 오류")
            if records and (record["frame_idx"] <= records[-1]["frame_idx"]
                            or record["t_sec"] <= records[-1]["t_sec"]):
                raise ValueError(f"frames.jsonl {line_no}행: 프레임 번호와 시간은 증가해야 합니다.")
            records.append(record)
    if not records:
        raise ValueError("frames.jsonl에 평가 레코드가 없습니다.")
    return records


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("영상 변환용 FFmpeg를 찾지 못했습니다. 관리자에게 실행 환경 확인을 요청하세요.") from exc


def _video_error(detail):
    """FFmpeg의 흔한 실패 이유를 사용자에게 한국어로 전달한다."""
    lower = detail.lower()
    for marker, message in (
        ("permission denied", "파일 접근 권한이 없거나 다른 프로그램이 파일을 사용 중입니다."),
        ("no space left", "영상을 저장할 디스크 공간이 부족합니다."),
        ("no such file", "업로드 파일을 찾을 수 없습니다."),
        ("unknown encoder", "현재 FFmpeg에 H.264 인코더가 없습니다."),
        ("invalid data", "파일이 손상되었거나 지원하지 않는 영상 형식입니다."),
        ("moov atom not found", "MP4의 재생 정보가 없습니다. 업로드 파일이 완전한지 확인하세요."),
        ("matches no streams", "파일에서 영상 스트림을 찾지 못했습니다."),
    ):
        if marker in lower:
            return message
    return "영상 정보를 읽거나 변환하지 못했습니다. 다른 영상 파일로 시도하거나 작업 로그를 확인하세요."


def probe_video_codec(path):
    """별도 ffprobe 설치 없이 FFmpeg로 첫 영상 스트림만 조사한다."""
    try:
        completed = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-i", str(path), "-map", "0:v:0",
             "-an", "-sn", "-dn", "-frames:v", "0", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("영상 정보 확인이 30초를 초과했습니다. 파일 상태를 확인하세요.") from exc
    except OSError as exc:
        raise RuntimeError("영상 변환 도구를 실행하지 못했습니다. 실행 권한을 확인하세요.") from exc
    stream = re.search(r"Stream #0:\d+[^\r\n]*?Video:\s*([\w]+)[^\r\n]*", completed.stderr)
    if completed.returncode or not stream:
        raise ValueError(_video_error(completed.stderr))
    container = re.search(r"Input #0, ([^\r\n]+?), from", completed.stderr)
    return {"codec": stream.group(1), "yuv420p": bool(re.search(r"\byuv420p\b", stream.group(0))),
            "mp4": bool(container and "mp4" in container.group(1).split(","))}


def prepare_upload(source, out_path):
    """업로드를 MP4/H.264/yuv420p로 정규화. 호환 H.264는 재인코딩하지 않는다.

    반환값은 (출력 경로, 처리 방식). source는 변경하지 않는다.
    """
    source, out_path = Path(source), Path(out_path)
    if source.resolve() == out_path.resolve():
        raise ValueError("업로드 원본과 변환 출력 경로는 달라야 합니다.")
    info = probe_video_codec(source)
    compatible = info["codec"] == "h264" and info["yuv420p"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upload-", dir=out_path.parent) as temp:
        prepared = Path(temp) / "prepared.mp4"
        if compatible and info["mp4"]:
            shutil.copy2(source, prepared)
            mode = "H.264 유지 · 재인코딩 생략"
        else:
            command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                       "-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac"]
            if compatible:
                command += ["-c:v", "copy"]
                mode = "H.264 유지 · MP4 컨테이너 변환"
            else:
                command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
                mode = "H.264/yuv420p 변환"
            command += ["-movflags", "+faststart", str(prepared)]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                (out_path.parent / "conversion.log").write_text(result.stderr, encoding="utf-8")
                raise RuntimeError(_video_error(result.stderr))
        prepared.replace(out_path)
    return str(out_path), mode


def encode_mp4(silent_path, out_path, audio_source=None, duration=None):
    """OpenCV 임시 영상을 브라우저용 H.264로 변환. 원본 음성은 있으면 보존."""
    out_path = Path(out_path)
    command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent_path)]
    if audio_source:
        command += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac"]
    if duration is not None:
        command += ["-t", str(duration)]
    command += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(out_path)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace")[-1800:]
        raise RuntimeError(f"H.264 영상 변환 실패: {detail}") from exc
    return str(out_path)


def _output_size(width, height, max_width=None):
    """기본 원본 크기(+홀수 가장자리 1px 패딩). 옵션은 축소만 허용한다."""
    if max_width is not None and (isinstance(max_width, bool)
                                  or not isinstance(max_width, int) or max_width < 2):
        raise ValueError("max_width는 2 이상의 정수 또는 None이어야 합니다.")
    if max_width is None or max_width >= width:
        return width + width % 2, height + height % 2
    out_w = max_width // 2 * 2
    out_h = max(2, round(height * out_w / width) // 2 * 2)
    return out_w, out_h


def render_video(video_path, run_dir, out_path, *, max_width=None):
    """기본 원본 해상도/FPS 유지. max_width 지정 시 HUD 합성 후에만 축소한다.

    평가 사이에는 직전 레코드를 유지한다. 홀수 크기는 우측/하단 1px 패딩.
    """
    video_path, out_path = Path(video_path), Path(out_path)
    if video_path.resolve() == out_path.resolve():
        raise ValueError("출력 경로는 원본 영상과 달라야 합니다.")
    records = load_records(run_dir)
    _ffmpeg()  # 긴 렌더를 시작하기 전에 인코더 의존성을 확인한다.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("원본 영상 FPS를 읽지 못했습니다.")
    width, height = (int(cap.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1:
        cap.release()
        raise ValueError("원본 영상 크기를 읽지 못했습니다.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="hud-", dir=out_path.parent) as temp:
            raw_path = Path(temp) / "silent.mp4"
            size = _output_size(width, height, max_width)
            shrinking = max_width is not None and max_width < width
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise RuntimeError("OpenCV MP4 writer를 열지 못했습니다.")
            index, cursor, current = 0, 0, None
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    while cursor < len(records) and records[cursor]["frame_idx"] <= index:
                        current = records[cursor]
                        cursor += 1
                    cached = render_frame(frame, current)
                    if shrinking:
                        cached = cv2.resize(cached, size, interpolation=cv2.INTER_AREA)
                    elif size != (width, height):
                        cached = cv2.copyMakeBorder(cached, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                    writer.write(cached)
                    index += 1
            finally:
                writer.release()
            if not index or cursor != len(records):
                raise ValueError("영상이 비었거나 레코드가 영상 프레임 범위를 벗어났습니다.")
            if expected_frames > 0 and index < expected_frames - 1:
                raise ValueError(f"영상 디코딩이 중단되었습니다: {index}/{expected_frames} 프레임")
            encoded = Path(temp) / "encoded.mp4"
            encode_mp4(raw_path, encoded, audio_source=video_path, duration=index / fps)
            encoded.replace(out_path)
    finally:
        cap.release()
    return str(out_path)


def _end_time(run_dir, records):
    meta_path = Path(run_dir) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig")) if meta_path.exists() else {}
    source = Path(meta.get("source", ""))
    if not source.is_absolute():
        local_source = Path(run_dir) / source
        if local_source.is_file():
            source = local_source
    if source.is_file():
        cap = cv2.VideoCapture(str(source))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and count / fps > records[-1]["t_sec"]:
                return count / fps
        finally:
            cap.release()
    step = (records[-1]["t_sec"] - records[-2]["t_sec"] if len(records) > 1
            else 1 / max(float(meta.get("fps_eval", 1)), 0.01))
    return records[-1]["t_sec"] + step


def render_timeline(run_dir, out_path, *, time_of_event=None, time_of_alert=None):
    """위험등급/경고 구간. 선택적 사고·경고 기준 시각(초)을 두 축에 겹친다."""
    from roadscope.config import RUNS_DIR
    os.environ.setdefault("MPLCONFIGDIR", str(RUNS_DIR / ".matplotlib"))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    markers = []
    for value, label, color in ((time_of_event, "사고 시각", "#FF3838"),
                                (time_of_alert, "경고 기준", "#FFA500")):
        if value is not None:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label}은 0 이상의 유한한 초 단위 값이어야 합니다.")
            markers.append((value, label, color))
    records = load_records(run_dir)
    font_manager.fontManager.addfont(korean_font_path())
    font = font_manager.FontProperties(fname=korean_font_path())
    end = _end_time(run_dir, records)
    times = [r["t_sec"] for r in records] + [end]
    bins = [r["bin"] for r in records]
    # 제목/범례 공간을 명시적으로 확보한다. 자동 배치의 상단 경계 의존을 피한다.
    fig = Figure(figsize=(13, 6.6), facecolor="#101824")
    FigureCanvasAgg(fig)
    upper, lower = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.84, hspace=0.35)
    fig.suptitle("시간에 따른 위험도", x=0.15, y=0.965, ha="left", va="top",
                 color="white", fontproperties=font, fontsize=17)
    for ax in (upper, lower):
        ax.set_facecolor("#152132")
        ax.tick_params(colors="#D3DDEB", length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#3D4C60", alpha=0.4)
        ax.set_xlim(0, max([end] + [t + max(end * 0.01, 0.1) for t, _, _ in markers]))
    upper.step(times, bins + bins[-1:], where="post", color="#EBF2FC", linewidth=2)
    for i, level in enumerate(bins):
        upper.axvspan(times[i], times[i + 1], color=BIN_COLORS[level], alpha=0.12)
    upper.set_ylim(-0.3, 3.5)
    upper.set_yticks(range(4), [f"L{i}  {label}" for i, label in enumerate(BIN_LABELS)], fontproperties=font)
    lower.set_yticks(range(4), [f"{name}  {label}" for name, label in WARNING_NAMES.items()], fontproperties=font)
    lower.set_ylim(3.6, -0.6)
    lower.set_xlabel("영상 시간 (초) · 평가 사이에는 직전 결과 유지", color="#C6D4E6", fontproperties=font, labelpad=12)
    for row, name in enumerate(WARNING_NAMES):
        start, previous = times[0], 0
        for i in range(len(records) + 1):
            level = records[i].get("warnings", {}).get(name, {}).get("level", 0) if i < len(records) else 0
            if level != previous:
                if previous > 0:
                    lower.broken_barh([(start, times[i] - start)], (row - 0.28, 0.56), facecolors=BIN_COLORS[previous])
                    if times[i] - start >= end * 0.035:
                        lower.text((start + times[i]) / 2, row, f"L{previous}", ha="center", va="center", color="#101824", fontsize=9, weight="bold")
                start, previous = times[i], level
    if times[0] > 0:
        for ax in (upper, lower):
            ax.axvspan(0, times[0], facecolor="#5A6472", hatch="//", alpha=0.35)
        upper.text(times[0] / 2, 1.5, "미평가", fontproperties=font, color="white", ha="center")
    legend = [Patch(color=BIN_COLORS[i], label=f"L{i} {BIN_LABELS[i]}") for i in (1, 2, 3)]
    for value, label, color in markers:
        for ax in (upper, lower):
            ax.axvline(value, color=color, linestyle="--", linewidth=1.6, zorder=5)
        legend.append(Line2D([0], [0], color=color, linestyle="--", label=f"{label} {value:g}초"))
    fig.legend(handles=legend,
                 prop=font, loc="upper right", bbox_to_anchor=(0.98, 0.91), ncol=5 if markers else 3,
                 facecolor="#152132", edgecolor="none", labelcolor="white")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    fig.clear()
    return str(out_path)


def warning_events(run_dir):
    """경고 시작/등급/대상 변경 이벤트. 사유와 거리는 해당 시점의 값이다."""
    rows, previous = [], {}
    for record in load_records(run_dir):
        current = {}
        for name in WARNING_NAMES:
            warning = record.get("warnings", {}).get(name, {})
            level = warning.get("level", 0)
            if level <= 0:
                continue
            key = (level, warning.get("track_id"))
            current[name] = key
            if previous.get(name) != key:
                rows.append([round(record["t_sec"], 2), name, f"L{level}",
                             warning.get("reason", ""), _distance(warning.get("distance_m"))])
        previous = current
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-width", type=int, default=None, help="지정 폭까지 축소 (예: 960). 기본은 원본 해상도")
    args = parser.parse_args()
    print(render_video(args.video, args.run_dir, args.out or args.run_dir / "hud.mp4", max_width=args.max_width))
    print(render_timeline(args.run_dir, args.run_dir / "timeline.png"))
