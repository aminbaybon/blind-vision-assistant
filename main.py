"""YOLO26n video-to-text prototype for an assistive-vision experiment.

The pipeline downloads a street clip and the official YOLO26n weights when
they are missing, draws detections, and writes English descriptions with a
ground-plane distance estimate. It does not judge whether a path is safe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
from ultralytics.utils.downloads import attempt_download_asset

import describe
from download_video import DownloadError, ensure_sample_video


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "data" / "test_street.mp4"
DEFAULT_MODEL = ROOT / "models" / "yolo26n.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
MODEL_URL = "https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26n.pt"
MIN_WEIGHT_BYTES = 1_000_000

CLASS_COLORS = {
    "person": (60, 180, 255),
    "bicycle": (255, 128, 0),
    "car": (80, 200, 80),
    "motorcycle": (200, 80, 255),
    "bus": (0, 140, 255),
    "truck": (40, 90, 220),
    "traffic light": (0, 220, 220),
    "stop sign": (60, 60, 220),
    "dog": (180, 120, 40),
}


class VideoError(RuntimeError):
    """Raised when a video cannot be read or written."""


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect street objects with YOLO26n and describe the scene in English, with an estimated distance."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input MP4. Downloaded when missing.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO26n weights. Downloaded when missing.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Run YOLO on every Nth frame. Skipped frames keep the last boxes so the output stays at the source frame rate.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between scene descriptions.")
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.5,
        help="Camera height above the ground in meters. Distance scales with this value.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=70.0,
        help="Horizontal field of view in degrees. Used to turn box position into meters.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=0.5,
        help="Horizon position as a fraction of image height. 0.5 is a level camera.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when PyTorch can see a GPU, otherwise CPU.",
    )
    args = parser.parse_args()
    if args.conf <= 0 or args.conf > 1:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.frame_skip < 1:
        parser.error("--frame-skip must be at least 1")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.camera_height <= 0:
        parser.error("--camera-height must be positive")
    if args.fov <= 1 or args.fov >= 179:
        parser.error("--fov must be between 1 and 179 degrees")
    if args.horizon <= 0 or args.horizon >= 1:
        parser.error("--horizon must be between 0 and 1")
    return args


def resolve_device(requested: str) -> tuple[str, str]:
    """Return the Ultralytics device argument and a human-readable device name."""
    cuda_available = torch.cuda.is_available()
    if requested == "cpu" or (requested == "auto" and not cuda_available):
        return "cpu", "cpu"
    if requested in {"auto", "cuda"} and cuda_available:
        name = torch.cuda.get_device_name(0)
        return "0", f"cuda ({name})"
    print("CUDA was requested but PyTorch cannot see a GPU. Using CPU.", file=sys.stderr)
    return "cpu", "cpu"


def ensure_model(model_path: Path) -> None:
    if model_path.is_file() and model_path.stat().st_size >= MIN_WEIGHT_BYTES:
        print(f"Using existing weights: {model_path}")
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading official YOLO26n weights...")
    errors: list[str] = []
    try:
        _download_with_ultralytics(model_path)
    except Exception as exc:
        errors.append(f"ultralytics: {exc}")
        print(f"Ultralytics download failed ({exc}). Trying the Hugging Face copy.")
        try:
            _download_url(MODEL_URL, model_path)
        except Exception as fallback_exc:
            errors.append(f"huggingface: {fallback_exc}")
    if not model_path.is_file() or model_path.stat().st_size < MIN_WEIGHT_BYTES:
        detail = "\n".join(errors) if errors else "weights file is missing"
        raise VideoError(f"Could not download yolo26n.pt.\n{detail}")
    print(f"Saved weights to {model_path} ({model_path.stat().st_size} bytes).")


def _download_with_ultralytics(model_path: Path) -> None:
    downloaded = Path(attempt_download_asset(str(model_path)))
    if not downloaded.is_file():
        raise VideoError("Ultralytics did not return a weights file")
    if downloaded.resolve() != model_path.resolve():
        shutil.copy2(downloaded, model_path)
    if not model_path.is_file() or model_path.stat().st_size < MIN_WEIGHT_BYTES:
        raise VideoError("Ultralytics weights file is missing or unexpectedly small")


def _download_url(url: str, dest: Path) -> None:
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "blind-vision-prototype/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(dest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_model(model_path: Path) -> YOLO:
    ensure_model(model_path)
    return YOLO(str(model_path))


def color_for(class_name: str) -> tuple[int, int, int]:
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    digest = zlib.crc32(class_name.encode("utf-8")) & 0xFFFFFF
    blue = 64 + (digest & 0x7F)
    green = 64 + ((digest >> 8) & 0x7F)
    red = 64 + ((digest >> 16) & 0x7F)
    return blue, green, red


def parse_detections(
    result,
    frame_width: int,
    frame_height: int,
    frame_time_sec: float,
    camera: describe.CameraGeometry,
) -> list[describe.Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    detections: list[describe.Detection] = []
    for box, confidence, class_id in zip(xyxy, confidences, classes):
        bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        class_name = str(names[int(class_id)])
        detections.append(
            describe.Detection(
                class_name=class_name,
                confidence=float(confidence),
                bbox_xyxy=bbox,
                position=describe.horizontal_position(bbox, frame_width),
                frame_time_sec=frame_time_sec,
                distance_m=describe.estimate_ground_distance(bbox, frame_width, frame_height, camera),
            )
        )
    return detections


def draw_overlay(frame, detections: list[describe.Detection], timestamp: str) -> None:
    for detection in detections:
        _draw_box(frame, detection)
    _draw_timestamp(frame, timestamp)


def _draw_box(frame, detection: describe.Detection) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    color = color_for(detection.class_name)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{detection.class_name} {detection.confidence:.2f}"
    if detection.distance_m is not None:
        label = f"{label} {describe.short_meters(detection.distance_m)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = y1 - text_height - baseline - 6
    if top < 0:
        top = min(y1 + 2, height - text_height - baseline - 4)
    origin = (x1, max(0, top))
    cv2.rectangle(
        frame,
        origin,
        (min(width - 1, origin[0] + text_width + 6), min(height - 1, origin[1] + text_height + baseline + 4)),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (origin[0] + 3, origin[1] + text_height + 1),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _draw_timestamp(frame, timestamp: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(timestamp, font, scale, thickness)
    cv2.rectangle(frame, (8, 8), (18 + text_width, 18 + text_height + baseline), (0, 0, 0), -1)
    cv2.putText(frame, timestamp, (12, 14 + text_height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise VideoError(f"Could not open a video writer for {path}")


def format_clock(seconds: float, fractional: bool) -> str:
    if seconds < 0:
        seconds = 0.0
    if fractional:
        total_ms = int(round(seconds * 1000.0))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def process_video(args: argparse.Namespace, model: YOLO, device_arg: str, device_label: str) -> dict:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"OpenCV could not open {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1:
        source_fps = 25.0
    output_path = args.output_dir / "annotated_video.mp4"
    writer: cv2.VideoWriter | None = None
    codec = ""
    windows: dict[int, list[list[describe.Detection]]] = defaultdict(list)
    inference_seconds: list[float] = []
    frames_read = 0
    last_detections: list[describe.Detection] = []
    width = 0
    height = 0
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        height, width = frame.shape[:2]
        if writer is None:
            writer, codec = open_writer(output_path, source_fps, (width, height))
        frame_time = frames_read / source_fps
        if frames_read % args.frame_skip == 0:
            infer_started = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device_arg,
                verbose=False,
                save=False,
            )
            inference_seconds.append(time.perf_counter() - infer_started)
            camera = describe.CameraGeometry(args.camera_height, args.fov, args.horizon)
            last_detections = parse_detections(results[0], width, height, frame_time, camera)
            bucket = int(frame_time // args.interval)
            windows[bucket].append(last_detections)
            if len(inference_seconds) == 1 or len(inference_seconds) % 25 == 0:
                print(
                    f"Inferred frame {frames_read} at {format_clock(frame_time, fractional=True)}: "
                    f"{len(last_detections)} objects, {inference_seconds[-1] * 1000:.0f} ms",
                    flush=True,
                )
        draw_overlay(frame, last_detections, format_clock(frame_time, fractional=True))
        if writer is not None:
            writer.write(frame)
        frames_read += 1

    capture.release()
    if writer is not None:
        writer.release()
    if frames_read == 0 or writer is None:
        raise VideoError(f"No frames could be read from {args.video}")

    duration = frames_read / source_fps
    loop_seconds = time.perf_counter() - started
    descriptions = _build_descriptions(windows, duration, args.interval)
    return {
        "output_path": output_path,
        "codec": codec,
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "duration_sec": duration,
        "frames_read": frames_read,
        "frames_inferred": len(inference_seconds),
        "inference_seconds": inference_seconds,
        "loop_seconds": loop_seconds,
        "descriptions": descriptions,
        "device_label": device_label,
    }


def _build_descriptions(windows: dict[int, list[list[describe.Detection]]], duration: float, interval: float) -> list[dict]:
    built: list[dict] = []
    previous_english: str | None = None
    for bucket in sorted(windows):
        frames = windows[bucket]
        start = bucket * interval
        end = min((bucket + 1) * interval, duration)
        representative = describe.select_representative_detections(frames)
        english = describe.describe(representative)
        emitted = english != previous_english
        if emitted:
            previous_english = english
        built.append(
            {
                "timestamp": format_clock(end, fractional=False),
                "time_sec": round(end, 3),
                "window_start_sec": round(start, 3),
                "window_end_sec": round(end, 3),
                "description_en": english,
                "emitted": emitted,
                "raw_detection_count": sum(len(frame) for frame in frames),
                "inferred_frames_in_window": len(frames),
                "detections": representative,
            }
        )
    return built


def write_descriptions(path: Path, rows: list[dict], header: str) -> int:
    lines = [header.rstrip(), ""]
    emitted = 0
    for row in rows:
        if not row["emitted"]:
            continue
        emitted += 1
        lines.append(f"[{row['timestamp']}] {row['description_en']}")
        lines.append("")
    omitted = len(rows) - emitted
    if omitted:
        lines.append(f"Identical consecutive descriptions omitted: {omitted}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return emitted


def write_detections(path: Path, rows: list[dict], metadata: dict) -> None:
    intervals = []
    for row in rows:
        detections = []
        for detection in row["detections"]:
            x1, y1, x2, y2 = detection.bbox_xyxy
            detections.append(
                {
                    "timestamp": row["timestamp"],
                    "time_sec": row["time_sec"],
                    "frame_time_sec": round(detection.frame_time_sec, 3),
                    "class": detection.class_name,
                    "confidence": round(detection.confidence, 4),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "position": detection.position,
                    "distance_m": None if detection.distance_m is None else round(detection.distance_m, 2),
                    "description_en": row["description_en"],
                }
            )
        intervals.append(
            {
                "timestamp": row["timestamp"],
                "time_sec": row["time_sec"],
                "window_start_sec": row["window_start_sec"],
                "window_end_sec": row["window_end_sec"],
                "description_en": row["description_en"],
                "emitted": row["emitted"],
                "raw_detection_count": row["raw_detection_count"],
                "inferred_frames_in_window": row["inferred_frames_in_window"],
                "detections": detections,
            }
        )
    payload = dict(metadata)
    payload["intervals"] = intervals
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_performance(path: Path, stats: dict) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def validate_output_video(path: Path) -> dict[str, int]:
    if not path.is_file() or path.stat().st_size < 1000:
        raise VideoError(f"Annotated video is missing or empty: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"OpenCV could not open the annotated video: {path}")
    readable = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame.size == 0:
            break
        readable += 1
    capture.release()
    if readable == 0:
        raise VideoError(f"Annotated video has no readable frames: {path}")
    return {"readable_frames": readable, "bytes": path.stat().st_size}


def main() -> int:
    configure_stdio()
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required.", file=sys.stderr)
        return 1
    started = time.perf_counter()
    args = parse_args()
    try:
        describe.self_check()
        video_info = ensure_sample_video(args.video)
        device_arg, device_label = resolve_device(args.device)
        print(f"Loading YOLO26n on {device_label} (ultralytics {ULTRALYTICS_VERSION}).")
        load_started = time.perf_counter()
        model = load_model(args.model)
        model_load_sec = time.perf_counter() - load_started
        print(
            f"Processing {args.video} | conf={args.conf} | imgsz={args.imgsz} | "
            f"frame_skip={args.frame_skip} | interval={args.interval}s | "
            f"camera_height={args.camera_height}m | fov={args.fov} | horizon={args.horizon}"
        )
        result = process_video(args, model, device_arg, device_label)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        descriptions_path = args.output_dir / "descriptions.txt"
        detections_path = args.output_dir / "detections.json"
        performance_path = args.output_dir / "performance.json"
        header = "\n".join(
            [
                "Experimental assistive descriptions. This is not a safety judgment.",
                (
                    "Distance assumes a level camera and treats the bottom of each box as the ground contact. "
                    f"Camera height {args.camera_height:g} m, horizontal field of view {args.fov:g} degrees, "
                    f"horizon at {args.horizon:.0%} of the image height."
                ),
                (
                    f"Model: YOLO26n | confidence {args.conf} | image size {args.imgsz} | "
                    f"interval {args.interval:g}s | frame skip {args.frame_skip} | device {device_label}"
                ),
            ]
        )
        emitted = write_descriptions(descriptions_path, result["descriptions"], header)
        inference_seconds: list[float] = result["inference_seconds"]
        inference_total = sum(inference_seconds)
        average_ms = (inference_total / len(inference_seconds) * 1000.0) if inference_seconds else 0.0
        processing_fps = (len(inference_seconds) / inference_total) if inference_total else 0.0
        total_execution_sec = time.perf_counter() - started
        metadata = {
            "model": "yolo26n.pt",
            "ultralytics_version": ULTRALYTICS_VERSION,
            "source_video": str(args.video),
            "video_source": video_info,
            "confidence_threshold": args.conf,
            "image_size": args.imgsz,
            "frame_skip": args.frame_skip,
            "description_interval_sec": args.interval,
            "device": device_label,
            "camera_height_m": args.camera_height,
            "horizontal_fov_deg": args.fov,
            "horizon_ratio": args.horizon,
            "note": (
                "Each class count is taken from the inferred frame in that interval with the most boxes "
                "of that class. Horizontal position is the bounding-box center in the left, middle, or "
                "right third of the frame. distance_m is a ground-plane estimate from the bottom of the "
                "box, the camera height, and the field of view. Motion and path safety are not estimated."
            ),
        }
        write_detections(detections_path, result["descriptions"], metadata)
        video_check = validate_output_video(result["output_path"])
        performance = {
            "video_duration_sec": round(result["duration_sec"], 3),
            "width": result["width"],
            "height": result["height"],
            "source_fps": round(result["source_fps"], 3),
            "total_frames_processed": result["frames_inferred"],
            "total_frames_read": result["frames_read"],
            "total_frames_written": video_check["readable_frames"],
            "annotated_video_bytes": video_check["bytes"],
            "video_codec": result["codec"],
            "average_inference_ms": round(average_ms, 2),
            "first_inference_ms": round(inference_seconds[0] * 1000.0, 2) if inference_seconds else 0.0,
            "processing_fps": round(processing_fps, 3),
            "frame_loop_sec": round(result["loop_seconds"], 3),
            "model_load_sec": round(model_load_sec, 3),
            "total_execution_sec": round(total_execution_sec, 3),
            "device": device_label,
            "confidence_threshold": args.conf,
            "image_size": args.imgsz,
            "frame_skip": args.frame_skip,
            "camera_height_m": args.camera_height,
            "horizontal_fov_deg": args.fov,
            "horizon_ratio": args.horizon,
            "descriptions_emitted": emitted,
            "description_intervals": len(result["descriptions"]),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "ultralytics_version": ULTRALYTICS_VERSION,
        }
        write_performance(performance_path, performance)
    except (DownloadError, VideoError, AssertionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Descriptions")
    print("------------")
    shown = 0
    for row in result["descriptions"]:
        if not row["emitted"]:
            continue
        print(f"[{row['timestamp']}] {row['description_en']}")
        shown += 1
        if shown >= 12:
            break
    if emitted > shown:
        print(f"... {emitted - shown} more descriptions in {descriptions_path}")
    print("")
    print(
        f"Device: {performance['device']} | "
        f"frames inferred: {performance['total_frames_processed']} | "
        f"average inference: {performance['average_inference_ms']:.1f} ms | "
        f"processing FPS: {performance['processing_fps']:.2f} | "
        f"total: {performance['total_execution_sec']:.1f} s"
    )
    print(f"Annotated video: {result['output_path']} ({video_check['bytes']} bytes, {video_check['readable_frames']} frames)")
    print(f"Descriptions: {descriptions_path}")
    print(f"Detections: {detections_path}")
    print(f"Performance: {performance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
