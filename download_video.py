"""Download a short street video and reject HTML or unreadable files."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = ROOT / "data" / "test_street.mp4"
SOURCE_SIDECAR = ROOT / "data" / "VIDEO_SOURCE.json"
USER_AGENT = "blind-vision-prototype/1.0"
MIN_BYTES = 100_000


@dataclass(frozen=True)
class VideoSource:
    url: str
    title: str
    creator: str
    license_name: str
    license_url: str


SOURCES: tuple[VideoSource, ...] = (
    VideoSource(
        url="https://raw.githubusercontent.com/aparsoft/yolo-streamlit-detection-tracking/master/videos/pedestrians_summer_street.mp4",
        title="Pedestrians on a summer street",
        creator="Pexels contributor K, via aparsoft/yolo-streamlit-detection-tracking",
        license_name="Pexels License (free to use and modify; attribution not required)",
        license_url="https://www.pexels.com/license/",
    ),
    VideoSource(
        url="https://github.com/aparsoft/yolo-streamlit-detection-tracking/raw/master/videos/pedestrians_summer_street.mp4",
        title="Pedestrians on a summer street",
        creator="Pexels contributor K, via aparsoft/yolo-streamlit-detection-tracking",
        license_name="Pexels License (free to use and modify; attribution not required)",
        license_url="https://www.pexels.com/license/",
    ),
    VideoSource(
        url="https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
        title="Person, bicycle, and car detection sample",
        creator="Intel IoT DevKit sample-videos",
        license_name="Creative Commons Attribution 4.0 International",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    VideoSource(
        url="https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        title="Person, bicycle, and car detection sample",
        creator="Intel IoT DevKit sample-videos",
        license_name="Creative Commons Attribution 4.0 International",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    VideoSource(
        url="https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/people-detection.mp4",
        title="People detection sample",
        creator="Intel IoT DevKit sample-videos",
        license_name="Creative Commons Attribution 4.0 International",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
)


class DownloadError(RuntimeError):
    """Raised when every video source fails or the file cannot be decoded."""


def ensure_sample_video(path: Path = DEFAULT_VIDEO_PATH) -> dict[str, str]:
    """Return source metadata for an OpenCV-readable MP4 at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and _existing_video_is_usable(path):
        print(f"Using existing video: {path}")
        return _load_sidecar(path)

    errors: list[str] = []
    for source in SOURCES:
        print(f"Downloading test video from {source.url}")
        try:
            _download(source.url, path)
            readable, detail = opencv_video_status(path)
            if not readable:
                raise DownloadError(detail)
            info = _source_info(source, path)
            _write_sidecar(info)
            print(f"Downloaded {path} ({detail}).")
            return info
        except (DownloadError, OSError, urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{source.url} — {exc}")
            if path.exists():
                path.unlink()
            print(f"Download failed: {exc}")

    joined = "\n".join(errors)
    raise DownloadError(
        "Could not download a readable street video. "
        "Place an MP4 at data/test_street.mp4 and run again.\n"
        f"{joined}"
    )


def opencv_video_status(path: Path) -> tuple[bool, str]:
    """Open ``path`` with OpenCV and try to decode the first frame."""
    if not path.is_file():
        return False, "file does not exist"
    if path.stat().st_size < MIN_BYTES:
        return False, f"file is too small ({path.stat().st_size} bytes)"
    if not _has_mp4_header(path):
        return False, "file does not look like an MP4"

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return False, "OpenCV could not open the file"
    ok, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if not ok or frame is None or frame.size == 0:
        return False, "OpenCV could not read a frame"
    if width < 2 or height < 2:
        return False, "video has no usable resolution"
    return True, f"{width}x{height}, {frame_count} frames, {fps:.2f} fps"


def _existing_video_is_usable(path: Path) -> bool:
    readable, detail = opencv_video_status(path)
    if readable:
        print(f"Existing video is readable: {detail}")
        return True
    print(f"Existing file at {path} is not a readable video ({detail}). Replacing it.")
    return False


def _download(url: str, dest: Path) -> None:
    temporary = dest.with_suffix(dest.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise DownloadError(f"server returned HTML ({content_type})")
            written = 0
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    if written == 0 and chunk.lstrip().startswith(b"<"):
                        raise DownloadError("server returned an HTML page instead of video bytes")
                    if written == 0 and chunk.startswith(b"version https://git-lfs.github.com"):
                        raise DownloadError("server returned a Git LFS pointer, not video bytes")
                    handle.write(chunk)
                    written += len(chunk)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if written < MIN_BYTES or not _has_mp4_header(temporary):
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"downloaded {written} bytes that are not a valid MP4")
    temporary.replace(dest)


def _has_mp4_header(path: Path) -> bool:
    header = path.read_bytes()[:32]
    return b"ftyp" in header


def _source_info(source: VideoSource, path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "url": source.url,
        "title": source.title,
        "creator": source.creator,
        "license_name": source.license_name,
        "license_url": source.license_url,
    }


def _write_sidecar(info: dict[str, str]) -> None:
    SOURCE_SIDECAR.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_sidecar(path: Path) -> dict[str, str]:
    if SOURCE_SIDECAR.is_file():
        loaded = json.loads(SOURCE_SIDECAR.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("url"):
            return {str(key): str(value) for key, value in loaded.items()}
    return {
        "path": str(path),
        "url": "",
        "title": "Local video",
        "creator": "",
        "license_name": "",
        "license_url": "",
    }


def main() -> int:
    try:
        info = ensure_sample_video()
    except DownloadError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
