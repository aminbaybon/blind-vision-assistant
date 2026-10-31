"""English descriptions of YOLO detections, including a ground-plane distance estimate.

Counts come from the busiest inferred frame for each class inside a time window.
Distance uses the pinhole model: a level camera, a known height, and the field of
view. The bottom of each box is treated as the point where the object meets the
ground. Nothing here claims the path is safe or that an object is moving.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraGeometry:
    height_m: float
    horizontal_fov_deg: float
    horizon_ratio: float


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    position: str
    frame_time_sec: float
    distance_m: float | None


NAVIGATION_PRIORITY: tuple[str, ...] = (
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "train",
    "traffic light",
    "stop sign",
    "dog",
    "fire hydrant",
    "parking meter",
    "bench",
    "suitcase",
    "backpack",
    "umbrella",
    "handbag",
)

MAX_CLAUSES = 4
_SIMILAR_DISTANCE_RATIO = 1.25

_COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}

_POSITION_RANK = {"left": 0, "center": 1, "right": 2, "across": 3}

_POSITION_EN = {
    "left": "on the left",
    "center": "in the center",
    "right": "on the right",
}

_IRREGULAR_PLURALS = {
    "knife": "knives",
    "mouse": "mice",
    "person": "people",
}


@dataclass(frozen=True)
class _Clause:
    class_name: str
    count: int
    position: str
    priority: int
    singular: str
    plural: str
    near_m: float | None
    far_m: float | None


def horizontal_position(bbox_xyxy: tuple[float, float, float, float], frame_width: int) -> str:
    """Map a box center to the left, center, or right third of the frame."""
    x1, _, x2, _ = bbox_xyxy
    center_x = (x1 + x2) / 2.0
    if frame_width <= 0:
        return "center"
    if center_x < frame_width / 3.0:
        return "left"
    if center_x < (2.0 * frame_width) / 3.0:
        return "center"
    return "right"


def estimate_ground_distance(
    bbox_xyxy: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    camera: CameraGeometry,
) -> float | None:
    """Estimate meters to the ground contact at the bottom of the box.

    For a level camera, the horizon is a fixed fraction of the image height and
    the ground distance is camera_height * focal_length / pixels_below_horizon.
    The result is None when the contact point is above the horizon or so close
    to it that a one-pixel error would dominate.
    """
    if frame_width <= 1 or frame_height <= 1 or camera.height_m <= 0:
        return None
    if camera.horizontal_fov_deg <= 0 or camera.horizontal_fov_deg >= 180:
        return None
    _, _, _, y2 = bbox_xyxy
    half_fov = math.radians(camera.horizontal_fov_deg) / 2.0
    focal_y = (frame_width / 2.0) / math.tan(half_fov)
    horizon_y = camera.horizon_ratio * frame_height
    pixels_below = y2 - horizon_y
    if pixels_below <= 1.0:
        return None
    distance_m = camera.height_m * focal_y / pixels_below
    if distance_m < 0.3 or distance_m > 60.0:
        return None
    return distance_m


def priority_of(class_name: str) -> int:
    try:
        return NAVIGATION_PRIORITY.index(class_name)
    except ValueError:
        return len(NAVIGATION_PRIORITY)


def select_representative_detections(frames: list[list[Detection]]) -> list[Detection]:
    """Keep, per class, the inferred frame in the window that saw the most instances."""
    best: dict[str, list[Detection]] = {}
    for frame in frames:
        grouped: dict[str, list[Detection]] = {}
        for detection in frame:
            grouped.setdefault(detection.class_name, []).append(detection)
        for class_name, items in grouped.items():
            current = best.get(class_name)
            if _is_clearer_view(items, current):
                best[class_name] = items
    selected: list[Detection] = []
    for class_name in sorted(best, key=lambda name: (priority_of(name), name)):
        selected.extend(best[class_name])
    return selected


def describe(detections: list[Detection]) -> str:
    """Return one English sentence."""
    clauses, omitted_count = _clauses_for(detections)
    return _english(clauses, omitted_count)


def short_meters(distance_m: float) -> str:
    """Compact label such as 4.7m or 12m."""
    return f"{_format_value(distance_m)}m"


def self_check() -> None:
    """Check sentence shape and the ground-plane distance formula."""
    camera = CameraGeometry(height_m=1.5, horizontal_fov_deg=70.0, horizon_ratio=0.5)
    distance = estimate_ground_distance((0.0, 400.0, 100.0, 650.0), 1280, 720, camera)
    if distance is None or abs(distance - 4.727) > 0.02:
        raise AssertionError(f"ground distance was {distance}, expected about 4.727 m")
    above_horizon = estimate_ground_distance((0.0, 100.0, 40.0, 200.0), 1280, 720, camera)
    if above_horizon is not None:
        raise AssertionError("a box above the horizon should not get a ground distance")

    def det(class_name: str, position: str, distance_m: float | None, confidence: float = 0.9) -> Detection:
        return Detection(class_name, confidence, (0.0, 0.0, 10.0, 10.0), position, 0.0, distance_m)

    cases = [
        (
            [det("person", "center", 5.0), det("person", "center", 5.2)],
            "Two people in the center, about 5 meters away.",
        ),
        (
            [
                det("bicycle", "left", 8.0),
                det("person", "right", 4.0),
                det("person", "right", 5.0),
                det("person", "right", 6.0),
            ],
            "A bicycle on the left, about 8 meters away and three people on the right, about 4 to 6 meters away.",
        ),
        (
            [det("car", "center", 7.0)],
            "A car in the center, about 7 meters away.",
        ),
        (
            [
                det("person", "left", 3.0),
                det("person", "left", 4.0),
                det("person", "center", 8.0),
                det("person", "right", 12.0),
            ],
            "Multiple people across the scene, about 3 to 12 meters away.",
        ),
        (
            [det("car", "left", None)],
            "A car on the left.",
        ),
        ([], "No notable objects detected."),
    ]
    for detections, expected in cases:
        english = describe(detections)
        if english != expected:
            raise AssertionError(f"description was {english!r}, expected {expected!r}")


def _is_clearer_view(candidate: list[Detection], current: list[Detection] | None) -> bool:
    if current is None or len(candidate) > len(current):
        return True
    if len(candidate) < len(current):
        return False
    return _confidence_sum(candidate) > _confidence_sum(current)


def _confidence_sum(detections: list[Detection]) -> float:
    return sum(detection.confidence for detection in detections)


def _span(items: list[Detection]) -> tuple[float | None, float | None]:
    distances = [item.distance_m for item in items if item.distance_m is not None]
    if not distances:
        return None, None
    return min(distances), max(distances)


def _clauses_for(detections: list[Detection]) -> tuple[list[_Clause], int]:
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.class_name].append(detection)

    clauses: list[_Clause] = []
    for class_name, items in grouped.items():
        by_position: dict[str, list[Detection]] = defaultdict(list)
        for detection in items:
            by_position[detection.position].append(detection)
        occupied = [position for position in ("left", "center", "right") if by_position[position]]
        singular, plural = _english_nouns(class_name)
        rank = priority_of(class_name)
        if len(occupied) >= 2 and len(items) >= 3:
            near_m, far_m = _span(items)
            clauses.append(_Clause(class_name, len(items), "across", rank, singular, plural, near_m, far_m))
            continue
        for position in occupied:
            near_m, far_m = _span(by_position[position])
            clauses.append(
                _Clause(class_name, len(by_position[position]), position, rank, singular, plural, near_m, far_m)
            )

    ranked = sorted(clauses, key=lambda clause: (clause.priority, _POSITION_RANK[clause.position], clause.class_name))
    selected = ranked[:MAX_CLAUSES]
    omitted_count = sum(clause.count for clause in ranked[MAX_CLAUSES:])
    selected.sort(key=lambda clause: (_POSITION_RANK[clause.position], clause.priority, clause.class_name))
    return selected, omitted_count


def _english_nouns(class_name: str) -> tuple[str, str]:
    parts = class_name.split(" ")
    last = parts[-1]
    if last in _IRREGULAR_PLURALS:
        plural_last = _IRREGULAR_PLURALS[last]
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        plural_last = last[:-1] + "ies"
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        plural_last = last + "es"
    else:
        plural_last = last + "s"
    parts[-1] = plural_last
    return class_name, " ".join(parts)


def _count_word(count: int) -> str:
    return _COUNT_WORDS.get(count, str(count))


def _format_value(distance_m: float) -> str:
    if distance_m >= 10:
        return str(int(round(distance_m)))
    value = round(distance_m, 1)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def _format_meters(distance_m: float) -> str:
    text = _format_value(distance_m)
    unit = "meter" if text == "1" else "meters"
    return f"{text} {unit}"


def _distance_phrase(near_m: float | None, far_m: float | None) -> str:
    if near_m is None or far_m is None:
        return ""
    if far_m <= near_m * _SIMILAR_DISTANCE_RATIO or _format_value(near_m) == _format_value(far_m):
        return f"about {_format_meters(near_m)} away"
    return f"about {_format_value(near_m)} to {_format_meters(far_m)} away"


def _join_english(parts: list[str]) -> str:
    if len(parts) == 1:
        text = parts[0]
    else:
        text = ", ".join(parts[:-1]) + " and " + parts[-1]
    return text[0].upper() + text[1:] + "."


def _clause_phrase(clause: _Clause) -> str:
    distance = _distance_phrase(clause.near_m, clause.far_m)
    if clause.position == "across":
        text = f"multiple {clause.plural} across the scene"
    elif clause.count == 1:
        article = "an" if clause.singular[0].lower() in "aeiou" else "a"
        text = f"{article} {clause.singular} {_POSITION_EN[clause.position]}"
    else:
        noun = clause.plural
        text = f"{_count_word(clause.count)} {noun} {_POSITION_EN[clause.position]}"
    if distance:
        return f"{text}, {distance}"
    return text


def _english(clauses: list[_Clause], omitted_count: int) -> str:
    if not clauses and omitted_count == 0:
        return "No notable objects detected."
    parts = [_clause_phrase(clause) for clause in clauses]
    if omitted_count == 1:
        parts.append("one other object")
    elif omitted_count > 1:
        parts.append(f"{_count_word(omitted_count)} other objects")
    return _join_english(parts)
