"""Qualitative depth rules. Numeric twin of android/.../vision/Perception.kt.

Larger disparity means closer. That is the MiDaS inverse-depth convention,
not a measured distance. These tests do not prove real-world accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


VERY_NEAR = "VERY_NEAR"
NEAR = "NEAR"
MID = "MID"
FAR = "FAR"
UNKNOWN_PROX = "UNKNOWN"

SUFFICIENT = "SUFFICIENT"
WEAK = "WEAK"
INSUFFICIENT = "INSUFFICIENT"

FRONT = "POSSIBLE_FRONT_OBSTRUCTION"
SIDE = "POSSIBLE_SIDE_SURFACE"
CLEAR = "NO_OBSTRUCTION_DETECTED"
UNKNOWN_REGION = "UNKNOWN"

ASCENDING = "ASCENDING_HYPOTHESIS"
DESCENDING = "DESCENDING_HYPOTHESIS"


@dataclass
class DepthRange:
    far: float
    near: float
    span: float


@dataclass
class ProximityReading:
    proximity: str
    rank: float | None
    reliability: str
    samples: int


@dataclass
class RegionReading:
    state: str
    rank: float | None


@dataclass
class ForwardAssessment:
    left: RegionReading
    center: RegionReading
    right: RegionReading


@dataclass
class StairHypothesis:
    present: bool
    evidence: list[str] = field(default_factory=list)
    uncalibrated_score: float = 0.0
    direction: str | None = None
    status: str = "NO_CANDIDATE"
    suppressed_speech: str | None = None


def percentile(sorted_values: list[float], fraction: float) -> float:
    index = int((len(sorted_values) - 1) * fraction)
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def trimmed_median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    drop = int(len(ordered) * 0.10) if len(ordered) >= 20 else 0
    end = len(ordered) - drop
    if end <= drop:
        return None
    return ordered[(drop + end - 1) // 2]


def collect(frame: list[list[float]], x0: float, x1: float, y0: float, y1: float) -> list[float]:
    side = len(frame)
    left = int(min(1.0, max(0.0, x0)) * (side - 1))
    right = max(left, int(min(1.0, max(0.0, x1)) * (side - 1)))
    top = int(min(1.0, max(0.0, y0)) * (side - 1))
    bottom = max(top, int(min(1.0, max(0.0, y1)) * (side - 1)))
    step = max(1, (right - left + bottom - top) // 24)
    out: list[float] = []
    y = top
    while y <= bottom:
        x = left
        while x <= right:
            value = frame[y][x]
            if math.isfinite(value):
                out.append(value)
            x += step
        y += step
    return out


def depth_range(frame: list[list[float]]) -> DepthRange | None:
    values = collect(frame, 0.20, 0.80, 0.15, 0.95)
    if len(values) < 12:
        return None
    values.sort()
    far = percentile(values, 0.20)
    near = percentile(values, 0.90)
    mid = percentile(values, 0.50)
    span = near - far
    if not math.isfinite(span) or span < 1e-3 or span < abs(mid) * 0.02:
        return None
    return DepthRange(far, near, span)


def median(frame, x0, x1, y0, y1) -> float | None:
    return trimmed_median(collect(frame, x0, x1, y0, y1))


def rank_of(value: float, scale: DepthRange) -> float:
    return min(1.5, max(0.0, (value - scale.far) / scale.span))


def proximity_of(rank: float) -> str:
    if rank >= 0.75:
        return VERY_NEAR
    if rank >= 0.55:
        return NEAR
    if rank >= 0.35:
        return MID
    return FAR


def foreground_value(values: list[float], span: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = percentile(ordered, 0.50)
    closer = percentile(ordered, 0.80)
    if closer - middle > span * 0.15:
        return percentile(ordered, 0.75)
    return trimmed_median(values)


def read_box(frame, box, previous_rank: float | None) -> ProximityReading:
    scale = depth_range(frame)
    if scale is None:
        return ProximityReading(UNKNOWN_PROX, None, INSUFFICIENT, 0)
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    inner = collect(
        frame,
        left + width * 0.25,
        right - width * 0.25,
        top + height * 0.25,
        bottom - height * 0.25,
    )
    value = foreground_value(inner, scale.span)
    if value is None:
        return ProximityReading(UNKNOWN_PROX, None, INSUFFICIENT, 0)
    rank = rank_of(value, scale)
    if previous_rank is not None and abs(rank - previous_rank) > 0.50:
        return ProximityReading(UNKNOWN_PROX, rank, INSUFFICIENT, len(inner))
    reliability = SUFFICIENT
    if len(inner) < 8:
        reliability = INSUFFICIENT
    elif len(inner) < 12 or width * height < 0.02 or left < 0.02 or right > 0.98:
        reliability = WEAK
    if previous_rank is None and reliability == SUFFICIENT:
        reliability = WEAK
    return ProximityReading(proximity_of(rank), rank, reliability, len(inner))


def protrudes(frame, box, scale: DepthRange | None) -> bool:
    if scale is None:
        return False
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    inner = median(frame, left + width * 0.25, right - width * 0.25, top + height * 0.25, bottom - height * 0.25)
    around = median(
        frame,
        max(0.0, left - 0.08),
        min(1.0, right + 0.08),
        max(0.0, top - 0.08),
        min(1.0, bottom + 0.08),
    )
    if inner is None or around is None:
        return False
    return inner - around > scale.span * 0.12


def _column(frame, scale: DepthRange, x0: float, x1: float, front: bool) -> RegionReading:
    upper = median(frame, x0, x1, 0.35, 0.55)
    lower = median(frame, x0, x1, 0.70, 0.90)
    whole = median(frame, x0, x1, 0.38, 0.82)
    if upper is None or lower is None or whole is None:
        return RegionReading(UNKNOWN_REGION, None)
    rank = rank_of(whole, scale)
    flat = abs(upper - lower) < scale.span * 0.22
    receding = lower - upper > scale.span * 0.18
    if front and rank >= 0.62 and flat and not receding:
        state = FRONT
    elif (not front) and rank >= 0.60 and flat and not receding:
        state = SIDE
    elif receding or rank < 0.45:
        state = CLEAR
    else:
        state = UNKNOWN_REGION
    return RegionReading(state, rank)


def analyze_forward(frame) -> ForwardAssessment:
    scale = depth_range(frame)
    unknown = RegionReading(UNKNOWN_REGION, None)
    if scale is None:
        return ForwardAssessment(unknown, unknown, unknown)
    return ForwardAssessment(
        _column(frame, scale, 0.02, 0.30, False),
        _column(frame, scale, 0.34, 0.66, True),
        _column(frame, scale, 0.70, 0.98, False),
    )


def detect_stairs(frame) -> StairHypothesis:
    rows = []
    y0, y1, count = 0.50, 0.98, 32
    for row in range(count):
        top = y0 + (y1 - y0) * row / count
        bottom = y0 + (y1 - y0) * (row + 1) / count
        value = median(frame, 0.30, 0.70, top, bottom)
        if value is None:
            return StairHypothesis(False)
        rows.append(value)
    span = max(rows) - min(rows)
    if span < 1e-3:
        return StairHypothesis(False)
    jumps: list[int] = []
    signs: list[int] = []
    for index in range(1, len(rows) - 1):
        delta = rows[index] - rows[index - 1]
        quiet_before = abs(rows[index - 1] - rows[max(0, index - 2)]) < span * 0.10
        quiet_after = abs(rows[min(len(rows) - 1, index + 1)] - rows[index]) < span * 0.10
        if abs(delta) > span * 0.18 and (quiet_before or quiet_after):
            if not jumps or index - jumps[-1] >= 2:
                jumps.append(index)
                signs.append(1 if delta > 0 else -1)
    if len(jumps) < 3:
        return StairHypothesis(False)
    positive = sum(1 for sign in signs if sign > 0)
    negative = len(signs) - positive
    dominant = max(positive, negative)
    if dominant < len(signs) * 0.8:
        return StairHypothesis(False)
    gaps = [jumps[i + 1] - jumps[i] for i in range(len(jumps) - 1)]
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return StairHypothesis(False)
    variance = sum((gap - mean) ** 2 for gap in gaps) / len(gaps)
    cv = math.sqrt(variance) / mean
    if cv > 0.45:
        return StairHypothesis(False)
    direction = ASCENDING if positive >= negative else DESCENDING
    return StairHypothesis(
        present=True,
        evidence=[
            f"repeated_horizontal_depth_jumps={len(jumps)}",
            f"gap_cv={cv:.2f}",
            "experimental_unvalidated",
        ],
        uncalibrated_score=min(0.40, 0.10 * len(jumps)),
        direction=direction,
        status="EXPERIMENTAL_UNVALIDATED",
        suppressed_speech="احتمال وجود پله در مقابل شما وجود دارد.",
    )


def stair_verdict(learned_labels: list[str] | None) -> str:
    """Depth edges are ignored. None means no licensed detector is loaded."""
    if learned_labels is None:
        return "INSUFFICIENT_EVIDENCE"
    if not learned_labels:
        return "NO_STAIRS_DETECTED"
    names = {label.lower() for label in learned_labels}
    up = any("ascend" in name for name in names)
    down = any("descend" in name for name in names)
    if up and down:
        return "STAIRS_DIRECTION_UNKNOWN"
    if up:
        return "ASCENDING_STAIRS"
    if down:
        return "DESCENDING_STAIRS"
    return "STAIRS_DIRECTION_UNKNOWN"


def _fraction(mask: list[list[int]], x0: float, x1: float, y0: float, y1: float, label: int) -> float:
    height = len(mask)
    width = len(mask[0])
    left = int(min(1.0, max(0.0, x0)) * (width - 1))
    right = max(left, int(min(1.0, max(0.0, x1)) * (width - 1)))
    top = int(min(1.0, max(0.0, y0)) * (height - 1))
    bottom = max(top, int(min(1.0, max(0.0, y1)) * (height - 1)))
    hits = 0
    total = 0
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            total += 1
            if mask[y][x] == label:
                hits += 1
    if total == 0:
        return 0.0
    return hits / total


def front_wall(mask: list[list[int]] | None, wall_id: int = 1, floor_id: int = 2) -> str:
    if mask is None:
        return "UNKNOWN"
    center = _fraction(mask, 0.34, 0.66, 0.25, 0.75, wall_id)
    left = _fraction(mask, 0.02, 0.30, 0.25, 0.80, wall_id)
    right = _fraction(mask, 0.70, 0.98, 0.25, 0.80, wall_id)
    floor = _fraction(mask, 0.30, 0.70, 0.72, 0.98, floor_id)
    labeled = _fraction(mask, 0.05, 0.95, 0.05, 0.95, wall_id) + _fraction(mask, 0.05, 0.95, 0.05, 0.95, floor_id)
    if labeled < 0.15:
        return "UNKNOWN"
    if center >= 0.40 and floor >= 0.12:
        return "POSSIBLE_FRONT_WALL"
    if left >= 0.25 and right >= 0.25 and center < 0.20 and floor >= 0.15:
        return "POSSIBLE_CORRIDOR"
    if left >= 0.35 and center < 0.20:
        return "LEFT_WALL"
    if right >= 0.35 and center < 0.20:
        return "RIGHT_WALL"
    return "UNKNOWN"


def spoken_phrase(front_name: str | None, proximity: str, reliability: str, assessment: ForwardAssessment, depth_fresh: bool) -> str | None:
    """Mirror of PersianCopy.phrase. Stair text is intentionally absent."""
    close = reliability == SUFFICIENT and proximity in {VERY_NEAR, NEAR}
    if front_name and close:
        return f"یک {front_name} روبروی شماست و نزدیک به نظر می‌رسد. فاصله دقیق مانع مشخص نیست."
    if depth_fresh and assessment.center.state == FRONT:
        return "مانعی در قسمت روبه‌روی تصویر دیده می‌شود. فاصله دقیق مانع مشخص نیست."
    if front_name:
        return f"یک {front_name} روبروی شماست. فاصله دقیق مانع مشخص نیست."
    if depth_fresh and assessment.left.state == SIDE:
        return "یک سطح نزدیک در سمت چپ تشخیص داده شده است. فاصله دقیق مشخص نیست."
    if depth_fresh and assessment.right.state == SIDE:
        return "یک سطح نزدیک در سمت راست تشخیص داده شده است. فاصله دقیق مشخص نیست."
    return None


def blank(side: int, value: float = 0.0) -> list[list[float]]:
    return [[value for _ in range(side)] for _ in range(side)]


def paint(frame, x0, x1, y0, y1, value: float) -> None:
    side = len(frame)
    left = int(x0 * side)
    right = int(x1 * side)
    top = int(y0 * side)
    bottom = int(y1 * side)
    for y in range(top, min(side, bottom)):
        for x in range(left, min(side, right)):
            frame[y][x] = value
