"""Synthetic checks for qualitative depth. Not a claim of real-world accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.geometry import (  # noqa: E402
    ASCENDING,
    CLEAR,
    DESCENDING,
    FAR,
    FRONT,
    INSUFFICIENT,
    SIDE,
    SUFFICIENT,
    UNKNOWN_PROX,
    UNKNOWN_REGION,
    VERY_NEAR,
    WEAK,
    analyze_forward,
    blank,
    detect_stairs,
    front_wall as classify_wall,
    paint,
    read_box,
    spoken_phrase,
    stair_verdict,
)


def receding(side: int = 64):
    frame = blank(side)
    for y in range(side):
        value = y / (side - 1) * 10.0
        for x in range(side):
            frame[y][x] = value
    return frame


def front_wall(side: int = 64):
    frame = receding(side)
    paint(frame, 0.30, 0.70, 0.30, 0.90, 10.0)
    return frame


def left_surface(side: int = 64):
    frame = receding(side)
    paint(frame, 0.00, 0.28, 0.30, 0.90, 10.0)
    return frame


def steps(side: int = 96, rising: bool = True):
    frame = blank(side, 0.0)
    levels = [1.0, 4.0, 7.0, 10.0] if rising else [10.0, 7.0, 4.0, 1.0]
    edges = [0.50, 0.62, 0.74, 0.86, 0.98]
    last = side - 1
    for band, level in enumerate(levels):
        top = int(edges[band] * last)
        bottom = int(edges[band + 1] * last)
        left = int(0.25 * last)
        right = int(0.75 * last)
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                frame[y][x] = level
    return frame


def test_receding_ground_is_not_called_a_front_obstacle():
    assessment = analyze_forward(receding())
    assert assessment.center.state == CLEAR


def test_near_flat_center_is_a_possible_obstruction_not_a_wall():
    assessment = analyze_forward(front_wall())
    assert assessment.center.state == FRONT
    phrase = spoken_phrase(None, "UNKNOWN", INSUFFICIENT, assessment, True)
    assert phrase == "مانعی در قسمت روبه‌روی تصویر دیده می‌شود. فاصله دقیق مانع مشخص نیست."
    assert "دیوار" not in phrase
    assert "امن" not in phrase
    assert "متر" not in phrase


def test_clear_scene_is_not_described_as_safe():
    assessment = analyze_forward(receding())
    assert spoken_phrase(None, "FAR", "SUFFICIENT", assessment, True) is None


def test_left_surface_uses_the_side_sentence():
    assessment = analyze_forward(left_surface())
    phrase = spoken_phrase(None, "UNKNOWN", INSUFFICIENT, assessment, True)
    assert phrase is not None
    assert "سمت چپ" in phrase
    assert "دیوار" not in phrase


def test_constant_map_is_unknown():
    assessment = analyze_forward(blank(64, 3.0))
    assert assessment.center.state == UNKNOWN_REGION


def test_box_rank_is_qualitative_and_a_jump_is_unknown():
    frame = receding()
    paint(frame, 0.40, 0.60, 0.40, 0.70, 10.0)
    reading = read_box(frame, (0.40, 0.40, 0.60, 0.70), None)
    assert reading.proximity == VERY_NEAR
    jumped = read_box(frame, (0.40, 0.40, 0.60, 0.70), previous_rank=0.0)
    assert jumped.proximity == UNKNOWN_PROX
    assert jumped.rank is not None


def test_far_box_is_not_called_near():
    frame = receding()
    reading = read_box(frame, (0.40, 0.05, 0.60, 0.25), None)
    assert reading.proximity == FAR


def test_smooth_ramp_and_one_edge_are_not_stairs():
    assert detect_stairs(receding(96)).present is False
    one = blank(96, 1.0)
    paint(one, 0.0, 1.0, 0.0, 0.7, 1.0)
    paint(one, 0.0, 1.0, 0.7, 1.0, 8.0)
    assert detect_stairs(one).present is False


def test_repeated_steps_stay_experimental_and_unspoken():
    rising = detect_stairs(steps(rising=True))
    falling = detect_stairs(steps(rising=False))
    assert rising.present and rising.direction == ASCENDING
    assert falling.present and falling.direction == DESCENDING
    assert rising.status == "EXPERIMENTAL_UNVALIDATED"
    assert rising.uncalibrated_score <= 0.40
    assessment = analyze_forward(steps())
    phrase = spoken_phrase(None, "UNKNOWN", INSUFFICIENT, assessment, True)
    assert phrase is None or "پله" not in phrase
    assert rising.suppressed_speech and "پله" in rising.suppressed_speech


def test_first_sighting_is_not_announced_as_close():
    frame = receding()
    paint(frame, 0.30, 0.70, 0.35, 0.80, 10.0)
    box = (0.30, 0.35, 0.70, 0.80)
    first = read_box(frame, box, None)
    second = read_box(frame, box, first.rank)
    assert first.reliability == WEAK
    assert second.reliability == SUFFICIENT
    phrase = spoken_phrase("نفر", first.proximity, first.reliability, analyze_forward(frame), True)
    assert phrase is not None and "نزدیک" not in phrase


def test_depth_pattern_is_not_a_stair_class():
    pattern = detect_stairs(steps(rising=True))
    assert pattern.present
    assert stair_verdict(None) == "INSUFFICIENT_EVIDENCE"
    assert stair_verdict(["ascending"]) == "ASCENDING_STAIRS"
    assert stair_verdict(["descending"]) == "DESCENDING_STAIRS"
    phrase = spoken_phrase(None, "UNKNOWN", INSUFFICIENT, analyze_forward(steps()), True)
    assert phrase is None or "پله" not in phrase


def test_flat_depth_is_not_a_wall_and_a_mask_can_be():
    assert classify_wall(None) == "UNKNOWN"
    side = 40
    mask = [[0 for _ in range(side)] for _ in range(side)]
    for y in range(side):
        for x in range(side):
            if 0.34 <= x / (side - 1) <= 0.66 and y / (side - 1) < 0.75:
                mask[y][x] = 1
            if y / (side - 1) >= 0.72:
                mask[y][x] = 2
    assert classify_wall(mask) == "POSSIBLE_FRONT_WALL"
    phrase = spoken_phrase(None, "UNKNOWN", INSUFFICIENT, analyze_forward(receding()), True)
    assert phrase is None or "دیوار" not in phrase


def test_tile_like_ripple_does_not_become_a_spoken_stair():
    frame = receding(96)
    for y in range(96):
        if (y // 4) % 2 == 0:
            for x in range(96):
                frame[y][x] += 0.4
    stair = detect_stairs(frame)
    phrase = spoken_phrase("نفر", "FAR", "SUFFICIENT", analyze_forward(frame), True)
    assert phrase is not None and "پله" not in phrase
    if stair.present:
        assert stair.status == "EXPERIMENTAL_UNVALIDATED"


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} perception tests passed")


if __name__ == "__main__":
    main()
