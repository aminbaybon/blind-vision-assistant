"""Write synthetic scene JSON. These maps are drawn in code, not photographs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.geometry import analyze_forward, detect_stairs, spoken_phrase  # noqa: E402
from tests.test_perception import front_wall, left_surface, receding, steps  # noqa: E402


def pack(name: str, frame, note: str) -> dict:
    assessment = analyze_forward(frame)
    stair = detect_stairs(frame)
    phrase = spoken_phrase(None, "UNKNOWN", "INSUFFICIENT", assessment, True)
    return {
        "name": name,
        "note": note,
        "real_world_accuracy": False,
        "center": assessment.center.state,
        "left": assessment.left.state,
        "right": assessment.right.state,
        "spoken": phrase,
        "stair_present": stair.present,
        "stair_status": stair.status,
        "stair_direction": stair.direction,
        "stair_spoken": False,
    }


def main() -> None:
    scenes = [
        pack("receding_ground", receding(), "Synthetic ground plane. Not a photograph."),
        pack("near_flat_center", front_wall(), "Synthetic near flat center. Not labeled as a wall."),
        pack("left_near_surface", left_surface(), "Synthetic left surface."),
        pack("rising_steps", steps(rising=True), "Synthetic plateaus. Direction is a hypothesis only."),
        pack("falling_steps", steps(rising=False), "Synthetic plateaus. Direction is a hypothesis only."),
    ]
    out = ROOT / "tests" / "output"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_scenes.json"
    path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
