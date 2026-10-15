"""Add anatomical relief to Tyrant's exposed chest and abdomen."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (350 < bounds[0][1] < 370 and 1400 < bounds[1][1] < 1450 and
            620 < bounds[2][1] < 640):
        raise ValueError(f"not the expected Tyrant torso: {bounds}")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 80.0) / 185.0)
        body = smoothstep((y - 155.0) / 85.0)
        body *= 1.0 - smoothstep((y - 1050.0) / 180.0)
        body *= 1.0 - smoothstep((abs(z) - 390.0) / 125.0)
        az = abs(z)
        pec = 28.0 * bump(y, 825.0, 105.0) * bump(az, 220.0, 125.0)
        sternum = -10.0 * bump(y, 825.0, 120.0) * bump(z, 0.0, 40.0)
        ribs = 11.0 * bump(y, 630.0, 80.0) * bump(az, 310.0, 85.0)
        abs_upper = 13.0 * bump(y, 515.0, 58.0) * bump(az, 120.0, 72.0)
        abs_lower = 10.0 * bump(y, 365.0, 56.0) * bump(az, 115.0, 75.0)
        groove = -5.0 * bump(y, 460.0, 155.0) * bump(z, 0.0, 35.0)
        return (strength * front * body *
                (pec + sternum + ribs + abs_upper + abs_lower + groove),
                0.0, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS:
            raise ValueError(f"selector {selector} is not Tyrant")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} torso -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
