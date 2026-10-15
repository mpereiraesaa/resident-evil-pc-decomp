"""Refine Hunter toe-tip contours while keeping the animated foot root fixed."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength, joint):
    xmax = max(p[0] for p in positions)
    xmin = min(p[0] for p in positions)
    if not 620.0 < xmax < 640.0 or not -180.0 < xmin < -165.0:
        raise ValueError(f"not an expected Hunter foot: {xmin}, {xmax}")
    # The two long toes occupy distinct Z bands; the opposite foot mirrors
    # them. Keep the central body and small outer toe out of the taper.
    centres = (-205.0, -20.0) if joint == 12 else (20.0, 205.0)

    def displace(position):
        x, y, z = position
        distal = smoothstep((x - 300.0) / 325.0) ** 2
        nearest = min(centres, key=lambda c: abs(z - c))
        toe = smoothstep((85.0 - abs(z - nearest)) / 55.0)
        w = strength * distal * toe
        return (45.0 * w,
                -0.06 * w * (y + 100.0),
                -0.12 * w * (z - nearest))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["10", "j:10"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1006", "em1106"}:
            raise ValueError(f"selector {selector} is not Hunter")
        for joint in (12, 15):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength,
                                                              joint),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
