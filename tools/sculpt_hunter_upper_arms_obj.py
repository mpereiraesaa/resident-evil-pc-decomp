"""Reshape the Hunter's upper arms while preserving shoulder/elbow rings."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (-235 < bounds[0][0] < -215 and -605 < bounds[1][0] < -580 and
            80 < bounds[1][1] < 100):
        raise ValueError(f"not a Hunter upper arm: {bounds}")

    def displace(position):
        x, y, z = position
        middle = smoothstep((-y - 65.0) / 105.0)
        middle *= 1.0 - smoothstep((-y - 440.0) / 105.0)
        deltoid = 0.18 * bump(y, -240.0, 120.0)
        elbow_taper = -0.17 * bump(y, -445.0, 95.0)
        scale = strength * middle * (deltoid + elbow_taper)
        return (x * scale, 0.0, z * scale)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["10", "j:10"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.4:
        raise ValueError("strength must be in (0, 1.4]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1006", "em1106"}:
            raise ValueError(f"selector {selector} is not Hunter")
        for joint in (4, 7):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} upper arm {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
