"""Add restrained cloth volume to the human pelvis joint.

This is joint 2 of the standing character rigs. The belt and upper-leg
attachment bands are kept in place so rigid animation remains aligned.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    low_y, high_y = bounds[1]
    width = max(abs(bounds[2][0]), abs(bounds[2][1]))
    if not (250 < high_y - low_y < 500 and 190 < width < 380 and
            bounds[0][1] > 150):
        raise ValueError(f"not a standing human pelvis: {bounds}")
    height = high_y - low_y

    def displace(position):
        x, y, z = position
        t = (high_y - y) / height  # 0: belt, 1: upper-leg attachment
        cloth = smoothstep((t - 0.10) / 0.20)
        cloth *= 1.0 - smoothstep((t - 0.72) / 0.20)
        front = smoothstep((x - 10.0) / 160.0)
        hip = bump(abs(z), 0.48 * width, 0.20 * width)
        seam = bump(z, 0.0, 0.15 * width)
        dx = strength * front * cloth * (
            18.0 * bump(t, 0.49, 0.22) * hip -
            9.0 * bump(t, 0.56, 0.20) * seam)
        side = smoothstep((abs(z) - 0.56 * width) / (0.28 * width))
        dz = strength * (1.0 if z >= 0 else -1.0) * cloth * side * (
            11.0 * bump(t, 0.43, 0.18))
        return (dx, 0.0, dz)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="+", help="model viewer selectors")
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.7:
        raise ValueError("strength must be in (0, 1.7]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        source = options.refined_dir / rig["joints"][2]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} pelvis -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
