"""Slope the bare zombie's outer torso shoulder ridge toward its arm sockets."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    width = max(abs(p[2]) for p in positions)
    if not -60 < ymin < -40 or not 710 < ymax < 730 or not 365 < width < 380:
        raise ValueError(f"not the expected naked zombie torso: {ymin}, {ymax}, {width}")

    def displace(position):
        _, y, z = position
        side = smoothstep((abs(z) - 0.53 * width) / (0.47 * width))
        shoulder = bump(y, 590.0, 118.0)
        return (0.0, -52.0 * strength * side * shoulder, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["5", "j:5"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1001", "em1101"}:
            raise ValueError(f"selector {selector} is not a naked zombie EMD")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} joint 1 -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
