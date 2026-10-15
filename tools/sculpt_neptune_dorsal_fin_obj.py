"""Sweep Neptune's rigid dorsal-fin tip without changing its body attachment."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    low = min(p[1] for p in positions)
    high = max(p[1] for p in positions)
    if not -110 < low < -80 or not 820 < high < 850:
        raise ValueError(f"not the expected Neptune dorsal fin: {low}, {high}")

    def displace(position):
        _, y, z = position
        tip = smoothstep((y - 150.0) / (high - 150.0))
        tip *= tip
        return (-70.0 * strength * tip,
                75.0 * strength * tip,
                -0.12 * strength * tip * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["15", "j:15"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100b", "em110b"}:
            raise ValueError(f"selector {selector} is not Neptune")
        source = options.refined_dir / rig["joints"][6]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} joint 6 -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
