"""Lengthen and taper Neptune's paired upper head and animated lower jaw.

The lower jaw's rest X offset is +165. Applying the same deformation in that
shared rest-space X keeps both tips aligned while preserving the jaw hinge.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, offset_x, strength):
    if max(p[0] + offset_x for p in positions) < 680:
        raise ValueError("not the expected Neptune snout")

    def displace(position):
        x, _, z = position
        tip = smoothstep((x + offset_x - 330.0) / 370.0)
        tip *= tip
        return (32.0 * strength * tip,
                0.0,
                -0.15 * strength * tip * z)

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
        if rig["joints"][1]["restTranslation"][0] != 165:
            raise ValueError("jaw rest offset changed")
        for joint in (0, 1):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            displacement = make_displacement(metadata[2], 165 if joint == 1 else 0,
                                             options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
