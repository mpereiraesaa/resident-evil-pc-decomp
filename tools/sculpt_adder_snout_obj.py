"""Taper the adder's paired upper head and animated lower jaw.

Joint 1 is offset -25 on X at rest. Deforming in their shared rest-space
keeps the two snout tips together throughout the jaw animation.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, offset_x, strength):
    front = max(position[0] + offset_x for position in positions)
    if not 110 <= front <= 130:
        raise ValueError(f"not the expected adder snout: front X {front}")

    def displace(position):
        x, _, z = position
        tip = smoothstep((x + offset_x - 38.0) / 82.0)
        tip *= tip
        return (12.0 * strength * tip,
                -2.0 * strength * tip,
                -0.13 * strength * tip * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["14", "j:14"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100a", "em110a"}:
            raise ValueError(f"selector {selector} is not an adder")
        if rig["joints"][1]["restTranslation"][0] != -25:
            raise ValueError("jaw rest offset changed")
        for joint in (0, 1):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            displacement = make_displacement(metadata[2], -25 if joint == 1 else 0,
                                             options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
