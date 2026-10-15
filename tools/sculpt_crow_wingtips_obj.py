"""Shape the crow's two outer wing feather joints without moving their hinges."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    if max(abs(p[2]) for p in positions) < 580:
        raise ValueError("not the expected crow outer wing")

    def displace(position):
        _, _, z = position
        tip = smoothstep((abs(z) - 185.0) / 365.0)
        return (0.0, -14.0 * strength * tip * tip,
                math.copysign(35.0 * strength * tip * tip, z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["9", "j:9"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1005", "em1105"}:
            raise ValueError(f"selector {selector} is not a crow")
        for joint in (5, 8):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            displacement = make_displacement(metadata[2], options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
