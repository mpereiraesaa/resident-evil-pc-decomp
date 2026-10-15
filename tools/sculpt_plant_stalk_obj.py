"""Add soft organic volume to the two segmented plant families.

Each joint retains its first and last axial rings so the animated chain keeps
its original seam positions. The deformation is applied to all drawable joints
in both scenario EMDs of Plant 42 and the hanging monster plant.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, joint, strength, pointed_tip):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    low, high = min(ys), max(ys)
    if high - low < 100 or max(zs) - min(zs) < 50:
        raise ValueError(f"not an elongated plant segment: joint {joint}")
    centre_x = (min(xs) + max(xs)) * 0.5
    centre_z = (min(zs) + max(zs)) * 0.5
    phase = joint * 0.63

    def displace(position):
        x, y, z = position
        t = max(0.0, min(1.0, (y - low) / (high - low)))
        envelope = math.sin(math.pi * t) ** 2
        rib = 0.8 + 0.2 * math.sin(4.0 * math.pi * t + phase)
        scale = 0.12 * strength * envelope * rib
        tip = max(0.0, 1.0 - t / 0.55) ** 2 if pointed_tip else 0.0
        scale -= 0.28 * strength * tip
        return (scale * (x - centre_x), -85.0 * strength * tip,
                scale * (z - centre_z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["12", "19", "j:12", "j:19"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1008", "em100f", "em1108", "em110f"}:
            raise ValueError(f"selector {selector} is not a plant stalk")
        for joint, entry in enumerate(rig["joints"]):
            source = options.refined_dir / entry["obj"]
            metadata = parse_obj(source)
            pointed_tip = joint == 0 and Path(rig["sourcePath"]).stem in {"em1008", "em1108"}
            displacement = make_displacement(metadata[2], joint, options.strength,
                                             pointed_tip)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
