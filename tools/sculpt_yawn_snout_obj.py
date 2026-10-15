"""Refine Yawn's upper head and animated lower jaw across both encounters.

The jaw's rest translation is +425 X. Both halves are shaped in shared
rest-space so their animated hinge and original gap remain intact.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, offset_x, strength):
    front = max(position[0] + offset_x for position in positions)
    if not 1300 <= front <= 1650:
        raise ValueError(f"not the expected Yawn snout: front X {front}")

    def displace(position):
        x, _, z = position
        tip = smoothstep((x + offset_x - 350.0) / 1150.0)
        tip *= tip
        return (54.0 * strength * tip,
                0.0,
                -0.10 * strength * tip * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["17", "22", "j:17", "j:22"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100d", "em1012", "em110d", "em1112"}:
            raise ValueError(f"selector {selector} is not Yawn")
        if rig["joints"][1]["restTranslation"][0] != 425:
            raise ValueError("jaw rest offset changed")
        for joint in (0, 1):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            displacement = make_displacement(metadata[2], 425 if joint == 1 else 0,
                                             options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
