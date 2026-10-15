"""Give Cerberus's long front shins tendon and calf volume.

The upper and paw attachment bands stay fixed so existing animation joints
remain aligned. This writes trial render meshes for both scenario EMDs.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    low = min(p[1] for p in positions)
    high = max(p[1] for p in positions)
    if not -850 < low < -810 or not 90 < high < 115:
        raise ValueError("not the expected Cerberus lower front leg")

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (y - low) / (high - low)))
        belly = math.sin(math.pi * u) ** 2
        # The exposed outside contour is fuller in the upper shin; a narrow
        # tendon remains toward the paw. Radial expansion preserves UVs.
        strength_here = strength * belly * (0.13 + 0.09 * u)
        centre_x = 3.0 - 18.0 * (1.0 - u)
        return (strength_here * (x - centre_x), 0.0,
                strength_here * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["6", "j:6"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1002", "em1102"}:
            raise ValueError(f"selector {selector} is not Cerberus")
        for joint in (6, 8):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
