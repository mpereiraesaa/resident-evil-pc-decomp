"""Taper the separate human hand's distal fingers without changing the wrist.

Human rig joints 11 and 14 are hands (verified with a viewer displacement
probe). Their fingers extend toward local negative Y; local Z spans the palm.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_human_toes_obj import HUMAN_STEMS
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    xmin = min(p[0] for p in positions)
    xmax = max(p[0] for p in positions)
    zmin = min(p[2] for p in positions)
    zmax = max(p[2] for p in positions)
    span = ymax - ymin
    if not 230 < span < 430 or not 100 < xmax - xmin < 280 or not 80 < zmax - zmin < 240:
        raise ValueError(f"not an expected hand: {xmin, xmax, ymin, ymax, zmin, zmax}")
    distal = [p for p in positions if p[1] < ymin + 0.27 * span]
    finger_low = min(p[2] for p in distal)
    finger_high = max(p[2] for p in distal)
    centers = [finger_low + (finger_high - finger_low) * i / 4.0 for i in range(5)]
    centre_x = (min(p[0] for p in distal) + max(p[0] for p in distal)) * 0.5

    def displace(position):
        x, y, z = position
        distal_weight = smoothstep((ymin + 0.58 * span - y) / (0.58 * span))
        distal_weight *= distal_weight
        finger_z = min(centers, key=lambda centre: abs(z - centre))
        return (-0.10 * strength * distal_weight * (x - centre_x),
                -8.0 * strength * distal_weight,
                -0.16 * strength * distal_weight * (z - finger_z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="+", help="viewer selectors, e.g. 0 1 36 j:51")
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in HUMAN_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not a known human EMD")
        for joint in (11, 14):
            # em1028's right-hand mesh extends 574 units along Y, unlike the
            # 299..403-unit hand set. Its fingers need separate inspection.
            if Path(rig["sourcePath"]).stem == "em1028" and joint == 14:
                print(f"{selector} joint {joint}: skipped nonstandard hand")
                continue
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
