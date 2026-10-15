"""Sweep Neptune's paired rigid pectoral-fin tips outward and back."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, side, strength):
    ymin = min(p[1] for p in positions)
    ztip = min(p[2] for p in positions) if side < 0 else max(p[2] for p in positions)
    if not -860 < ymin < -835 or not 395 < side * ztip < 420:
        raise ValueError("not the expected Neptune pectoral fin")

    def displace(position):
        _, y, z = position
        tip = smoothstep((-y - 300.0) / (-ymin - 300.0))
        tip *= tip
        outside = smoothstep((side * z - 80.0) / 260.0)
        weight = tip * outside
        return (-29.0 * strength * weight,
                -48.0 * strength * weight,
                side * 36.0 * strength * weight)

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
        for joint, side in ((3, -1), (4, 1)):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], side,
                                                              options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
