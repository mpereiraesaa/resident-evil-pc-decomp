"""Round Yawn's animated body segments without moving their joint ends."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


YAWN_STEMS = {"em100d", "em1012", "em110d", "em1112"}
BODY_JOINTS = tuple(range(3, 13))


def make_displacement(positions, strength):
    xlow = min(p[0] for p in positions)
    xhigh = max(p[0] for p in positions)
    span = xhigh - xlow
    if not 1300.0 < span < 1800.0:
        raise ValueError(f"not an expected Yawn body segment: {span}")

    def displace(position):
        x, y, z = position
        t = max(0.0, min(1.0, (x - xlow) / span))
        # Leave the rigid attachment rings unchanged. A smooth midsection
        # roundover removes the long cylinder silhouette without opening the
        # animated seams between adjacent joints.
        middle = math.sin(math.pi * t) ** 2
        return (0.0,
                0.11 * strength * middle * y,
                0.085 * strength * middle * z)

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
        if Path(rig["sourcePath"]).stem not in YAWN_STEMS:
            raise ValueError(f"selector {selector} is not Yawn")
        for joint in BODY_JOINTS:
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
