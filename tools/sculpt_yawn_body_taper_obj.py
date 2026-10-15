"""Reduce Yawn's repeated rigid-section body bulges about each section axis."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


YAWN_STEMS = {"em100d", "em1012", "em110d", "em1112"}
BODY_JOINTS = tuple(range(3, 13))


def make_displacement(positions, joint, strength):
    xlo, xhi = min(p[0] for p in positions), max(p[0] for p in positions)
    span = xhi - xlo
    if not 1300 < span < 1800:
        raise ValueError(f"unexpected Yawn body length at joint {joint}: {span}")
    mid = [p for p in positions if 0.40 < (p[0] - xlo) / span < 0.60]
    if len(mid) < 50:
        raise ValueError("too few midsection vertices")
    cy = sum(p[1] for p in mid) / len(mid)
    cz = sum(p[2] for p in mid) / len(mid)
    # Front segments have the largest repeated swell; taper the already
    # smaller rear sections less to avoid an hourglass silhouette.
    taper = 0.12 if joint <= 5 else 0.08 if joint == 6 else 0.045

    def displace(position):
        x, y, z = position
        t = max(0.0, min(1.0, (x - xlo) / span))
        scale = -strength * taper * math.sin(math.pi * t) ** 2
        return (0.0, (y - cy) * scale, (z - cz) * scale)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["17", "22", "j:17", "j:22"])
    options = parser.parse_args()
    if not 0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in YAWN_STEMS:
            raise ValueError(f"selector {selector} is not Yawn")
        for joint in BODY_JOINTS:
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            output = options.out / source.name.replace("_refined.obj", ".r1mesh")
            count = write_mesh(output, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], joint,
                                                            options.strength),
                               normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {output} ({count} triangles)")


if __name__ == "__main__":
    main()
