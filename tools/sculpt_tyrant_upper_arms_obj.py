"""Round Tyrant's exposed and armored upper-arm bellies between their joints."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def make_displacement(positions, strength, bare):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    span = ymax - ymin
    if bare:
        if not 900 < span < 950:
            raise ValueError(f"not the expected bare Tyrant arm: {span}")
    elif not 830 < span < 880:
        raise ValueError(f"not the expected armored Tyrant arm: {span}")
    cx = (min(p[0] for p in positions) + max(p[0] for p in positions)) * 0.5
    cz = (min(p[2] for p in positions) + max(p[2] for p in positions)) * 0.5

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (y - ymin) / span))
        weight = math.sin(math.pi * u) ** 4
        dx = x - cx
        dz = z - cz
        radius = math.hypot(dx, dz)
        if radius < 1.0:
            return (0.0, 0.0, 0.0)
        fullness = (30.0 if bare else 20.0) * strength * weight
        return (dx / radius * fullness + (10.0 if bare else 5.0) * strength * weight,
                0.0,
                dz / radius * fullness)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not Tyrant")
        for joint in (3, 6):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2],
                                                              options.strength,
                                                              joint == 3),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
