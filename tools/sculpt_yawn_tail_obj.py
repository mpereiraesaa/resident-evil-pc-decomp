"""Shape Yawn's two animated tail sections around their local centrelines."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


YAWN_STEMS = {"em100d", "em1012", "em110d", "em1112"}


def make_displacement(positions, joint, strength):
    xlo, xhi = min(p[0] for p in positions), max(p[0] for p in positions)
    if not -2220 < xlo < -2170 or not 300 < xhi < 335:
        raise ValueError(f"unexpected Yawn tail bounds: {xlo}, {xhi}")
    near_axis = [p for p in positions if .40 < (p[0] - xlo) / (xhi - xlo) < .65]
    cy = sum(p[1] for p in near_axis) / len(near_axis)
    cz = sum(p[2] for p in near_axis) / len(near_axis)

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (x - xlo) / (xhi - xlo)))
        if joint == 13:
            # Reduce its middle swell; both articulated ends remain fixed.
            scale = -0.09 * strength * math.sin(math.pi * u) ** 2
        else:
            # The tip section keeps its broad root but narrows progressively
            # toward its free end without extending or shortening it.
            scale = -0.16 * strength * smoothstep((0.82 - u) / 0.82)
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
        for joint in (13, 14):
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
