"""Shape the palm and articulated fingers of Chris's typing close-ups.

The four EMDs each have a forearm, palm, and fourteen finger segments. This
keeps every finger root and intermediate joint end fixed; terminal tips are
rounded inward without extending keyboard contact distance.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


HAND_STEMS = {"em1014", "em1015", "em1114", "em1115"}
TIP_JOINTS = {3, 6, 9, 12, 15}


def make_displacement(positions, joint, strength):
    xlo = min(p[0] for p in positions)
    xhi = max(p[0] for p in positions)
    span = xhi - xlo
    if joint == 1:
        if not 160 < xhi < 210 or not -50 < xlo < -30:
            raise ValueError(f"unexpected typing palm: {xlo}, {xhi}")

        def palm(position):
            x, y, z = position
            centre = bump(x, 0.57 * xhi, 0.28 * xhi)
            width = 1.0 - smoothstep((abs(z) - 45.0) / 60.0)
            return (0.0, 5.5 * strength * centre * width * smoothstep(y / 40.0),
                    0.035 * strength * centre * z)

        return palm

    if not 30.0 < span < 125.0:
        raise ValueError(f"unexpected typing finger span: joint {joint} {span}")
    middle = [p for p in positions if 0.35 < (p[0] - xlo) / span < 0.65]
    cy = (min(p[1] for p in middle) + max(p[1] for p in middle)) * 0.5
    cz = (min(p[2] for p in middle) + max(p[2] for p in middle)) * 0.5
    tip = joint in TIP_JOINTS

    def finger(position):
        x, y, z = position
        u = max(0.0, min(1.0, (x - xlo) / span))
        knuckle = math.sin(math.pi * u) ** 4
        fullness = 0.10 * strength * knuckle
        taper = 0.22 * strength * smoothstep((u - 0.61) / 0.39) if tip else 0.0
        scale = fullness - taper
        return (0.0, (y - cy) * scale, (z - cz) * scale)

    return finger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["24", "25", "j:24", "j:25"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in HAND_STEMS or len(rig["joints"]) != 16:
            raise ValueError(f"selector {selector} is not a typing hand")
        for joint in range(1, 16):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], joint,
                                                                options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
