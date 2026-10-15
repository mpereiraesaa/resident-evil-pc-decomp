"""Add calf and shin relief to Tyrant's lower legs without moving sockets."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def make_displacement(positions, side, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (-255 < bounds[0][0] < -235 and -1050 < bounds[1][0] < -1010 and
            95 < bounds[1][1] < 120):
        raise ValueError(f"not the expected Tyrant calf: {bounds}")
    centre_z = -45.0 if side == 0 else 45.0

    def displace(position):
        x, y, z = position
        shaft = smoothstep((-y - 90.0) / 115.0)
        shaft *= 1.0 - smoothstep((-y - 830.0) / 140.0)
        calf = strength * shaft * bump(y, -375.0, 155.0)
        shin = strength * shaft * bump(y, -650.0, 170.0)
        rear = 1.0 - smoothstep((x + 75.0) / 155.0)
        front = smoothstep((x + 85.0) / 150.0)
        flank = smoothstep((abs(z - centre_z) - 70.0) / 85.0)
        dx = -31.0 * calf * rear + 10.0 * shin * front
        dz = math.copysign(17.0 * calf * flank, z - centre_z)
        return (dx, 0.0, dz)

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
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS:
            raise ValueError(f"selector {selector} is not Tyrant")
        for side, joint in enumerate((10, 13)):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(
                destination, *metadata, 0.1,
                deformer=make_displacement(metadata[2], side, options.strength),
                normal_update=moved_normal)
            print(f"{selector} calf {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
