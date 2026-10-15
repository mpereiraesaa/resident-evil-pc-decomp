"""Lengthen Tyrant's short head-mesh neck without moving its joint base.

The head is rigidly attached to torso joint 1. The lower head vertices remain
at the original attachment while the face rises gradually above them. The
existing Tyrant facial sculpt is composed with the neck change.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import make_face_displacement, moved_normal


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def make_displacement(positions, length):
    face_displace = make_face_displacement(positions, 1.0)
    if not (0.0 < min(p[1] for p in positions) < 20.0 and
            700.0 < max(p[1] for p in positions) < 750.0):
        raise ValueError("not the expected Tyrant head height")

    def displace(position):
        x, y, z = face_displace(position)
        # The first 20 units of the neck stay at the existing torso socket.
        rise = length * smoothstep((position[1] - 20.0) / 270.0)
        return (x, y + rise, z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=float, default=95.0)
    parser.add_argument("selectors", nargs="*", default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 < options.length <= 150.0:
        raise ValueError("length must be in (0, 150]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS:
            raise ValueError(f"selector {selector} is not Tyrant")
        source = options.refined_dir / rig["joints"][2]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.length),
                               normal_update=moved_normal)
        print(f"{selector} head/neck -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
