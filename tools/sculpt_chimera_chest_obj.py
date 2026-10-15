"""Add shallow paired chest volume to the Chimera's rigid torso joint.

The central front is +X in this OBJ. The sculpt fades before the wing, arm,
neck, and abdomen joins; UVs and the original joint animation remain intact.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    low_x = min(p[0] for p in positions)
    high_x = max(p[0] for p in positions)
    wide_z = max(abs(p[2]) for p in positions)
    if not low_x < -300 or not high_x > 270 or not wide_z > 700:
        raise ValueError("not the expected Chimera chest/wing joint")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 20.0) / 150.0)
        central = 1.0 - smoothstep((abs(z) - 255.0) / 95.0)
        upper = bump(y, 25.0, 125.0)
        paired = bump(abs(z), 175.0, 115.0)
        sternum = bump(z, 0.0, 65.0)
        dx = front * central * upper * strength * (23.0 * paired - 5.0 * sternum)
        return (dx, 0.0, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["13", "j:13"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1009", "em1109"}:
            raise ValueError(f"selector {selector} is not Chimera")
        source = options.refined_dir / rig["joints"][2]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} joint 2 -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
