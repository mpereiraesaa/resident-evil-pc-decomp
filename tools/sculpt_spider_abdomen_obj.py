"""Round the giant-spider abdomen without changing its animated leg joints.

Joint 19 is the abdomen on Web Spinner and Black Tiger in both scenarios.
The deformation grows the shell's Y/Z cross-section mostly behind its body
attachment, preserving its front seam, original atlas UVs, and joint motion.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    if min(p[0] for p in positions) > -1200 or max(p[1] for p in positions) < 750:
        raise ValueError("not the expected giant-spider abdomen")

    def displace(position):
        x, y, z = position
        attachment = smoothstep((-x - 80.0) / 210.0)
        middle = math.exp(-0.5 * ((x + 640.0) / 390.0) ** 2)
        amount = 0.080 * strength * attachment * middle
        return (0.0, amount * (y - 180.0), amount * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*",
                        default=["7", "8", "j:7", "j:8"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 2.0:
        raise ValueError("strength must be in (0, 2]")
    for selector in options.selectors:
        rig_name = selector.replace(":", "_")
        rig = json.loads((options.refined_dir / f"{rig_name}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1003", "em1004", "em1103", "em1104"}:
            raise ValueError(f"selector {selector} is not a giant-spider EMD")
        source = options.refined_dir / rig["joints"][19]["obj"]
        metadata = parse_obj(source)
        displacement = make_displacement(metadata[2], options.strength)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=displacement, normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
