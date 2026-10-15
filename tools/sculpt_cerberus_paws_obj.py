"""Lengthen Cerberus's individual front and rear toe silhouettes.

Only the forward tips near the bottom of each paw move. The leg connections,
UVs, and rigid joint animation stay on the original skeleton.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength, front):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    xmax = max(p[0] for p in positions)
    if front:
        if not -845 < ymin < -815 or not 175 < xmax < 195:
            raise ValueError("not the expected Cerberus front paw")
        foot_top = -520.0
        toe_start = 65.0
    else:
        if not -390 < ymin < -365 or not 210 < xmax < 230:
            raise ValueError("not the expected Cerberus rear paw")
        foot_top = -170.0
        toe_start = 85.0
    toe_vertices = [p for p in positions
                    if p[1] < ymin + 0.30 * (ymax - ymin) and p[0] > toe_start]
    zlow = min(p[2] for p in toe_vertices)
    zhigh = max(p[2] for p in toe_vertices)
    toe_centers = [zlow + (zhigh - zlow) * (i + 0.5) / 3.0 for i in range(3)]

    def displace(position):
        x, y, z = position
        toe = smoothstep((x - toe_start) / (xmax - toe_start))
        low = smoothstep((foot_top - y) / (foot_top - ymin))
        weight = toe * low
        toe_z = min(toe_centers, key=lambda centre: abs(z - centre))
        return (27.0 * strength * weight,
                0.0,
                -0.13 * strength * weight * (z - toe_z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["6", "j:6"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1002", "em1102"}:
            raise ValueError(f"selector {selector} is not Cerberus")
        for joint in (6, 8, 11, 14):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2],
                                                              options.strength,
                                                              joint in (6, 8)),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
