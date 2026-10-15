"""Shape Chimera's long shins and broad feet without moving their joints."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


CHIMERA_STEMS = {"em1009", "em1109"}


def calf_displacement(positions, strength):
    xlo = min(p[0] for p in positions)
    ylo = min(p[1] for p in positions)
    yhi = max(p[1] for p in positions)
    if not (-385 < xlo < -365 and -765 < ylo < -745 and 45 < yhi < 65):
        raise ValueError(f"unexpected Chimera shin: {xlo}, {ylo}, {yhi}")

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (y - ylo) / (yhi - ylo)))
        muscle = math.sin(math.pi * u) ** 2
        rear = 1.0 - smoothstep((x + 290.0) / 230.0)
        front = smoothstep((x + 95.0) / 175.0)
        return (strength * muscle * (-47.0 * rear + 11.0 * front),
                0.0,
                strength * muscle * 0.06 * z)

    return displace


def foot_displacement(positions, strength):
    xlo = min(p[0] for p in positions)
    xhi = max(p[0] for p in positions)
    if not (-135 < xlo < -115 and 540 < xhi < 560):
        raise ValueError(f"unexpected Chimera foot: {xlo}, {xhi}")
    centre_z = (min(p[2] for p in positions) + max(p[2] for p in positions)) * 0.5

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (x - 240.0) / (xhi - 240.0)))
        tip = smoothstep(u)
        return (36.0 * strength * tip,
                0.0,
                -0.27 * strength * tip * (z - centre_z))

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
        if Path(rig["sourcePath"]).stem not in CHIMERA_STEMS or len(rig["joints"]) != 17:
            raise ValueError(f"selector {selector} is not Chimera")
        for joint in (12, 13, 15, 16):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            deformation = (calf_displacement if joint in (12, 15)
                           else foot_displacement)(metadata[2], options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=deformation, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
