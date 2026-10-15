"""Lengthen and taper the separate white/green zombie finger silhouettes."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


ZOMBIE_STEMS = {"em1000", "em1011", "em1100", "em1111"}


def make_displacement(positions, strength, joint):
    zmin = min(p[2] for p in positions)
    zmax = max(p[2] for p in positions)
    span = zmax - zmin
    if not 360 < span < 420 or joint not in (5, 8):
        raise ValueError(f"not the expected zombie hand: joint {joint}, Z span {span}")
    distal = [p for p in positions if
              (p[2] < zmin + 0.18 * span if joint == 5
               else p[2] > zmax - 0.18 * span)]
    xlo = min(p[0] for p in distal)
    xhi = max(p[0] for p in distal)
    ycentre = (min(p[1] for p in distal) + max(p[1] for p in distal)) * 0.5
    centers = [xlo + (xhi - xlo) * (i + 0.5) / 4.0 for i in range(4)]

    def displace(position):
        x, y, z = position
        along = (zmax - z) / span if joint == 5 else (z - zmin) / span
        weight = smoothstep((along - 0.48) / 0.52) ** 2
        center = min(centers, key=lambda value: abs(x - value))
        return (-0.18 * strength * weight * (x - center),
                -0.10 * strength * weight * (y - ycentre),
                (-18.0 if joint == 5 else 18.0) * strength * weight)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["4", "21", "j:4", "j:21"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in ZOMBIE_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not a white/green zombie EMD")
        for joint in (5, 8):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength,
                                                              joint),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
