"""Shape the naked zombie's fingers embedded in its forearm meshes."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength, joint):
    zmin = min(p[2] for p in positions)
    zmax = max(p[2] for p in positions)
    span = zmax - zmin
    if not 700 < span < 750 or joint not in (4, 7):
        raise ValueError(f"not the expected naked-zombie forearm: {span}")
    distal = [p for p in positions if
              (p[2] < zmin + 0.10 * span if joint == 4
               else p[2] > zmax - 0.10 * span)]
    xlow = min(p[0] for p in distal)
    xhigh = max(p[0] for p in distal)
    ycentre = (min(p[1] for p in distal) + max(p[1] for p in distal)) * 0.5
    centers = [xlow + (xhigh - xlow) * (i + 0.5) / 4.0 for i in range(4)]

    def displace(position):
        x, y, z = position
        along = (zmax - z) / span if joint == 4 else (z - zmin) / span
        tip = smoothstep((along - 0.68) / 0.32) ** 2
        nearest = min(centers, key=lambda value: abs(x - value))
        return (-0.16 * strength * tip * (x - nearest),
                -0.09 * strength * tip * (y - ycentre),
                (-27.0 if joint == 4 else 27.0) * strength * tip)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["5", "j:5"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1001", "em1101"}:
            raise ValueError(f"selector {selector} is not naked zombie")
        for joint in (4, 7):
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
