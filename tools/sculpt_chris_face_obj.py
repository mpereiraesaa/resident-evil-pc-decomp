"""Sculpt the shared Chris facial anatomy in player, NPC, and outfit EMDs."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    xmax = max(p[0] for p in positions)
    ymax = max(p[1] for p in positions)
    zmax = max(abs(p[2]) for p in positions)
    if not 220 <= xmax <= 226 or not 430 <= ymax <= 440 or not 120 <= zmax <= 130:
        raise ValueError(f"not the expected Chris head: {xmax}, {ymax}, {zmax}")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 65.0) / 120.0)
        az = abs(z)
        brow = 6.0 * bump(y, 296.0, 24.0) * bump(az, 44.0, 30.0)
        socket = -4.5 * bump(y, 266.0, 22.0) * bump(az, 42.0, 23.0)
        bridge = 3.5 * bump(y, 238.0, 30.0) * bump(z, 0.0, 23.0)
        nose = 6.0 * bump(y, 214.0, 25.0) * bump(z, 0.0, 27.0)
        cheek = 4.5 * bump(y, 210.0, 31.0) * bump(az, 66.0, 32.0)
        mouth = -2.8 * bump(y, 165.0, 14.0) * bump(z, 0.0, 34.0)
        lower_lip = 2.0 * bump(y, 147.0, 13.0) * bump(z, 0.0, 33.0)
        chin = 4.5 * bump(y, 111.0, 28.0) * bump(z, 0.0, 38.0)
        dx = strength * front * (brow + socket + bridge + nose + cheek +
                                 mouth + lower_lip + chin)
        jaw = 0.055 * strength * bump(y, 119.0, 39.0) * bump(az, 76.0, 34.0)
        return (dx, 0.0, -z * jaw * front)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["0", "36", "51", "52"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    valid = {"char10", "em1020", "em1030", "em1032"}
    for selector in options.selectors:
        rig = json.loads((options.refined_dir / f"{selector}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in valid:
            raise ValueError(f"selector {selector} is not a Chris model")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
