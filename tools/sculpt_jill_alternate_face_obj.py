"""Bring Jill's two alternate-outfit faces in line with her main face sculpt."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    xmax = max(p[0] for p in positions)
    ymax = max(p[1] for p in positions)
    zmax = max(abs(p[2]) for p in positions)
    if not 165 <= xmax <= 171 or not 390 <= ymax <= 400 or not 140 <= zmax <= 145:
        raise ValueError(f"not the expected alternate Jill head: {xmax}, {ymax}, {zmax}")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 82.0) / 42.0)
        az = abs(z)
        eye = -5.5 * bump(y, 252.0, 24.0) * bump(az, 42.0, 22.0)
        cheek = 7.0 * bump(y, 200.0, 30.0) * bump(az, 55.0, 32.0)
        bridge = 3.5 * bump(y, 220.0, 24.0) * bump(z, 0.0, 22.0)
        lip = 2.5 * bump(y, 152.0, 10.0) * bump(z, 0.0, 31.0)
        chin = 3.5 * bump(y, 116.0, 23.0) * bump(z, 0.0, 36.0)
        dx = front * strength * (eye + cheek + bridge + lip + chin)
        jaw = 0.065 * strength * bump(y, 131.0, 39.0) * bump(az, 63.0, 35.0)
        return (dx, 0.0, -z * jaw * front)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["j:51", "j:52"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1031", "em1033"}:
            raise ValueError(f"selector {selector} is not an alternate Jill model")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
