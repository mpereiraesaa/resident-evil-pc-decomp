"""Add facial relief to Enrico's separate head joint without changing his rig."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (215 < bounds[0][1] < 230 and 485 < bounds[1][1] < 505 and
            150 < bounds[2][1] < 170):
        raise ValueError(f"not the expected Enrico head bounds: {bounds}")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 70.0) / 130.0)
        az = abs(z)
        brow = 7.0 * bump(y, 310.0, 27.0) * bump(az, 57.0, 34.0)
        socket = -5.0 * bump(y, 282.0, 23.0) * bump(az, 53.0, 27.0)
        bridge = 4.5 * bump(y, 254.0, 32.0) * bump(z, 0.0, 29.0)
        nose = 7.5 * bump(y, 226.0, 28.0) * bump(z, 0.0, 32.0)
        cheek = 5.5 * bump(y, 223.0, 35.0) * bump(az, 88.0, 40.0)
        mouth = -3.5 * bump(y, 165.0, 17.0) * bump(z, 0.0, 46.0)
        lower_lip = 2.5 * bump(y, 146.0, 17.0) * bump(z, 0.0, 43.0)
        chin = 5.0 * bump(y, 113.0, 29.0) * bump(z, 0.0, 48.0)
        dx = strength * front * (brow + socket + bridge + nose + cheek +
                                 mouth + lower_lip + chin)
        jaw = 0.052 * strength * bump(y, 132.0, 43.0) * bump(az, 96.0, 45.0)
        dz = -z * jaw * front
        return (dx, 0.0, dz)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    rig = json.loads((options.refined_dir / "44_rig.json").read_text())
    if Path(rig["sourcePath"]).stem != "em1028":
        raise ValueError("selector 44 is not Enrico")
    source = options.refined_dir / rig["joints"][1]["obj"]
    metadata = parse_obj(source)
    destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
    triangles = write_mesh(destination, *metadata, 0.1,
                           deformer=make_displacement(metadata[2], options.strength),
                           normal_update=moved_normal)
    print(f"Enrico head -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
