"""Add restrained facial volume to Richard's separate head mesh."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (240 < bounds[0][1] < 260 and 465 < bounds[1][1] < 480 and
            125 < bounds[2][1] < 140):
        raise ValueError(f"not the expected Richard head: {bounds}")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 60.0) / 140.0)
        face = 1.0 - smoothstep((y - 345.0) / 55.0)
        az = abs(z)
        relief = (
            8.0 * bump(y, 298.0, 28.0) * bump(az, 51.0, 29.0) -
            5.0 * bump(y, 275.0, 22.0) * bump(az, 49.0, 26.0) +
            6.0 * bump(y, 246.0, 38.0) * bump(z, 0.0, 27.0) +
            9.0 * bump(y, 214.0, 28.0) * bump(z, 0.0, 30.0) +
            6.5 * bump(y, 213.0, 39.0) * bump(az, 74.0, 35.0) -
            3.0 * bump(y, 157.0, 16.0) * bump(z, 0.0, 42.0) +
            3.0 * bump(y, 140.0, 18.0) * bump(z, 0.0, 40.0) +
            5.5 * bump(y, 105.0, 30.0) * bump(z, 0.0, 44.0))
        jaw = 0.045 * bump(y, 126.0, 43.0) * bump(az, 82.0, 38.0)
        return (strength * front * face * relief,
                0.0,
                -strength * front * face * jaw * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    rig = json.loads((options.refined_dir / "43_rig.json").read_text())
    if Path(rig["sourcePath"]).stem != "em1027":
        raise ValueError("selector 43 is not Richard")
    source = options.refined_dir / rig["joints"][1]["obj"]
    metadata = parse_obj(source)
    destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
    triangles = write_mesh(destination, *metadata, 0.1,
                           deformer=make_displacement(metadata[2], options.strength),
                           normal_update=moved_normal)
    print(f"Richard head -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
