"""Apply Barry's existing facial sculpt to his dying cutscene head with its UVs."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (250 < bounds[0][1] < 260 and 510 < bounds[1][1] < 520 and
            140 < bounds[2][1] < 150):
        raise ValueError(f"not Barry's head: {bounds}")

    def displace(position):
        x, y, z = position
        front = max(0.0, min(1.0, (x - 130.0) / 95.0))
        az = abs(z)
        relief = (
            8.0 * bump(y, 318.0, 27.0) * bump(az, 53.0, 32.0) -
            6.5 * bump(y, 283.0, 24.0) * bump(az, 53.0, 26.0) +
            4.0 * bump(y, 272.0, 37.0) * bump(z, 0.0, 27.0) +
            7.0 * bump(y, 226.0, 28.0) * bump(z, 0.0, 34.0) +
            5.5 * bump(y, 222.0, 38.0) * bump(az, 87.0, 40.0) -
            4.0 * bump(y, 159.0, 18.0) * bump(z, 0.0, 46.0) +
            3.0 * bump(y, 141.0, 17.0) * bump(z, 0.0, 42.0) +
            5.5 * bump(y, 103.0, 31.0) * bump(z, 0.0, 47.0))
        jaw = 0.055 * bump(y, 118.0, 56.0) * bump(az, 93.0, 43.0)
        return (strength * front * relief, 0.0, -strength * front * jaw * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    rig = json.loads((options.refined_dir / "46_rig.json").read_text())
    if Path(rig["sourcePath"]).stem != "em102a":
        raise ValueError("selector 46 is not dying Barry")
    source = options.refined_dir / rig["joints"][1]["obj"]
    metadata = parse_obj(source)
    destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
    triangles = write_mesh(destination, *metadata, 0.1,
                           deformer=make_displacement(metadata[2], options.strength),
                           normal_update=moved_normal)
    print(f"Dying Barry head -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
