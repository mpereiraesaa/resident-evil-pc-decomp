"""Add shallow lower-face volume to Forest's damaged corpse head."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (515 < bounds[0][1] < 535 and 500 < bounds[1][1] < 515 and
            165 < bounds[2][1] < 180):
        raise ValueError(f"not Forest's head: {bounds}")

    def displace(position):
        x, y, z = position
        # The long hair and forehead wound extend beyond this face region.
        face = smoothstep((x - 175.0) / 145.0)
        face *= 1.0 - smoothstep((x - 400.0) / 100.0)
        face *= 1.0 - smoothstep((y - 300.0) / 70.0)
        az = abs(z)
        relief = (
            6.0 * bump(y, 254.0, 34.0) * bump(z, 0.0, 30.0) +
            8.0 * bump(y, 224.0, 30.0) * bump(z, 0.0, 33.0) +
            6.0 * bump(y, 218.0, 39.0) * bump(az, 80.0, 37.0) -
            3.0 * bump(y, 164.0, 19.0) * bump(z, 0.0, 45.0) +
            3.0 * bump(y, 145.0, 18.0) * bump(z, 0.0, 43.0) +
            5.0 * bump(y, 111.0, 30.0) * bump(z, 0.0, 45.0))
        jaw = 0.045 * bump(y, 132.0, 45.0) * bump(az, 85.0, 40.0)
        return (strength * face * relief,
                0.0,
                -strength * face * jaw * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    rig = json.loads((options.refined_dir / "42_rig.json").read_text())
    if Path(rig["sourcePath"]).stem != "em1026":
        raise ValueError("selector 42 is not Forest")
    source = options.refined_dir / rig["joints"][1]["obj"]
    metadata = parse_obj(source)
    destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
    triangles = write_mesh(destination, *metadata, 0.1,
                           deformer=make_displacement(metadata[2], options.strength),
                           normal_update=moved_normal)
    print(f"Forest head -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
