"""Shape Kenneth's intact lower face in both corpse variants."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (255 < bounds[0][1] < 270 and 485 < bounds[1][1] < 495 and
            140 < bounds[2][1] < 150):
        raise ValueError(f"not Kenneth's head: {bounds}")

    def displace(position):
        x, y, z = position
        # The devoured mesh has extra damaged-eye triangles. Fade out before
        # that region so both variants retain their authored gore silhouette.
        front = smoothstep((x - 100.0) / 135.0)
        lower = 1.0 - smoothstep((y - 265.0) / 45.0)
        az = abs(z)
        relief = (
            6.0 * bump(y, 242.0, 33.0) * bump(z, 0.0, 28.0) +
            8.0 * bump(y, 208.0, 27.0) * bump(z, 0.0, 30.0) +
            6.0 * bump(y, 209.0, 40.0) * bump(az, 77.0, 34.0) -
            3.5 * bump(y, 155.0, 18.0) * bump(z, 0.0, 42.0) +
            3.0 * bump(y, 138.0, 18.0) * bump(z, 0.0, 41.0) +
            5.0 * bump(y, 104.0, 30.0) * bump(z, 0.0, 44.0))
        jaw = 0.045 * bump(y, 125.0, 45.0) * bump(az, 82.0, 38.0)
        return (strength * front * lower * relief,
                0.0,
                -strength * front * lower * jaw * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector, stem in (("41", "em1025"), ("45", "em1029")):
        rig = json.loads((options.refined_dir / f"{selector}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem != stem:
            raise ValueError(f"selector {selector} is not {stem}")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{stem} head -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
