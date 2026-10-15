"""Trial anatomical chest relief for the Hunter torso joint."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (320 < bounds[0][1] < 345 and 920 < bounds[1][1] < 950 and
            460 < bounds[2][1] < 480):
        raise ValueError(f"not the expected Hunter torso: {bounds}")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 35.0) / 170.0)
        centre = 1.0 - smoothstep((abs(z) - 300.0) / 125.0)
        vertical = smoothstep((y - 120.0) / 90.0)
        vertical *= 1.0 - smoothstep((y - 735.0) / 125.0)
        az = abs(z)
        pec = 53.0 * bump(y, 585.0, 105.0) * bump(az, 205.0, 125.0)
        sternum = -18.0 * bump(y, 565.0, 135.0) * bump(z, 0.0, 48.0)
        ribs = 26.0 * bump(y, 385.0, 105.0) * bump(az, 270.0, 100.0)
        abdomen = 18.0 * bump(y, 275.0, 80.0) * bump(az, 105.0, 80.0)
        return (strength * front * centre * vertical *
                (pec + sternum + ribs + abdomen), 0.0, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["10", "j:10"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1006", "em1106"}:
            raise ValueError(f"selector {selector} is not Hunter")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} torso -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
