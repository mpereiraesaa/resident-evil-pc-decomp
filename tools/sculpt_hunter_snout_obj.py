"""Shape the Hunter's paired upper muzzle and animated lower jaw."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, offset_x, strength):
    front = max(p[0] + offset_x for p in positions)
    if not 385 <= front <= 410:
        raise ValueError(f"not the expected Hunter muzzle: front X {front}")

    def displace(position):
        x, _, z = position
        tip = smoothstep((x + offset_x - 145.0) / 255.0)
        tip *= tip
        return (46.0 * strength * tip,
                0.0,
                -0.14 * strength * tip * z)

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
            raise ValueError(f"selector {selector} is not a Hunter")
        if rig["joints"][3]["restTranslation"][0] != 189:
            raise ValueError("Hunter jaw rest offset changed")
        for joint in (2, 3):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            displacement = make_displacement(metadata[2], 189 if joint == 3 else 0,
                                             options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
