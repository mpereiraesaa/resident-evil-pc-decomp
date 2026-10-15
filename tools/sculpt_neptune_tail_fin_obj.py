"""Give Neptune's rigid tail fork a longer swept upper lobe."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    xmin = min(p[0] for p in positions)
    ymax = max(p[1] for p in positions)
    ymin = min(p[1] for p in positions)
    if not -640 < xmin < -610 or not 820 < ymax < 840 or not -810 < ymin < -780:
        raise ValueError("not the expected Neptune tail fin")

    def displace(position):
        x, y, z = position
        fork = smoothstep((-x - 160.0) / 430.0)
        upper = smoothstep((y - 220.0) / (ymax - 220.0))
        lower = smoothstep((-y - 220.0) / (-ymin - 220.0))
        weight = fork * (upper * upper + lower * lower)
        return (-28.0 * strength * weight,
                strength * fork * (70.0 * upper * upper - 23.0 * lower * lower),
                -0.10 * strength * weight * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["15", "j:15"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100b", "em110b"}:
            raise ValueError(f"selector {selector} is not Neptune")
        source = options.refined_dir / rig["joints"][12]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} joint 12 -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
