"""Shape the clothed zombies' thigh and shin contours at their rigid joints."""

import argparse
import json
from pathlib import Path

from sculpt_human_leg_contours_obj import make_displacement
from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


ZOMBIE_STEMS = {"em1000", "em1011", "em1100", "em1111"}
LEG_JOINTS = (9, 10, 12, 13)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("selectors", nargs="*", default=["4", "21", "j:4", "j:21"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in ZOMBIE_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not a clothed zombie")
        for joint in LEG_JOINTS:
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2],
                                                              options.strength,
                                                              joint in (9, 12)),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
