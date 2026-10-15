"""Round the forearm belly in Chris's separate close-up hand rigs.

Joint 0 is the forearm. Both its ends remain fixed so the elbow and the
animated hand at joint 1 retain their original attachment rings.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    xmax = max(p[0] for p in positions)
    if not 430 <= xmax <= 535:
        raise ValueError(f"not the expected Chris forearm: max X {xmax}")

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, x / xmax))
        envelope = math.sin(math.pi * u) ** 2
        return (0.0, 0.055 * strength * envelope * y,
                0.085 * strength * envelope * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["24", "25", "j:24", "j:25"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1014", "em1015", "em1114", "em1115"}:
            raise ValueError(f"selector {selector} is not a Chris close-up arm")
        source = options.refined_dir / rig["joints"][0]["obj"]
        metadata = parse_obj(source)
        displacement = make_displacement(metadata[2], options.strength)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=displacement, normal_update=moved_normal)
        print(f"{selector} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
