"""Round the central carapace of the four giant-spider EMDs.

Joint 0 carries the cephalothorax. Keep the rear abdomen seam and the front
mouth end fixed, and concentrate the raised shell between the leg sockets.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


SPIDER_STEMS = {"em1003", "em1004", "em1103", "em1104"}


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (0 < bounds[0][0] < 20 and 1030 < bounds[0][1] < 1060 and
            330 < bounds[1][1] < 355 and 465 < bounds[2][1] < 490):
        raise ValueError(f"not the expected spider carapace: {bounds}")

    def displace(position):
        x, y, z = position
        length = smoothstep((x - 85.0) / 125.0)
        length *= 1.0 - smoothstep((x - 690.0) / 155.0)
        crown = smoothstep((y - 35.0) / 190.0)
        middle = 1.0 - smoothstep((abs(z) - 115.0) / 245.0)
        lift = 55.0 * strength * length * crown * middle
        # A small outward arch at the flanks preserves the leg-root centres.
        flank = bump(abs(z), 235.0, 120.0) * bump(y, 145.0, 125.0)
        widen = 18.0 * strength * length * flank
        return (0.0, lift, math.copysign(widen, z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["7", "8", "j:7", "j:8"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in SPIDER_STEMS or len(rig["joints"]) != 20:
            raise ValueError(f"selector {selector} is not a giant spider")
        source = options.refined_dir / rig["joints"][0]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} carapace -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
