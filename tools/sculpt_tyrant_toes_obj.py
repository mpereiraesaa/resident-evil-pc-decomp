"""Give Tyrant's five large separate toe tips a longer tapered shape."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def make_displacement(positions, strength):
    xmax = max(p[0] for p in positions)
    ymin = min(p[1] for p in positions)
    zmin = min(p[2] for p in positions)
    zmax = max(p[2] for p in positions)
    if not 540 < xmax < 560 or not -330 < ymin < -305 or not 330 < zmax - zmin < 365:
        raise ValueError("not the expected Tyrant foot")
    distal = [p for p in positions if p[0] > 450.0]
    low = min(p[2] for p in distal)
    high = max(p[2] for p in distal)
    centers = [low + (high - low) * (i + 0.5) / 5.0 for i in range(5)]

    def displace(position):
        x, y, z = position
        tip = smoothstep((x - 330.0) / (xmax - 330.0)) ** 2
        nearest = min(range(5), key=lambda index: abs(z - centers[index]))
        length = (0.65, 0.85, 1.0, 0.85, 0.65)[nearest]
        return (42.0 * strength * tip * length,
                0.0,
                -0.15 * strength * tip * (z - centers[nearest]))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not Tyrant")
        for joint in (11, 14):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
