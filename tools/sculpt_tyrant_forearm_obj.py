"""Round Tyrant's normal forearm while preserving elbow and wrist sockets."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def make_displacement(positions, strength):
    bounds = [(min(p[i] for p in positions), max(p[i] for p in positions))
              for i in range(3)]
    if not (-185 < bounds[0][0] < -170 and -705 < bounds[1][0] < -685 and
            50 < bounds[1][1] < 80):
        raise ValueError(f"not the expected Tyrant forearm: {bounds}")

    def displace(position):
        x, y, z = position
        shaft = smoothstep((-y - 70.0) / 120.0)
        shaft *= 1.0 - smoothstep((-y - 545.0) / 115.0)
        # Keep both rigid-joint rings fixed. Narrow the boxy wrist-side shaft
        # and give the upper forearm a mild muscle swell.
        profile = (-0.16 * bump(y, -440.0, 145.0) +
                   0.045 * bump(y, -215.0, 105.0))
        scale = strength * shaft * profile
        return (x * scale, 0.0, z * scale)

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
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS:
            raise ValueError(f"selector {selector} is not Tyrant")
        source = options.refined_dir / rig["joints"][4]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} forearm -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
