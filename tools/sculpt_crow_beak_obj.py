"""Round and slightly hook the crow's separate beak joint (joint 2)."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    if max(p[0] for p in positions) < 115 or max(abs(p[2]) for p in positions) > 35:
        raise ValueError("not the expected crow beak")

    def displace(position):
        x, _, z = position
        tip = smoothstep((x - 24.0) / 94.0)
        tip *= tip
        return (11.0 * strength * tip,
                -17.0 * strength * tip,
                -0.18 * strength * tip * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["9", "j:9"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1005", "em1105"}:
            raise ValueError(f"selector {selector} is not a crow")
        source = options.refined_dir / rig["joints"][2]["obj"]
        metadata = parse_obj(source)
        displacement = make_displacement(metadata[2], options.strength)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=displacement, normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
