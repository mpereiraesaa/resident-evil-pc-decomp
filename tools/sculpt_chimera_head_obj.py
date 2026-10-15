"""Give the Chimera's separate insect head more brow and mandible volume.

Only joint 3 changes. Source UVs, face texture and all animation transforms
remain attached to their original rigid joint.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    if max(p[0] for p in positions) < 300 or max(p[1] for p in positions) < 590:
        raise ValueError("not the expected Chimera head")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 20.0) / 195.0)
        eyes = bump(z, -117.0, 60.0) + bump(z, 117.0, 60.0)
        mandibles = bump(z, -137.0, 64.0) + bump(z, 137.0, 64.0)
        dx = front * strength * (
            17.0 * bump(y, 425.0, 68.0) * eyes +
            16.0 * bump(y, 185.0, 78.0) * mandibles +
            9.0 * bump(y, 290.0, 85.0) * bump(z, 0.0, 54.0))
        return (dx, 0.0, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["13", "j:13"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 2.0:
        raise ValueError("strength must be in (0, 2]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1009", "em1109"}:
            raise ValueError(f"selector {selector} is not Chimera")
        source = options.refined_dir / rig["joints"][3]["obj"]
        metadata = parse_obj(source)
        displacement = make_displacement(metadata[2], options.strength)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=displacement, normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
