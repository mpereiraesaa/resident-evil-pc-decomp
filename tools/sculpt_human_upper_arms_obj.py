"""Shape the human deltoid and biceps between rigid shoulder/elbow joints.

Human joints 9 and 12 are upper arms. This follows each arm's existing tilted
centreline and leaves its attachment ends, UVs, and animation untouched.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_human_toes_obj import HUMAN_STEMS
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    height = ymax - ymin
    width = max(p[0] for p in positions) - min(p[0] for p in positions)
    if not 350 < height < 600 or not 110 < width < 340:
        raise ValueError(f"not an expected human upper arm: height {height}, width {width}")

    def lateral_centre(u):
        sample = [p[2] for p in positions if abs((ymax - p[1]) / height - u) < 0.09]
        return (min(sample) + max(sample)) * 0.5

    shoulder_z = lateral_centre(0.14)
    elbow_z = lateral_centre(0.88)
    centre_x = (min(p[0] for p in positions) + max(p[0] for p in positions)) * 0.5

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (ymax - y) / height))
        # A cubic arch peaks one-third along the arm, where the deltoid ends
        # and the biceps belly begins. Both attachment ends stay fixed.
        arch = 6.75 * u * (1.0 - u) * (1.0 - u)
        axis_z = shoulder_z + (elbow_z - shoulder_z) * u
        front = smoothstep((x - centre_x) / (0.55 * width))
        return (0.065 * strength * width * arch * front,
                0.0,
                0.095 * strength * arch * (z - axis_z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="+", help="viewer selectors, e.g. 0 1 36 j:51")
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in HUMAN_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not a known human EMD")
        for joint in (9, 12):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
