"""Subtle Tyrant head anatomy sculpt for the four identical PC EMD variants.

The head is rig joint 2. Its front is positive viewport X, its height is Y,
and the left/right axis is Z. Source UVs, texture pages and rigid animation
remain unchanged. Review the candidate in the same-engine viewer before use.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, normalise, write_mesh


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def moved_normal(position, normal, displace):
    step = 0.25
    columns = []
    for axis in range(3):
        plus = list(position)
        minus = list(position)
        plus[axis] += step
        minus[axis] -= step
        dp = displace(plus)
        dm = displace(minus)
        column = [(dp[i] - dm[i]) / (2.0 * step) for i in range(3)]
        column[axis] += 1.0
        columns.append(column)
    a, b, c = columns
    cofactor = (cross(b, c), cross(c, a), cross(a, b))
    determinant = sum(a[i] * cofactor[0][i] for i in range(3))
    if determinant < 1e-4:
        raise ValueError("face deformation folds the mesh")
    moved = tuple(sum(normal[j] * cofactor[j][i] for j in range(3)) /
                  determinant for i in range(3))
    return normalise(moved)


def make_face_displacement(positions, strength):
    if max(p[0] for p in positions) < 300 or max(p[1] for p in positions) < 700:
        raise ValueError("not the expected Tyrant head bounds")

    def displace(position):
        x, y, z = position
        front = smoothstep((x - 85.0) / 180.0)
        left_brow = bump(z, -75.0, 44.0)
        right_brow = bump(z, 75.0, 44.0)
        cheeks = bump(z, -110.0, 50.0) + bump(z, 110.0, 50.0)
        ridge = bump(z, 0.0, 34.0)
        dx = front * strength * (
            15.0 * bump(y, 505.0, 56.0) * (left_brow + right_brow) +
            12.0 * bump(y, 382.0, 66.0) * cheeks +
            11.0 * bump(y, 417.0, 77.0) * ridge +
            13.0 * bump(y, 213.0, 54.0) * bump(z, 0.0, 67.0))
        return (dx, 0.0, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*",
                        default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 2.0:
        raise ValueError("strength must be in (0, 2]")
    for selector in options.selectors:
        rig_name = selector.replace(":", "_")
        rig = json.loads((options.refined_dir / f"{rig_name}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100c", "em1010", "em110c", "em1110"}:
            raise ValueError(f"selector {selector} is not a Tyrant EMD")
        source = options.refined_dir / rig["joints"][2]["obj"]
        metadata = parse_obj(source)
        displacement = make_face_displacement(metadata[2], options.strength)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=displacement, normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
