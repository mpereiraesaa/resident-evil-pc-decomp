"""Lower the central Tyrant collar to reveal the head-rigged neck.

The side shoulder boundary stays fixed. This is a trial mesh; judge it in the
same-engine viewer before installing it.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import normalise, parse_obj, smoothstep, write_mesh


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}


def collar_displacement(position, depth):
    x, y, z = position
    height = smoothstep((y - 1080.0) / 230.0)
    centre = 1.0 - smoothstep((abs(z) - 145.0) / 230.0)
    front = smoothstep((x + 350.0) / 180.0)
    return 0.0, -depth * height * centre * front, 0.0


def collar_normal(position, normal, displace):
    step = 0.25
    def derivative(axis):
        plus = list(position)
        minus = list(position)
        plus[axis] += step
        minus[axis] -= step
        return (displace(plus)[1] - displace(minus)[1]) / (2.0 * step)
    dx, dy, dz = (derivative(axis) for axis in range(3))
    adjusted_y = normal[1] / (1.0 + dy)
    return normalise((normal[0] - dx * adjusted_y,
                      adjusted_y,
                      normal[2] - dz * adjusted_y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=float, default=120.0)
    parser.add_argument("selectors", nargs="*", default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 < options.depth <= 160.0:
        raise ValueError("depth must be in (0, 160]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS:
            raise ValueError(f"selector {selector} is not Tyrant")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=lambda p: collar_displacement(p, options.depth),
                               normal_update=collar_normal)
        print(f"{selector} collar -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
