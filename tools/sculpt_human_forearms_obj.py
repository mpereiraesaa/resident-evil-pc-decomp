"""Shape the separate human forearm joints without moving their attachments.

The character EMDs use rigid joint animation. This changes only the render
mesh between the elbow and wrist rings; the skeleton, UVs and end rings
remain in their original positions.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_toes_obj import HUMAN_STEMS
from sculpt_human_shoulders_obj import parse_obj, write_mesh


def make_displacement(positions, faces, strength):
    low = min(p[1] for p in positions)
    high = max(p[1] for p in positions)
    span = high - low
    width = max(p[0] for p in positions) - min(p[0] for p in positions)
    if not 300 < span < 550 or not 90 < width < 300:
        raise ValueError(f"not an expected forearm: height {span}, width {width}")
    front = max(p[0] for p in positions)
    key = lambda p: tuple(round(value, 2) for value in p)
    groups = {}
    for position in positions:
        groups.setdefault(key(position), []).append(position)
    original = {k: tuple(sum(p[i] for p in values) / len(values)
                         for i in range(3)) for k, values in groups.items()}
    neighbors = {k: set() for k in groups}
    edge_counts = {}
    for face in faces:
        corners = [key(positions[index]) for index in face]
        for a, b in ((corners[0], corners[1]), (corners[1], corners[2]),
                     (corners[2], corners[0])):
            if a == b:
                continue
            neighbors[a].add(b)
            neighbors[b].add(a)
            edge = tuple(sorted((a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = {vertex for edge, count in edge_counts.items() if count == 1
                for vertex in edge}
    # The refined mesh contains planar midpoints. Smooth across coincident UV
    # seams to round its silhouette, but leave open joint ends fixed.
    smoothed = original.copy()
    for _ in range(8):
        next_positions = smoothed.copy()
        for vertex, connected in neighbors.items():
            if vertex in boundary or not connected:
                continue
            average = tuple(sum(smoothed[other][i] for other in connected) /
                            len(connected) for i in range(3))
            next_positions[vertex] = tuple(smoothed[vertex][i] * 0.52 +
                                           average[i] * 0.48 for i in range(3))
        smoothed = next_positions

    def displace(position):
        if key(position) in boundary:
            return (0.0, 0.0, 0.0)
        x, y, z = position
        u = (high - y) / span
        # Hold the connection rings and their nearby vertices in place.
        envelope = math.sin(math.pi * max(0.0, min(1.0, u))) ** 4
        # The exposed muscle belly occupies the front of the forearm; the
        # opposite side receives a smaller lift. Keep both articulated ends.
        front_weight = max(0.0, min(1.0, (x / front + 0.15) / 0.9))
        smooth = smoothed[key(position)]
        dx = strength * envelope * (smooth[0] - x + 4.0 + 14.0 * front_weight)
        dz = strength * envelope * (smooth[2] - z + 0.08 * z)
        return (dx, 0.0, dz)

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
        for joint in (10, 13):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], metadata[-1],
                                                              options.strength),
                                   normal_update=lambda position, normal, displace: normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
