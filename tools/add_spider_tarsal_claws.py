"""Prototype paired tarsal claws on the eight distal spider leg meshes."""

import argparse
import json
import math
from pathlib import Path
import struct

from sculpt_human_shoulders_obj import parse_obj


VERTEX = struct.Struct("<11f")
DISTAL_JOINTS = (4, 6, 8, 10, 12, 14, 16, 18)


def add_claws(data, source, joint):
    magic, source_hash, elements = struct.unpack_from("<8sII", data)
    if magic != b"R1MESH01" or elements != 1:
        raise ValueError("expected one spider-leg mesh element")
    count, triangles = struct.unpack_from("<II", data, 16)
    if count != 4416:
        raise ValueError("expected the baseline tapered leg; refusing to append claws twice")
    vertex_start = 24
    index_start = vertex_start + count * VERTEX.size
    if len(data) != index_start + triangles * 6:
        raise ValueError("source replacement size differs from header")
    vertices = bytearray(data[vertex_start:index_start])
    indices = list(struct.unpack_from(f"<{triangles * 3}H", data, index_start))

    _, _, positions, uvs, _, _, _ = parse_obj(source)
    sign = -1 if joint < 11 else 1
    length = max(sign * p[2] for p in positions)
    tip_ids = [i for i, p in enumerate(positions) if sign * p[2] > 0.99 * length]
    cx = sum(positions[i][0] for i in tip_ids) / len(tip_ids)
    cy = sum(positions[i][1] for i in tip_ids) / len(tip_ids)
    nearest = min(tip_ids, key=lambda i: abs(positions[i][0] - cx) +
                  abs(positions[i][1] - cy))
    u, source_v = uvs[nearest]
    v = 1.0 - source_v

    sides = 8
    rings = ((-21.0, 12.0, 0.0), (18.0, 9.0, -2.0),
             (52.0, 5.0, -11.0), (81.0, 0.7, -25.0))
    for offset in (-18.0, 18.0):
        start = len(vertices) // VERTEX.size
        for along, radius, drop in rings:
            for k in range(sides):
                angle = 2.0 * math.pi * k / sides
                nx, ny = math.cos(angle), math.sin(angle)
                vertices += VERTEX.pack(cx + offset + radius * nx,
                                        cy + drop + radius * ny,
                                        sign * (length + along),
                                        nx * 0.92, ny * 0.92, sign * 0.39,
                                        1.0, 1.0, 1.0, u, v)
        for ring in range(len(rings) - 1):
            for k in range(sides):
                a = start + ring * sides + k
                b = start + ring * sides + (k + 1) % sides
                c = start + (ring + 1) * sides + k
                d = start + (ring + 1) * sides + (k + 1) % sides
                if sign > 0:
                    indices.extend((a, b, c, b, d, c))
                else:
                    indices.extend((a, c, b, b, c, d))

    new_count = len(vertices) // VERTEX.size
    if new_count > 65535:
        raise ValueError("spider leg exceeds 16-bit index limit")
    result = bytearray(struct.pack("<8sII", magic, source_hash, elements))
    result += struct.pack("<II", new_count, len(indices) // 3)
    result += vertices
    result += struct.pack(f"<{len(indices)}H", *indices)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--installed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("selectors", nargs="*", default=["7", "8", "j:7", "j:8"])
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1003", "em1004", "em1103", "em1104"}:
            raise ValueError(f"selector {selector} is not a spider")
        for joint in DISTAL_JOINTS:
            source = options.refined_dir / rig["joints"][joint]["obj"]
            name = source.name.replace("_refined.obj", ".r1mesh")
            data = add_claws((options.installed_dir / name).read_bytes(), source, joint)
            destination = options.out / name
            destination.write_bytes(data)
            print(selector, joint, destination)


if __name__ == "__main__":
    main()
