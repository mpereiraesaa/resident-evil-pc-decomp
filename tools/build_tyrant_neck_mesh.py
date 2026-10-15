"""Trial: add a short head-rigged neck surface under Tyrant's jaw.

The bottom sits inside the torso collar and the top inside the existing jaw.
The original head replacement is kept intact; a textured neck surface is
added to its single TMD element. Review motion and the UV sample in the game
viewer before installing this experimental candidate.
"""

import argparse
import json
import math
from pathlib import Path
import struct


TYRANT_STEMS = {"em100c", "em1010", "em110c", "em1110"}
VERTEX = struct.Struct("<11f")


def add_neck(data, placement="upper", lift_head=0.0):
    magic, source_hash, elements = struct.unpack_from("<8sII", data)
    if magic != b"R1MESH01" or elements != 1:
        raise ValueError("expected single-element Tyrant head replacement")
    count, triangles = struct.unpack_from("<II", data, 16)
    vertex_start = 24
    index_start = vertex_start + count * VERTEX.size
    index_size = triangles * 3 * 2
    if len(data) != index_start + index_size:
        raise ValueError("head replacement size differs from its header")
    vertices = bytearray(data[vertex_start:index_start])
    if lift_head:
        for i in range(count):
            offset = i * VERTEX.size
            values = list(VERTEX.unpack_from(vertices, offset))
            values[1] += lift_head
            VERTEX.pack_into(vertices, offset, *values)
    indices = list(struct.unpack_from(f"<{triangles * 3}H", data, index_start))

    # Local head axis: +X forward, +Y upward, Z across the face. The lower
    # rings are fully buried in the torso; the upper rings overlap the jaw.
    if placement == "slim":
        rings = [(55.0, 122.0, 53.0, 66.0),
                 (105.0, 130.0, 62.0, 75.0),
                 (155.0, 139.0, 66.0, 81.0),
                 (205.0, 146.0, 67.0, 84.0),
                 (255.0, 152.0, 66.0, 86.0),
                 (305.0, 154.0, 64.0, 86.0)]
    elif placement == "tapered":
        # Broad sternocleidomastoid roots converge below the jaw. Keep the
        # forward contour curved to avoid the flat slab of the slim tube.
        rings = [(45.0, 110.0, 65.0, 105.0),
                 (95.0, 125.0, 65.0, 96.0),
                 (145.0, 139.0, 64.0, 86.0),
                 (195.0, 150.0, 60.0, 76.0),
                 (245.0, 160.0, 54.0, 66.0),
                 (300.0, 170.0, 48.0, 58.0)]
    elif placement == "bridge":
        # Reach under the original flat jaw rim, then settle back into the
        # chest. The slim surface stayed behind that rim and read as a block.
        rings = [(45.0, 147.0, 78.0, 92.0),
                 (95.0, 152.0, 84.0, 92.0),
                 (145.0, 155.0, 82.0, 88.0),
                 (195.0, 157.0, 73.0, 81.0),
                 (245.0, 160.0, 66.0, 75.0),
                 (305.0, 162.0, 61.0, 70.0)]
    elif placement == "forward":
        # Low local Y is below the jaw. Move the neck's front contour toward
        # the facial silhouette while leaving the root behind the collar.
        rings = [(55.0, 185.0, 105.0, 98.0),
                 (100.0, 197.0, 106.0, 105.0),
                 (145.0, 207.0, 106.0, 109.0),
                 (195.0, 211.0, 102.0, 109.0),
                 (245.0, 210.0, 96.0, 106.0),
                 (295.0, 205.0, 87.0, 99.0)]
    elif placement == "upper":
        rings = [(70.0, 145.0, 85.0, 94.0),
                 (110.0, 151.0, 98.0, 105.0),
                 (155.0, 159.0, 106.0, 109.0),
                 (205.0, 168.0, 112.0, 109.0),
                 (255.0, 175.0, 111.0, 106.0),
                 (300.0, 178.0, 103.0, 99.0)]
    else:
        raise ValueError(placement)
    sides = 16
    for y, centre_x, radius_x, radius_z in rings:
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides
            cosine, sine = math.cos(angle), math.sin(angle)
            x = centre_x + radius_x * cosine
            z = radius_z * sine
            nx, nz = cosine / radius_x, sine / radius_z
            norm = math.hypot(nx, nz)
            nx, nz = nx / norm, nz / norm
            # A narrow grey-skin patch from the lower cheek area of the
            # original head atlas; V is in the runtime orientation.
            u = 0.225 + 0.018 * sine
            v = 1.0 - (0.687 + 0.030 * (y - rings[0][0]) /
                       (rings[-1][0] - rings[0][0]))
            tone = 1.0
            if placement == "bridge":
                # Modulate the otherwise uniform cheek-UV sample to suggest
                # two neck muscles and a darker insertion below the jaw.
                root = (y - rings[0][0]) / (rings[-1][0] - rings[0][0])
                tendon = math.exp(-0.5 * ((abs(sine) - 0.53) / 0.15) ** 2)
                tone = 0.79 + 0.14 * tendon + 0.07 * root
            vertices += VERTEX.pack(x, y, z, nx, 0.0, nz,
                                    tone, tone * 0.98, tone * 0.99, u, v)
    for ring in range(len(rings) - 1):
        for i in range(sides):
            a = count + ring * sides + i
            b = count + ring * sides + (i + 1) % sides
            c = count + (ring + 1) * sides + i
            d = count + (ring + 1) * sides + (i + 1) % sides
            indices.extend((a, c, b, c, d, b))

    output = bytearray(struct.pack("<8sII", magic, source_hash, elements))
    output += struct.pack("<II", count + len(rings) * sides, len(indices) // 3)
    output += vertices
    output += struct.pack(f"<{len(indices)}H", *indices)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--installed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--placement", choices=("upper", "forward", "slim", "tapered", "bridge"), default="upper")
    parser.add_argument("--lift-head", type=float, default=0.0)
    parser.add_argument("selectors", nargs="*", default=["16", "20", "j:16", "j:20"])
    options = parser.parse_args()
    if not 0.0 <= options.lift_head <= 120.0:
        raise ValueError("head lift must be in [0, 120]")
    options.out.mkdir(parents=True, exist_ok=True)
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in TYRANT_STEMS:
            raise ValueError(f"selector {selector} is not Tyrant")
        source = rig["joints"][2]["obj"].replace("_refined.obj", ".r1mesh")
        data = add_neck((options.installed_dir / source).read_bytes(),
                        options.placement, options.lift_head)
        destination = options.out / source
        destination.write_bytes(data)
        print(selector, destination, len(data))


if __name__ == "__main__":
    main()
