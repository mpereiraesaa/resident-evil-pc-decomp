"""Sculpt an RE1 human torso in viewport coordinates, without Blender.

The OBJ's Z axis is shoulder width and Y is height, as confirmed by the
left/right arm joint translations. Blender imports it as (X, -Z, Y).
This tool reads joint 0 from a verified refined rig and writes one render-only
.r1mesh with four triangles per input triangle. Rig joints and UVs are kept.
Candidates should be inspected in the game viewer before installation.
"""

import argparse
import json
import math
from pathlib import Path
import struct


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def bump(value, centre, width):
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def normalise(values):
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values) if length > 1e-12 else tuple(values)


def parse_obj(path):
    positions, uvs, normals, colours, faces = [], [], [], [], []
    source_hash = None
    element_count = None
    source_faces = {}
    for line in path.read_text().splitlines():
        if line.startswith("# re1_mesh_hash "):
            source_hash = int(line.split()[2], 16)
        elif line.startswith("# re1_element_count "):
            element_count = int(line.split()[2])
        elif line.startswith("# re1_element "):
            _, _, index, _, lists, sides = line.split()
            source_faces[int(index)] = int(lists) * (int(sides) - 2)
        elif line.startswith("# re1_color "):
            colours.append(tuple(map(float, line.split()[2:5])))
        elif line.startswith("v "):
            positions.append(tuple(map(float, line.split()[1:4])))
        elif line.startswith("vt "):
            uvs.append(tuple(map(float, line.split()[1:3])))
        elif line.startswith("vn "):
            normals.append(tuple(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            corners = [tuple(int(value) - 1 for value in word.split("/"))
                       for word in line.split()[1:]]
            if len(corners) != 3 or any(a != b or b != c for a, b, c in corners):
                raise ValueError(f"{path}: expected aligned triangle indices")
            faces.append(tuple(corner[0] for corner in corners))
    if source_hash is None or element_count is None:
        raise ValueError(f"{path}: missing source metadata")
    if len(positions) != len(uvs) or len(positions) != len(normals):
        raise ValueError(f"{path}: position/UV/normal counts differ")
    if colours and len(colours) != len(positions):
        raise ValueError(f"{path}: colour count differs")
    if sum(source_faces.values()) != len(faces) or len(source_faces) != element_count:
        raise ValueError(f"{path}: face metadata differs")
    if not colours:
        colours = [(1.0, 1.0, 1.0)] * len(positions)
    return source_hash, source_faces, positions, uvs, normals, colours, faces


def make_displacement(positions, strength):
    width = max(abs(position[2]) for position in positions)
    low = min(position[1] for position in positions)
    span = max(position[1] for position in positions) - low
    if width < 1.0 or span < 1.0:
        raise ValueError("torso has degenerate bounds")

    def displace(position):
        _, y, z = position
        side = smoothstep((abs(z) - 0.55 * width) / (0.45 * width))
        shoulder = bump(y, low + 0.82 * span, 0.17 * span)
        ridge = bump(y, low + 0.88 * span, 0.13 * span)
        dz = -math.copysign(strength * width * side * shoulder, z)
        dy = -0.025 * span * side * ridge
        return (0.0, dy, dz)

    return displace


def moved_normal(position, normal, displace):
    # F(x,y,z)=(x, y+dy(y,z), z+dz(y,z)). Solve J^T n' = n.
    step = 0.25
    dyp = displace((position[0], position[1] + step, position[2]))
    dym = displace((position[0], position[1] - step, position[2]))
    dzp = displace((position[0], position[1], position[2] + step))
    dzm = displace((position[0], position[1], position[2] - step))
    a = 1.0 + (dyp[1] - dym[1]) / (2.0 * step)
    b = (dzp[1] - dzm[1]) / (2.0 * step)
    c = (dyp[2] - dym[2]) / (2.0 * step)
    d = 1.0 + (dzp[2] - dzm[2]) / (2.0 * step)
    determinant = a * d - b * c
    if determinant < 1e-4:
        raise ValueError("shoulder deformation folds the mesh")
    return normalise((normal[0],
                      (d * normal[1] - c * normal[2]) / determinant,
                      (-b * normal[1] + a * normal[2]) / determinant))


def midpoint(a, b):
    return tuple((x + y) * 0.5 for x, y in zip(a, b))


def write_mesh(path, source_hash, source_faces, positions, uvs, normals,
               colours, faces, strength, deformer=None, normal_update=None):
    displace = deformer or make_displacement(positions, strength)
    normal_update = normal_update or moved_normal
    vertices = []
    for position, uv, normal, colour in zip(positions, uvs, normals, colours):
        offset = displace(position)
        moved = tuple(position[i] + offset[i] for i in range(3))
        normal = normal_update(position, normal, displace)
        vertices.append((*moved, *normal, *colour, uv[0], 1.0 - uv[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<8sII", b"R1MESH01", source_hash,
                                 len(source_faces)))
        cursor = 0
        for element in range(len(source_faces)):
            count = source_faces[element]
            part_vertices = []
            part_indices = []
            for face in faces[cursor:cursor + count]:
                a, b, c = (vertices[index] for index in face)
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                # Interpolated midpoint normals must remain unit length.
                triangle = [a, b, c, ab, bc, ca]
                triangle = [(*v[:3], *normalise(v[3:6]), *v[6:])
                            for v in triangle]
                base = len(part_vertices)
                part_vertices.extend(triangle)
                for child in ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)):
                    part_indices.extend(base + index for index in child)
            cursor += count
            if len(part_vertices) > 65535:
                raise ValueError(f"element {element}: exceeds 16-bit vertex indices")
            output.write(struct.pack("<II", len(part_vertices), len(part_indices) // 3))
            for vertex in part_vertices:
                if not all(math.isfinite(value) for value in vertex):
                    raise ValueError("non-finite vertex")
                output.write(struct.pack("<11f", *vertex))
            output.write(struct.pack(f"<{len(part_indices)}H", *part_indices))
    return sum(source_faces.values()) * 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.10)
    parser.add_argument("selectors", nargs="+")
    options = parser.parse_args()
    if not 0.0 < options.strength <= 0.2:
        raise ValueError("strength must be in (0, 0.2]")
    for selector in options.selectors:
        rig_name = selector.replace(":", "_")
        rig = json.loads((options.refined_dir / f"{rig_name}_rig.json").read_text())
        source = options.refined_dir / rig["joints"][0]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, options.strength)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
