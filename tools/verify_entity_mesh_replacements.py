"""Check every installed .r1mesh against the same-engine refined OBJ export.

Run after `export_entity_objs.py --mesh refined --refinement 3`. This catches
wrong EMD names, TMD offsets, source hashes, element counts, truncated files,
invalid indices, and non-finite vertices before testing in the game viewer.
"""

import argparse
import math
from pathlib import Path
import re
import struct


NAME = re.compile(r"^([A-Za-z0-9]+)_([0-9a-fA-F]{8})\.r1mesh$")
SOURCE = re.compile(r"^# RE1 TMD viewport geometry: ([^ ]+) offset 0x([0-9a-fA-F]+) ")
HEADER = struct.Struct("<8sII")
PART = struct.Struct("<II")
VERTEX = struct.Struct("<11f")


def source_metadata(path):
    mesh_hash = None
    elements = None
    source_parts = {}
    declared = None
    with path.open() as source:
        for line in source:
            if line.startswith("# RE1 TMD viewport geometry: "):
                declared = SOURCE.match(line)
            elif line.startswith("# re1_mesh_hash "):
                mesh_hash = int(line.split()[2], 16)
            elif line.startswith("# re1_element_count "):
                elements = int(line.split()[2])
            elif line.startswith("# re1_element "):
                _, _, index, vertices, primitives, sides = line.split()
                source_parts[int(index)] = (int(vertices),
                                            int(primitives) * (int(sides) - 2))
    if mesh_hash is None or elements is None or len(source_parts) != elements:
        raise ValueError(f"{path}: incomplete source metadata")
    stem, offset = path.name.removesuffix("_refined.obj").rsplit("_", 1)
    if declared is None or Path(declared.group(1)).stem != stem or \
            int(declared.group(2), 16) != int(offset, 16):
        raise ValueError(f"{path}: declared EMD or TMD offset differs from filename")
    return mesh_hash, [source_parts[index] for index in range(elements)]


def replacement_metadata(path, expected_hash, original_parts):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("short header")
    magic, mesh_hash, count = HEADER.unpack_from(data)
    if magic != b"R1MESH01":
        raise ValueError("bad magic")
    if mesh_hash != expected_hash:
        raise ValueError(f"source hash {mesh_hash:08x} != {expected_hash:08x}")
    if count != len(original_parts):
        raise ValueError(f"element count {count} != {len(original_parts)}")
    offset = HEADER.size
    vertices_total = triangles_total = 0
    for index, (_, source_triangles) in enumerate(original_parts):
        if offset + PART.size > len(data):
            raise ValueError(f"element {index}: truncated header")
        vertices, triangles = PART.unpack_from(data, offset)
        offset += PART.size
        if vertices > 65535 or triangles > 131072:
            raise ValueError(f"element {index}: viewport limit exceeded")
        if (vertices == 0) != (triangles == 0):
            raise ValueError(f"element {index}: vertices and triangles disagree")
        if (source_triangles > 0) != (triangles > 0):
            raise ValueError(f"element {index}: source geometry lost or added")
        vertex_end = offset + vertices * VERTEX.size
        index_end = vertex_end + triangles * 3 * 2
        if index_end > len(data):
            raise ValueError(f"element {index}: truncated data")
        for position in range(offset, vertex_end, VERTEX.size):
            if not all(math.isfinite(value) for value in VERTEX.unpack_from(data, position)):
                raise ValueError(f"element {index}: non-finite vertex")
        for (value,) in struct.iter_unpack("<H", data[vertex_end:index_end]):
            if value >= vertices:
                raise ValueError(f"element {index}: index {value} >= {vertices}")
        offset = index_end
        vertices_total += vertices
        triangles_total += triangles
    if offset != len(data):
        raise ValueError(f"trailing {len(data) - offset} bytes")
    return vertices_total, triangles_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--replacements", type=Path,
                        default=Path("tools/data/entity_mesh_replacements"))
    options = parser.parse_args()
    sources = {}
    for path in options.refined_dir.glob("*_refined.obj"):
        key = path.name.removesuffix("_refined.obj")
        if key in sources:
            raise ValueError(f"duplicate source {key}")
        sources[key] = path
    replacements = sorted(options.replacements.glob("*.r1mesh"))
    if not replacements:
        raise ValueError("no replacement meshes found")
    failed = 0
    for path in replacements:
        try:
            match = NAME.fullmatch(path.name)
            if match is None:
                raise ValueError("invalid replacement name")
            source = sources.get(path.stem)
            if source is None:
                raise ValueError("matching refined OBJ not found")
            mesh_hash, parts = source_metadata(source)
            vertices, triangles = replacement_metadata(path, mesh_hash, parts)
            print(f"OK {path.name}: {vertices} vertices, {triangles} triangles")
        except (OSError, ValueError, struct.error) as exc:
            print(f"FAIL {path.name}: {exc}")
            failed += 1
    print(f"Validated {len(replacements) - failed}/{len(replacements)} replacements")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
