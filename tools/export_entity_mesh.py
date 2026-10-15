"""Export arbitrary per-joint Blender geometry for RE1's TMD viewport.

Run on a .blend assembled by assemble_entity_obj.py:
  blender --background edited.blend --python tools/export_entity_mesh.py -- \
      --out tools/data/entity_mesh_replacements --only-joint 2

The source TMD element still selects the texture/material. Each replacement
element may have new triangles and vertices, but uses the same UV atlas and
original rigid joint. The 16-bit viewport index format limits each element to
65,535 vertices. Blender modifiers are evaluated before export.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-joint", type=int)
    parser.add_argument("--preserve-source-normals", action="store_true")
    return parser.parse_args(argv)


def export_object(obj, output, depsgraph, preserve_normals):
    original_counts = json.loads(obj["re1_element_counts"])
    count = len(original_counts)
    bind_values = obj["re1_bind_matrix"]
    bind = Matrix([list(bind_values[i:i + 4]) for i in range(0, 16, 4)])
    relative = bind.inverted() @ obj.matrix_world
    normal_matrix = relative.to_3x3().inverted().transposed()

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        elements = mesh.attributes.get("re1_element")
        colors = mesh.attributes.get("re1_color")
        normals = mesh.attributes.get("re1_export_normal")
        uv_layer = mesh.uv_layers.active
        if (elements is None or elements.domain != "FACE" or
                colors is None or colors.domain != "POINT" or
                uv_layer is None):
            raise RuntimeError(f"{obj.name}: missing RE1 face, colour or UV data")
        if preserve_normals and (normals is None or
                                 len(normals.data) != len(mesh.vertices)):
            raise RuntimeError(f"{obj.name}: source normals unavailable")

        mesh.calc_loop_triangles()
        parts = [dict(vertices=[], indices=[], lookup={}) for _ in range(count)]
        for triangle in mesh.loop_triangles:
            element = elements.data[triangle.polygon_index].value
            if element < 0 or element >= count:
                raise RuntimeError(f"{obj.name}: invalid element {element}")
            part = parts[element]
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                index = loop.vertex_index
                uv = uv_layer.data[loop_index].uv
                # A vertex may straddle an atlas seam. Duplicate only those
                # UV corners; the game stores UVs in its vertex buffer.
                key = (index, struct.pack("<2f", uv[0], uv[1]))
                if key not in part["lookup"]:
                    vertex = mesh.vertices[index]
                    position = relative @ vertex.co
                    normal = (Vector(normals.data[index].vector)
                              if preserve_normals else Vector(vertex.normal))
                    normal = normal_matrix @ normal
                    if normal.length > 1e-8:
                        normal.normalize()
                    color = colors.data[index].color
                    values = (*position, *normal,
                              color[0], color[1], color[2],
                              uv[0], 1.0 - uv[1])
                    if not all(math.isfinite(value) for value in values):
                        raise RuntimeError(f"{obj.name}: non-finite vertex")
                    if len(part["vertices"]) >= 65535:
                        raise RuntimeError(f"{obj.name}: element {element} exceeds 16-bit indices")
                    part["lookup"][key] = len(part["vertices"])
                    part["vertices"].append(values)
                part["indices"].append(part["lookup"][key])

        for element, part in enumerate(parts):
            if (original_counts[element] > 0) != bool(part["indices"]):
                raise RuntimeError(f"{obj.name}: element {element} lost its geometry")
            if len(part["indices"]) // 3 > 131072:
                raise RuntimeError(f"{obj.name}: element {element} has too many triangles")

        source = Path(obj["re1_source_obj"])
        stem = source.stem.rsplit("_", 1)[0]
        path = output / f"{stem}.r1mesh"
        with path.open("wb") as target:
            target.write(struct.pack("<8sII", b"R1MESH01",
                                     int(obj["re1_mesh_hash"], 16), count))
            for part in parts:
                target.write(struct.pack("<II", len(part["vertices"]),
                                         len(part["indices"]) // 3))
                for vertex in part["vertices"]:
                    target.write(struct.pack("<11f", *vertex))
                if part["indices"]:
                    target.write(struct.pack(f"<{len(part['indices'])}H",
                                             *part["indices"]))
        return path, sum(len(part["vertices"]) for part in parts), len(mesh.loop_triangles)
    finally:
        evaluated.to_mesh_clear()


def main():
    options = args()
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    written = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "re1_mesh_hash" not in obj:
            continue
        if options.only_joint is not None and obj["re1_joint"] != options.only_joint:
            continue
        path, vertices, triangles = export_object(
            obj, output, depsgraph, options.preserve_source_normals)
        written += 1
        print("RE1_MESH", obj["re1_joint"], vertices, triangles, path)
    if written == 0:
        raise RuntimeError("no RE1 meshes found in the Blender scene")
    print("RE1_MESH_WRITTEN", written)


if __name__ == "__main__":
    main()
