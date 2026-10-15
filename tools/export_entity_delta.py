"""Write position/normal patches from an assembled RE1 Blender scene.

Run after editing the saved .blend in Blender:
  blender --background edited.blend --python tools/export_entity_delta.py -- \
      --out mods/entity-mesh/mesh

The patch keeps the game's UVs, colours, triangle order, and joint animation.
Vertex count and order must stay intact. Sculpting and object transforms are
supported; topology-changing modifiers must be applied through a future mesh
replacement path instead.
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
    parser.add_argument("--recompute-normals", action="store_true")
    return parser.parse_args(argv)


def main():
    options = args()
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "re1_mesh_hash" not in obj:
            continue
        joint = int(obj["re1_joint"])
        if options.only_joint is not None and joint != options.only_joint:
            continue
        counts = json.loads(obj["re1_element_counts"])
        if sum(counts) != len(obj.data.vertices):
            raise RuntimeError(f"{obj.name}: topology changed; expected {sum(counts)} vertices")
        source = Path(obj["re1_source_obj"])
        stem = source.stem.rsplit("_", 1)[0]
        path = output / f"{stem}.r1delta"
        hash_value = int(obj["re1_mesh_hash"], 16)
        bind = Matrix([list(obj["re1_bind_matrix"][i:i + 4])
                       for i in range(0, 16, 4)])
        relative = bind.inverted() @ obj.matrix_world
        normal_matrix = relative.to_3x3().inverted().transposed()
        source_normals = obj.data.attributes.get("re1_export_normal")
        if source_normals is None and not options.recompute_normals:
            raise RuntimeError(f"{obj.name}: missing exported normal attribute")

        with path.open("wb") as patch:
            patch.write(struct.pack("<8sII", b"R1DELTA1", hash_value, len(counts)))
            patch.write(struct.pack(f"<{len(counts)}I", *counts))
            for index, vertex in enumerate(obj.data.vertices):
                position = relative @ vertex.co
                if options.recompute_normals:
                    normal = Vector(vertex.normal)
                else:
                    normal = Vector(source_normals.data[index].vector)
                normal = normal_matrix @ normal
                if normal.length > 1e-8:
                    normal.normalize()
                values = (*position, *normal)
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"{obj.name}: non-finite vertex {index}")
                patch.write(struct.pack("<6f", *values))
        written += 1
        print("RE1_DELTA", joint, len(obj.data.vertices), path)
    if written == 0:
        raise RuntimeError("no RE1 meshes found in the Blender scene")
    print("RE1_DELTA_WRITTEN", written)


if __name__ == "__main__":
    main()
