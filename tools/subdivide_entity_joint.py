"""Add actual triangles to one assembled RE1 joint in Blender.

Run against a .blend from assemble_entity_obj.py, then export with
export_entity_mesh.py. This keeps UV, colour, and TMD element attributes so
newly sculpted geometry can use the game's original texture and animation.
"""

import argparse
from pathlib import Path
import sys

import bmesh
import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=int, required=True)
    parser.add_argument("--cuts", type=int, default=1)
    parser.add_argument("--blend", type=Path, required=True)
    options = parser.parse_args(argv)
    if not 1 <= options.cuts <= 3:
        raise ValueError("cuts must be 1..3")
    obj = next(obj for obj in bpy.data.objects
               if obj.type == "MESH" and obj.get("re1_joint") == options.joint)
    old_vertices = len(obj.data.vertices)
    old_faces = len(obj.data.polygons)
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    bmesh.ops.subdivide_edges(mesh, edges=list(mesh.edges),
                              cuts=options.cuts, use_grid_fill=True)
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    if obj.data.attributes.get("re1_element") is None or \
            obj.data.attributes.get("re1_color") is None:
        raise RuntimeError("Blender lost the RE1 element or colour attributes")
    options.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(options.blend.resolve()))
    print("RE1_SUBDIVIDED", options.joint,
          old_vertices, "->", len(obj.data.vertices),
          old_faces, "->", len(obj.data.polygons))


if __name__ == "__main__":
    main()
