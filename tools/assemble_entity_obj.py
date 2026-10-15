"""Assemble RE1's per-joint OBJ exports in Blender's rest pose.

Run with Blender's bundled Python:
  blender --background --python tools/assemble_entity_obj.py -- \
      build/model-viewer/entity-mesh/obj-export/chris_rig.json \
      --blend build/model-viewer/entity-mesh/obj-export/chris.blend \
      --render build/model-viewer/entity-mesh/obj-export/chris.png

The source TMD viewport has Y up, while the GTE's rest joint translations
have Y down. Blender's OBJ importer maps viewport Y to Blender Z.
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def obj_metadata(path):
    mesh_hash = None
    element_count = None
    counts = {}
    faces = {}
    normals = []
    colors = []
    with path.open() as source:
        for line in source:
            if line.startswith("# re1_mesh_hash "):
                mesh_hash = line.split()[2]
            elif line.startswith("# re1_element_count "):
                element_count = int(line.split()[2])
            elif line.startswith("# re1_element "):
                _, _, index, count, lists, sides = line.split()
                counts[int(index)] = int(count)
                faces[int(index)] = int(lists) * max(int(sides) - 2, 0)
            elif line.startswith("# re1_color "):
                colors.append(tuple(float(value) for value in line.split()[2:5]))
            elif line.startswith("vn "):
                normals.append(tuple(float(value) for value in line.split()[1:4]))
    if element_count is None:
        return None
    return (mesh_hash,
            [counts.get(index, 0) for index in range(element_count)],
            [faces.get(index, 0) for index in range(element_count)],
            normals, colors)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--focus-joint", type=int)
    return parser.parse_args(argv)


def main():
    options = args()
    manifest = options.manifest.resolve()
    data = json.loads(manifest.read_text())
    joints = data["joints"]
    assert len(joints) == data["jointCount"]
    by_index = {joint["index"]: joint for joint in joints}

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    assembled = {}

    def rest_position(index):
        if index in assembled:
            return assembled[index]
        joint = by_index[index]
        if joint["parent"] < 0:
            position = Vector((0.0, 0.0, 0.0))
        else:
            offset = joint["restTranslation"]
            position = rest_position(joint["parent"]) + Vector(
                (offset[0], -offset[1], offset[2])
            )
        assembled[index] = position
        return position

    objects = []
    for joint in joints:
        filename = joint["obj"]
        if not filename:
            continue
        path = manifest.parent / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        bpy.ops.wm.obj_import(filepath=str(path))
        imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        if len(imported) != 1:
            raise RuntimeError(f"expected one mesh in {path}, got {len(imported)}")
        obj = imported[0]
        obj.name = f"joint_{joint['index']:02d}_{path.stem}"
        # OBJ import's rotation maps local (X,Y,Z) to Blender (X,-Z,Y).
        x, y, z = rest_position(joint["index"])
        obj.location = (x, -z, y)
        bpy.context.view_layer.update()
        obj["re1_joint"] = joint["index"]
        obj["re1_parent"] = joint["parent"]
        obj["re1_source_obj"] = filename
        obj["re1_bind_matrix"] = [obj.matrix_world[row][col]
                                  for row in range(4) for col in range(4)]
        metadata = obj_metadata(path)
        if metadata is not None:
            mesh_hash, counts, face_counts, normals, colors = metadata
            if sum(counts) != len(obj.data.vertices) or len(normals) != len(obj.data.vertices):
                raise RuntimeError(f"Blender changed vertex order/count for {path}")
            if sum(face_counts) != len(obj.data.polygons):
                raise RuntimeError(f"Blender changed face order/count for {path}")
            if colors and len(colors) != len(obj.data.vertices):
                raise RuntimeError(f"colour metadata count mismatch for {path}")
            obj["re1_mesh_hash"] = mesh_hash
            obj["re1_element_counts"] = json.dumps(counts)
            normal_attribute = obj.data.attributes.new(
                name="re1_export_normal", type="FLOAT_VECTOR", domain="POINT")
            for index, normal in enumerate(normals):
                normal_attribute.data[index].vector = normal
            color_attribute = obj.data.attributes.new(
                name="re1_color", type="FLOAT_COLOR", domain="POINT")
            for index in range(len(obj.data.vertices)):
                r, g, b = colors[index] if colors else (1.0, 1.0, 1.0)
                color_attribute.data[index].color = (r, g, b, 1.0)
            element_attribute = obj.data.attributes.new(
                name="re1_element", type="INT", domain="FACE")
            cursor = 0
            for element, face_count in enumerate(face_counts):
                for index in range(cursor, cursor + face_count):
                    element_attribute.data[index].value = element
                cursor += face_count
        objects.append(obj)

    if not objects:
        raise RuntimeError("manifest contains no drawable meshes")

    framed = objects
    if options.focus_joint is not None:
        framed = [obj for obj in objects if obj["re1_joint"] == options.focus_joint]
        if not framed:
            raise RuntimeError(f"joint {options.focus_joint} has no mesh")
        for obj in objects:
            obj.hide_render = obj not in framed
    coords = [obj.matrix_world @ Vector(corner)
              for obj in framed for corner in obj.bound_box]
    lower = Vector(tuple(min(point[axis] for point in coords) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in coords) for axis in range(3)))
    center = (lower + upper) * 0.5
    height = max(upper.z - lower.z, 1.0)

    camera_data = bpy.data.cameras.new("InspectionCamera")
    camera = bpy.data.objects.new("InspectionCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    # The player heads' facial geometry points along local +X. Keep a small
    # side component so the shoulder and elbow joins remain visible.
    camera.location = center + Vector((height * 1.8, -height * 0.55, height * 0.3))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = height * 1.35
    camera_data.clip_end = height * 10.0
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 1100
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    if options.blend:
        options.blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(options.blend.resolve()))
    if options.render:
        options.render.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(options.render.resolve())
        bpy.ops.render.render(write_still=True)

    print("RE1_ASSEMBLY", data["model"], "parts", len(objects),
          "vertices", sum(len(obj.data.vertices) for obj in objects),
          "faces", sum(len(obj.data.polygons) for obj in objects),
          "bounds", tuple(lower), tuple(upper))


if __name__ == "__main__":
    main()
