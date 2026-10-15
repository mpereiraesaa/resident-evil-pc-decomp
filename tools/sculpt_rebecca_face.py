"""Add subtle facial relief to Rebecca's refined em1023 head joint.

This keeps the hair, neck seam, UV atlas, and rigid animation joint intact.
The coordinates are specific to the em1023 TMD viewport.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sculpt_normal_update import displaced_normal


def bump(value, centre, width):
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def displace(position):
    x, y, z = position
    front = max(0.0, min(1.0, (x - 74.0) / 59.0))
    az = abs(z)
    socket = -4.0 * bump(y, 202.0, 20.0) * bump(az, 46.0, 22.0)
    upper_cheek = 6.5 * bump(y, 158.0, 28.0) * bump(az, 65.0, 32.0)
    bridge = 4.0 * bump(y, 168.0, 31.0) * bump(z, 0.0, 22.0)
    nose = 5.0 * bump(y, 137.0, 18.0) * bump(z, 0.0, 25.0)
    lower_lip = 2.5 * bump(y, 104.0, 12.0) * bump(z, 0.0, 29.0)
    chin = 3.5 * bump(y, 70.0, 21.0) * bump(z, 0.0, 38.0)
    dx = front * (socket + upper_cheek + bridge + nose + lower_lip + chin)
    jaw = 0.045 * bump(y, 87.0, 35.0) * bump(az, 74.0, 28.0)
    return Vector((dx, 0.0, -z * jaw * front))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    options = parser.parse_args(argv)

    head = next(obj for obj in bpy.data.objects
                if obj.type == "MESH" and obj.get("re1_joint") == 1)
    if not head.get("re1_source_obj", "").startswith("em1023_"):
        raise RuntimeError("this sculpt is specific to Rebecca's em1023 head")
    normals = head.data.attributes.get("re1_export_normal")
    if normals is None:
        raise RuntimeError("Rebecca head has no source normal attribute")
    changed = 0
    largest = 0.0
    for index, vertex in enumerate(head.data.vertices):
        position = vertex.co.copy()
        offset = displace(position)
        if offset.length > 0.001:
            normals.data[index].vector = displaced_normal(
                position, normals.data[index].vector, displace)
            vertex.co += offset
            changed += 1
            largest = max(largest, offset.length)
    head.data.update()

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = obj != head
    bpy.context.view_layer.update()
    coords = [head.matrix_world @ Vector(corner) for corner in head.bound_box]
    lower = Vector(tuple(min(point[axis] for point in coords) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in coords) for axis in range(3)))
    centre = (lower + upper) * 0.5
    height = upper.z - lower.z
    camera = bpy.context.scene.camera
    camera.location = centre + Vector((height * 1.8, -height * 0.55, height * 0.3))
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = height * 1.35

    options.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(options.blend.resolve()))
    options.render.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(options.render.resolve())
    bpy.ops.render.render(write_still=True)
    print("RE1_REBECCA_FACE", "changed", changed,
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
