"""Add restrained facial relief to Barry's refined em1022 head joint.

Run inside Blender on the scene from assemble_entity_obj.py. Coordinates are
specific to em1022 joint 1; the mesh keeps its UVs and original rigid joint.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def bump(value, centre, width):
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    options = parser.parse_args(argv)

    head = next(obj for obj in bpy.data.objects
                if obj.type == "MESH" and obj.get("re1_joint") == 1)
    if not head.get("re1_source_obj", "").startswith("em1022_"):
        raise RuntimeError("this sculpt is specific to Barry's em1022 head")

    changed = 0
    largest = 0.0
    for vertex in head.data.vertices:
        x, y, z = vertex.co
        front = max(0.0, min(1.0, (x - 130.0) / 95.0))
        if front == 0.0:
            continue
        az = abs(z)
        brow = 8.0 * bump(y, 318.0, 27.0) * bump(az, 53.0, 32.0)
        socket = -6.5 * bump(y, 283.0, 24.0) * bump(az, 53.0, 26.0)
        bridge = 4.0 * bump(y, 272.0, 37.0) * bump(z, 0.0, 27.0)
        nose = 7.0 * bump(y, 226.0, 28.0) * bump(z, 0.0, 34.0)
        cheek = 5.5 * bump(y, 222.0, 38.0) * bump(az, 87.0, 40.0)
        mouth = -4.0 * bump(y, 159.0, 18.0) * bump(z, 0.0, 46.0)
        lower_lip = 3.0 * bump(y, 141.0, 17.0) * bump(z, 0.0, 42.0)
        chin = 5.5 * bump(y, 103.0, 31.0) * bump(z, 0.0, 47.0)
        dx = front * (brow + socket + bridge + nose + cheek + mouth + lower_lip + chin)
        jaw = 0.055 * bump(y, 118.0, 56.0) * bump(az, 93.0, 43.0)
        dz = -z * jaw * front
        if abs(dx) + abs(dz) > 0.001:
            vertex.co.x += dx
            vertex.co.z += dz
            changed += 1
            largest = max(largest, math.hypot(dx, dz))
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
    print("RE1_BARRY_FACE", "changed", changed,
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
