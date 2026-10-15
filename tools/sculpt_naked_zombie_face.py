"""Give the naked zombie's texture-only face restrained anatomical relief.

Use on the refined em1101 joint-2 Blender scene. The eye, nose, mouth, and
jaw positions are specific to this EMD's head coordinates and UV artwork.
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
                if obj.type == "MESH" and obj.get("re1_joint") == 2)
    changed = 0
    largest = 0.0
    for vertex in head.data.vertices:
        x, y, z = vertex.co
        # The face points along +X; preserve the scalp and back of the skull.
        front = max(0.0, min(1.0, (x - 125.0) / 65.0))
        if front == 0.0:
            continue
        az = abs(z)
        brow = 10.0 * bump(y, 317.0, 23.0) * bump(az, 49.0, 31.0)
        eye = -8.0 * bump(y, 283.0, 20.0) * bump(az, 47.0, 25.0)
        bridge = 11.0 * bump(y, 258.0, 42.0) * bump(z, 0.0, 27.0)
        nose = 19.0 * bump(y, 211.0, 25.0) * bump(z, 0.0, 34.0)
        cheek = -12.0 * bump(y, 224.0, 35.0) * bump(az, 79.0, 34.0)
        mouth = -7.0 * bump(y, 165.0, 20.0) * bump(z, 0.0, 41.0)
        lip_edge = 5.0 * bump(y, 170.0, 23.0) * bump(az, 48.0, 20.0)
        chin = 6.0 * bump(y, 116.0, 24.0) * bump(z, 0.0, 41.0)
        dx = front * (brow + eye + bridge + nose + cheek + mouth + lip_edge + chin)
        jaw = 0.09 * bump(y, 138.0, 50.0) * bump(az, 95.0, 42.0)
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
    print("RE1_NAKED_ZOMBIE_FACE", "changed", changed,
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
