"""A conservative facial shape pass on Jill's refined RE1 head.

Run with Blender's bundled Python against the assembled refined .blend. This
keeps the original vertex order, UVs, and joint so export_entity_delta.py can
send the result through the game's own animation and texture path.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def gaussian(value, centre, width):
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    options = parser.parse_args(argv)

    head = next(obj for obj in bpy.data.objects
                if obj.type == "MESH" and obj.get("re1_joint") == 1)
    changed = 0
    largest = 0.0
    for vertex in head.data.vertices:
        # OBJ local X is forward. The importer retains local XYZ and puts the
        # Y-up to Z-up conversion on the object's rotation matrix.
        x, y, z = vertex.co.x, vertex.co.y, vertex.co.z
        front = max(0.0, min(1.0, (x - 82.0) / 42.0))
        if front == 0.0:
            continue
        az = abs(z)

        # Pull the eye sockets back and give the cheeks a gentle forward arc.
        eye = -5.5 * gaussian(y, 272.0, 24.0) * gaussian(az, 48.0, 24.0)
        cheek = 7.0 * gaussian(y, 220.0, 30.0) * gaussian(az, 63.0, 35.0)
        # Reinforce the existing bridge and separate the lower lip from chin.
        bridge = 3.5 * gaussian(y, 240.0, 24.0) * gaussian(z, 0.0, 24.0)
        lip = 2.5 * gaussian(y, 172.0, 10.0) * gaussian(z, 0.0, 34.0)
        chin = 3.5 * gaussian(y, 136.0, 23.0) * gaussian(z, 0.0, 39.0)
        dx = front * (eye + cheek + bridge + lip + chin)

        # Reduce the square lower-face outline, strongest at the jaw angle.
        jaw = 0.065 * gaussian(y, 151.0, 39.0) * gaussian(az, 72.0, 38.0)
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
    print("RE1_JILL_FACE", "changed", changed, "of", len(head.data.vertices),
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
