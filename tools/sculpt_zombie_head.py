"""Shape pass for the white-coat zombie's refined head (em1000 joint 2).

Run on the .blend assembled from the 16x refined OBJ export. Vertex order and
topology are preserved for export_entity_delta.py.
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
        # The OBJ importer leaves source XYZ on the mesh and rotates the
        # object into Blender Z-up. The face points along local +X.
        x, y, z = vertex.co
        front = max(0.0, min(1.0, (x - 142.0) / 60.0))
        if front == 0.0:
            continue
        az = abs(z)

        brow = 9.0 * bump(y, 266.0, 22.0) * bump(az, 53.0, 33.0)
        eye = -8.0 * bump(y, 240.0, 17.0) * bump(az, 49.0, 22.0)
        gaunt_cheek = -18.0 * bump(y, 179.0, 35.0) * bump(az, 74.0, 32.0)
        bridge = 6.0 * bump(y, 205.0, 37.0) * bump(z, 0.0, 27.0)
        mouth = -5.0 * bump(y, 103.0, 17.0) * bump(z, 0.0, 48.0)
        dx = front * (brow + eye + gaunt_cheek + bridge + mouth)

        # Narrow the heavy lower jaw without moving the neck or crown.
        jaw = 0.11 * bump(y, 115.0, 49.0) * bump(az, 87.0, 39.0)
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
    print("RE1_ZOMBIE_HEAD", "changed", changed, "of", len(head.data.vertices),
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
