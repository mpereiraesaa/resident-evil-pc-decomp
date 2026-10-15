"""Lengthen and taper the Cerberus muzzle without separating its animated jaw.

The sculpt is specific to em1002 head joint 3 and lower-jaw joint 4. The head
and jaw retain their original rigid transforms, texture UVs, and animation.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def smoothstep(value, low, high):
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    return t * t * (3.0 - 2.0 * t)


def bump(value, centre, width):
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    options = parser.parse_args(argv)

    parts = {obj.get("re1_joint"): obj for obj in bpy.data.objects
             if obj.type == "MESH" and obj.get("re1_joint") in (3, 4)}
    if set(parts) != {3, 4} or any(
            not obj.get("re1_source_obj", "").startswith("em1002_")
            for obj in parts.values()):
        raise RuntimeError("this sculpt requires both em1002 muzzle joints")
    changed = {3: 0, 4: 0}
    largest = 0.0
    for joint, obj in parts.items():
        for vertex in obj.data.vertices:
            x, y, z = vertex.co
            if joint == 3:
                tip = smoothstep(x, 235.0, 530.0)
                region = bump(y, 113.0, 83.0) * bump(z, 0.0, 145.0)
                dx = 32.0 * tip * region
                dz = -z * 0.075 * tip * region
            else:
                tip = smoothstep(x, 35.0, 230.0)
                region = bump(y, -28.0, 74.0) * bump(z, 0.0, 105.0)
                dx = 25.0 * tip * region
                dz = -z * 0.06 * tip * region
            if abs(dx) + abs(dz) > 0.001:
                vertex.co.x += dx
                vertex.co.z += dz
                changed[joint] += 1
                largest = max(largest, math.hypot(dx, dz))
        obj.data.update()

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in parts.values()
    bpy.context.view_layer.update()
    coords = [obj.matrix_world @ Vector(corner)
              for obj in parts.values() for corner in obj.bound_box]
    lower = Vector(tuple(min(point[axis] for point in coords) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in coords) for axis in range(3)))
    centre = (lower + upper) * 0.5
    height = upper.z - lower.z
    camera = bpy.context.scene.camera
    camera.location = centre + Vector((height * 1.8, -height * 0.55, height * 0.3))
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = height * 1.5

    options.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(options.blend.resolve()))
    options.render.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(options.render.resolve())
    bpy.ops.render.render(write_still=True)
    print("RE1_CERBERUS_MUZZLE", changed,
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
