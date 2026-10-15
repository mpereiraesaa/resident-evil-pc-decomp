"""Refine Wesker's em1024 head and reduce the bulky glasses silhouette.

The coordinates are specific to joint 1 of the original em1024 viewport. This
only moves vertices; the original atlas and rigid animation joint are retained.
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
    if not head.get("re1_source_obj", "").startswith("em1024_"):
        raise RuntimeError("this sculpt is specific to Wesker's em1024 head")
    changed = 0
    largest = 0.0
    for vertex in head.data.vertices:
        x, y, z = vertex.co
        az = abs(z)
        front = max(0.0, min(1.0, (x - 160.0) / 90.0))
        # The sunglass lenses are the outer high-X bulges around local Y=350.
        # Keep the bridge and temples, while making each lens less spherical.
        lens = front * bump(y, 353.0, 37.0) * bump(az, 77.0, 36.0)
        lens *= max(0.0, min(1.0, (x - 220.0) / 30.0))
        dx = -11.0 * lens
        dy = -0.11 * (y - 353.0) * lens

        # Shape the cheek under the glasses and the narrow, angular jaw.
        dx += front * (5.0 * bump(y, 267.0, 37.0) * bump(az, 67.0, 37.0)
                       + 3.5 * bump(y, 191.0, 29.0) * bump(z, 0.0, 40.0))
        jaw = 0.045 * bump(y, 198.0, 67.0) * bump(az, 101.0, 44.0)
        dz = -z * jaw * front
        if abs(dx) + abs(dy) + abs(dz) > 0.001:
            vertex.co.x += dx
            vertex.co.y += dy
            vertex.co.z += dz
            changed += 1
            largest = max(largest, math.sqrt(dx * dx + dy * dy + dz * dz))
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
    print("RE1_WESKER_FACE", "changed", changed,
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
