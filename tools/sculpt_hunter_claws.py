"""Extend and taper both Hunter claw joints without changing their bind poses.

Apply to a Blender scene assembled from em1006's 16x refined OBJ export.
The two hands are mirror images in local Z and use the same deformation.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    options = parser.parse_args(argv)

    hands = [obj for obj in bpy.data.objects
             if obj.type == "MESH" and obj.get("re1_joint") in (6, 9)]
    if len(hands) != 2:
        raise RuntimeError("expected Hunter hand joints 6 and 9")
    changed = 0
    largest = 0.0
    for hand in hands:
        for vertex in hand.data.vertices:
            x, y, z = vertex.co
            # The wrist is near Y=0. The claw tips extend to about Y=-516.
            # Fade the edit in only beyond the broad palm (about Y=-250).
            t = smoothstep((-y - 245.0) / 270.0)
            if t <= 0.0:
                continue
            tip = t * t
            dx = 15.0 * tip - 0.08 * t * (x + 55.0)
            dy = -64.0 * tip
            dz = -0.085 * t * z
            vertex.co.x += dx
            vertex.co.y += dy
            vertex.co.z += dz
            changed += 1
            largest = max(largest, math.sqrt(dx*dx + dy*dy + dz*dz))
        hand.data.update()

    # Frame one hand for an inspectable clay before/after comparison.
    focus = next(hand for hand in hands if hand["re1_joint"] == 6)
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = obj != focus
    bpy.context.view_layer.update()
    coords = [focus.matrix_world @ Vector(corner) for corner in focus.bound_box]
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
    print("RE1_HUNTER_CLAWS", "changed", changed,
          "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
