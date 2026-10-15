"""Round the outer shoulder contour on an assembled human torso joint.

This works in the TMD viewport's local coordinates on joint 0. It preserves
the rigid arm joints, torso UVs, and smooth exported normals. Inspect each
model's textured pose before installing its .r1mesh replacement.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sculpt_normal_update import displaced_normal


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def bump(value, centre, width):
    return math.exp(-0.5 * ((value - centre) / width) ** 2)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.10)
    options = parser.parse_args(argv)
    if not 0.0 < options.strength <= 0.2:
        raise ValueError("strength must be in (0, 0.2]")

    torso = next(obj for obj in bpy.data.objects
                 if obj.type == "MESH" and obj.get("re1_joint") == 0)
    normals = torso.data.attributes.get("re1_export_normal")
    if normals is None:
        raise RuntimeError("torso has no source normal attribute")
    # OBJ viewport (X,Y,Z) becomes Blender (X,-Z,Y). The left/right arm joint
    # translations confirm width is viewport Z, hence Blender Y; height is Z.
    z_min = min(vertex.co.z for vertex in torso.data.vertices)
    z_max = max(vertex.co.z for vertex in torso.data.vertices)
    y_max = max(abs(vertex.co.y) for vertex in torso.data.vertices)
    z_span = z_max - z_min

    def displace(position):
        _, y, z = position
        side = smoothstep((abs(y) - 0.55 * y_max) / (0.45 * y_max))
        shoulder = bump(z, z_min + 0.82 * z_span, 0.17 * z_span)
        ridge = bump(z, z_min + 0.88 * z_span, 0.13 * z_span)
        dy = -math.copysign(options.strength * y_max * side * shoulder, y)
        dz = -0.025 * z_span * side * ridge
        return Vector((0.0, dy, dz))

    changed = 0
    largest = 0.0
    for index, vertex in enumerate(torso.data.vertices):
        position = vertex.co.copy()
        offset = displace(position)
        if offset.length > 0.001:
            normals.data[index].vector = displaced_normal(
                position, normals.data[index].vector, displace)
            vertex.co += offset
            changed += 1
            largest = max(largest, offset.length)
    torso.data.update()

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = obj != torso
    bpy.context.view_layer.update()
    coords = [torso.matrix_world @ Vector(corner) for corner in torso.bound_box]
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
    print("RE1_HUMAN_SHOULDERS", torso.get("re1_source_obj"),
          "changed", changed, "largest displacement", round(largest, 3))


if __name__ == "__main__":
    main()
