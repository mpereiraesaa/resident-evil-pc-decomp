"""Shared machinery for hand-authored Room3D meshes.

The game's Room3D loader renders OBJ triangles with point sampling, clamped
texture addressing and no lighting, so these helpers cut large surfaces into
tile-aligned cells whose UVs stay inside 0..1 and bake an authored light rig
into vertex colours. Coordinates are RE1 room units with negative Y up.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

def unit(v):
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / length for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def splits(start, length, tile, cell):
    """Cell boundaries along one axis: tile multiples plus lighting cells."""
    end = start + length
    marks = {start, end}
    if tile:
        first = math.floor(start / tile) + 1
        marks.update(k * tile for k in range(first, math.ceil(end / tile)))
    ordered = sorted(marks)
    result = [ordered[0]]
    for a, b in zip(ordered, ordered[1:]):
        pieces = max(1, math.ceil((b - a) / cell))
        result.extend(a + (b - a) * i / pieces for i in range(1, pieces + 1))
    return result


class Mesh:
    """OBJ writer for the Room3D loader: quads with UVs and baked vertex colours.

    `materials` maps material name to a spec dict ({"map", "tile", "kd",
    "emissive"}); `illumination(point, normal)` returns the baked colour.
    """

    def __init__(self, name, materials, illumination, header=(), cell=600.0):
        self.materials = materials
        self.illumination = illumination
        self.cell = cell
        self.lines = [*header, f"mtllib {name}.mtl"]
        self.vertex_count = 0
        self.uv_count = 0
        self.triangles = 0
        self.group = None

    def set_group(self, group):
        if group != self.group:
            self.lines.append("g " + group if group else "g")
            self.group = group

    def emit(self, points, uvs, material, normal):
        spec = self.materials[material]
        refs = []
        for point, uv in zip(points, uvs):
            if spec.get("emissive"):
                color = (1.0, 1.0, 1.0)
            else:
                color = self.illumination(point, normal)
            self.vertex_count += 1
            self.uv_count += 1
            self.lines.append("v %.0f %.0f %.0f %.3f %.3f %.3f" % (*point, *color))
            self.lines.append("vt %.4f %.4f" % uv)
            refs.append(f"{self.vertex_count}/{self.uv_count}")
        self.lines.append(f"usemtl {material}")
        self.lines.append("f " + " ".join(refs))
        self.triangles += len(points) - 2

    def rect(self, origin, eu, ev, lu, lv, material, flip=False):
        """Planar rectangle: origin plus lu along unit eu and lv along unit ev.

        Texture tiles are phased on world coordinates so neighbouring rectangles
        line up; each emitted cell stays inside one tile (clamped sampler).
        """
        eu, ev = unit(eu), unit(ev)
        normal = cross(ev, eu) if flip else cross(eu, ev)
        tile = self.materials[material].get("tile")
        su0, sv0 = dot(origin, eu), dot(origin, ev)
        tu, tv = tile if tile else (None, None)
        us = splits(su0, lu, tu, self.cell)
        vs = splits(sv0, lv, tv, self.cell)

        def uv_of(s, s0, length, size):
            if size:
                base = math.floor(s0 / size + 1e-9) * size
                return (s - base) / size
            return (s - s0) / length

        for a, b in zip(us, us[1:]):
            for c, d in zip(vs, vs[1:]):
                corners = []
                uvs = []
                for s, t in ((a, c), (b, c), (b, d), (a, d)):
                    corners.append(tuple(origin[k] + eu[k] * (s - su0) + ev[k] * (t - sv0) for k in range(3)))
                    uvs.append((min(1.0, uv_of(s, a, lu, tu) if tu else uv_of(s, su0, lu, None)),
                                min(1.0, uv_of(t, c, lv, tv) if tv else uv_of(t, sv0, lv, None))))
                if flip:
                    corners.reverse()
                    uvs.reverse()
                self.emit(corners, uvs, material, normal)

    def poly(self, points, material, flip=False):
        """Arbitrary planar polygon; the whole texture is stretched across it."""
        p0, p1, p2 = points[0], points[1], points[-1]
        normal = unit(cross(tuple(p1[k] - p0[k] for k in range(3)),
                            tuple(p2[k] - p0[k] for k in range(3))))
        if flip:
            normal = tuple(-c for c in normal)
        axis = unit(tuple(p1[k] - p0[k] for k in range(3)))
        other = unit(cross(normal, axis))
        us = [dot(p, axis) for p in points]
        vs = [dot(p, other) for p in points]
        du = (max(us) - min(us)) or 1.0
        dv = (max(vs) - min(vs)) or 1.0
        uvs = [((u - min(us)) / du, (v - min(vs)) / dv) for u, v in zip(us, vs)]
        self.emit(points, uvs, material, normal)

    def box(self, lo, hi, material, omit=()):
        """Axis-aligned box; `lo` holds the smaller coordinates (so the top is y0)."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        faces = {
            "top": ((x0, y0, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, False),
            "bottom": ((x0, y1, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, True),
            "south": ((x0, y1, z0), (1, 0, 0), (0, -1, 0), x1 - x0, y1 - y0, False),
            "north": ((x0, y1, z1), (1, 0, 0), (0, -1, 0), x1 - x0, y1 - y0, True),
            "west": ((x0, y1, z0), (0, 0, 1), (0, -1, 0), z1 - z0, y1 - y0, True),
            "east": ((x1, y1, z0), (0, 0, 1), (0, -1, 0), z1 - z0, y1 - y0, False),
        }
        for name, (origin, eu, ev, lu, lv, flip) in faces.items():
            if name in omit or lu <= 0 or lv <= 0:
                continue
            self.rect(origin, eu, ev, lu, lv, material, flip)

    def sphere(self, centre, radius, material, rings=6, segments=10, stretch=1.0):
        """Sphere; `stretch` scales it vertically (an egg or a hanging nest)."""
        cx, cy, cz = centre
        for i in range(rings):
            t0, t1 = math.pi * i / rings, math.pi * (i + 1) / rings
            for j in range(segments):
                p0, p1 = math.tau * j / segments, math.tau * (j + 1) / segments
                ring = []
                for t, p in ((t0, p0), (t1, p0), (t1, p1), (t0, p1)):
                    ring.append((cx + radius * math.sin(t) * math.cos(p),
                                 cy - radius * stretch * math.cos(t),
                                 cz + radius * math.sin(t) * math.sin(p)))
                if i == 0:
                    ring = [ring[0], ring[1], ring[2]]
                elif i == rings - 1:
                    ring = [ring[0], ring[1], ring[3]]
                self.poly(ring, material)

    def cylinder(self, x, z, radius, top, bottom, material, segments=10, caps=True):
        ring = [(x + radius * math.cos(i * math.tau / segments),
                 z + radius * math.sin(i * math.tau / segments)) for i in range(segments)]
        for i, (a, b) in enumerate(ring):
            c, d = ring[(i + 1) % segments]
            # side faces: normal flipped to point outward (the winding stays)
            self.poly([(a, bottom, b), (a, top, b), (c, top, d), (c, bottom, d)], material, flip=True)
        if caps:
            self.poly([(a, top, b) for a, b in ring], material)
            self.poly([(a, bottom, b) for a, b in reversed(ring)], material)


    def tube(self, start, end, radius, material, segments=10, caps=True):
        """Cylinder between two arbitrary points (pipes, lying tanks, rolls)."""
        axis = tuple(end[k] - start[k] for k in range(3))
        length = math.sqrt(sum(c * c for c in axis))
        if length < 1:
            return
        w = tuple(c / length for c in axis)
        helper = (0.0, 1.0, 0.0) if abs(w[1]) < 0.9 else (1.0, 0.0, 0.0)
        u = unit(cross(w, helper))
        v = cross(w, u)

        def at(base, angle):
            c, s_ = math.cos(angle) * radius, math.sin(angle) * radius
            return tuple(base[k] + u[k] * c + v[k] * s_ for k in range(3))
        angles = [i * math.tau / segments for i in range(segments)]
        for i, a in enumerate(angles):
            b = angles[(i + 1) % segments]
            self.poly([at(start, a), at(end, a), at(end, b), at(start, b)], material, flip=True)
        if caps:
            self.poly([at(start, a) for a in reversed(angles)], material)
            self.poly([at(end, a) for a in angles], material)


# ------------------------------------------------------------- furniture --


def balustrade(mesh, start, end, side=None, newels=(True, True), height=950, material="wood", spacing=320):
    """Handrail on balusters from `start` to `end` (3D, may slope).

    `side` is the unit horizontal direction pointing away from the walkway, used
    to offset nothing but kept for readability of the call sites.
    """
    sx, sy, sz = start
    ex, ey, ez = end
    length = math.hypot(ex - sx, ez - sz)
    dx, dz = (ex - sx) / length, (ez - sz) / length
    count = max(2, int(length // spacing))
    for i in range(count + 1):
        f = i / count
        x, y, z = sx + (ex - sx) * f, sy + (ey - sy) * f, sz + (ez - sz) * f
        if (i == 0 and newels[0]) or (i == count and newels[1]):
            continue
        mesh.box((x - 36, y - height + 60, z - 36), (x + 36, y, z + 36), material, omit=("bottom",))
    for pos, flag in ((start, newels[0]), (end, newels[1])):
        if not flag:
            continue
        px, py, pz = pos
        mesh.box((px - 100, py - height - 250, pz - 100), (px + 100, py, pz + 100), material, omit=("bottom",))
        mesh.box((px - 130, py - height - 340, pz - 130), (px + 130, py - height - 250, pz + 130), material)
        mesh.sphere((px, py - height - 440, pz), 110, material, rings=4, segments=8)
    # Handrail: a sloped bar built from four quads around the rail line.
    rail_half, rail_depth = 55, 90
    px, pz = -dz, dx  # perpendicular in the floor plane
    top = [(sx + px * rail_half, sy - height, sz + pz * rail_half),
           (ex + px * rail_half, ey - height, ez + pz * rail_half),
           (ex - px * rail_half, ey - height, ez - pz * rail_half),
           (sx - px * rail_half, sy - height, sz - pz * rail_half)]
    bottom = [(p[0], p[1] + rail_depth, p[2]) for p in top]
    mesh.poly(top, material)
    mesh.poly(list(reversed(bottom)), material)
    mesh.poly([top[0], top[3], bottom[3], bottom[0]], material)
    mesh.poly([top[2], top[1], bottom[1], bottom[2]], material)


def door(mesh, x0, x1, z, top, material, face, frame_depth=110, floor=0, frame="wood"):
    """Door panel recessed into a wall running along X; `face` is the room side."""
    inner = z + face * 70
    if face > 0:
        mesh.rect((x1, floor, inner), (-1, 0, 0), (0, -1, 0), x1 - x0, floor - top, material)
    else:
        mesh.rect((x0, floor, inner), (1, 0, 0), (0, -1, 0), x1 - x0, floor - top, material)
    za, zb = (z, z + frame_depth) if face > 0 else (z - frame_depth, z)
    for xa in (x0 - 90, x1):
        mesh.box((xa, top - 110, za), (xa + 90, floor, zb), frame, omit=("bottom",))
    mesh.box((x0 - 90, top - 110, za), (x1 + 90, top, zb), frame)
    # Reveal between the frame and the recessed panel.
    for xa in (x0, x1 - 1):
        pts = [(xa, floor, z), (xa, top, z), (xa, top, inner), (xa, floor, inner)]
        mesh.poly(pts if (xa == x0) == (face > 0) else list(reversed(pts)), frame)
    mesh.poly([(x0, top, z), (x1, top, z), (x1, top, inner), (x0, top, inner)] if face > 0 else
              [(x1, top, z), (x0, top, z), (x0, top, inner), (x1, top, inner)], frame)


def value_noise(shape, rng, octaves=4, base_freq=4, persistence=0.55):
    height, width = shape
    total = np.zeros((height, width))
    amplitude, freq, norm = 1.0, base_freq, 0.0
    for _ in range(octaves):
        grid = rng.random((freq + 1, freq + 1))
        grid[-1, :] = grid[0, :]
        grid[:, -1] = grid[:, 0]
        ys = np.linspace(0, freq, height, endpoint=False)
        xs = np.linspace(0, freq, width, endpoint=False)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        fy = ys - y0
        fx = xs - x0
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        a = grid[y0][:, x0]
        b = grid[y0][:, x0 + 1]
        c = grid[y0 + 1][:, x0]
        d = grid[y0 + 1][:, x0 + 1]
        layer = (a * (1 - fx) + b * fx) * (1 - fy)[:, None] + (c * (1 - fx) + d * fx) * fy[:, None]
        total += amplitude * layer
        norm += amplitude
        amplitude *= persistence
        freq *= 2
    return total / norm


def to_image(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"), "RGB")


def tint(mask, base, color):
    return base * (1 - mask[..., None]) + np.array(color) * mask[..., None]


def write_mtl(out, name, materials):
    lines = [f"# Materials for the hand-authored {name} mesh."]
    for material, spec in materials.items():
        kd = spec.get("kd", (1, 1, 1))
        lines.append(f"newmtl {material}")
        lines.append("Kd %.3f %.3f %.3f" % kd)
        if "map" in spec:
            lines.append(f"map_Kd {spec['map']}")
    (out / f"{name}.mtl").write_text("\n".join(lines) + "\n")


def preview(out, name, rdt):
    from room3d_preview import render
    from survey_room3d import survey

    cameras = survey(Path(rdt))["cameras"]
    for camera in cameras:
        image, _ = render(out / f"{name}.obj", camera, width=640, height=480)
        image.save(out / f"preview-camera{camera['index']}.png")
        print(out / f"preview-camera{camera['index']}.png")


def compare_sheet(out, name, originals, count, columns=2):
    """Side-by-side sheet: original view (left) and preview render (right)."""
    rows = math.ceil(count / columns)
    sheet = Image.new("RGB", (columns * 640, rows * 240))
    for index in range(count):
        original = Image.open(Path(originals) / f"camera{index}-original.png").convert("RGB")
        render = Image.open(out / f"preview-camera{index}.png").convert("RGB")
        x = (index % columns) * 640
        y = (index // columns) * 240
        sheet.paste(original.resize((320, 240), Image.NEAREST), (x, y))
        sheet.paste(render.resize((320, 240)), (x + 320, y))
    sheet.save(out / "compare.png")
    return out / "compare.png"


def edge_agreement(out, name, originals, count, tolerance=3):
    """Per-camera edge overlap between the preview render and the original view.

    Returns a list of (camera, precision, recall): the fraction of render edges
    within `tolerance` pixels of an original edge, and vice versa. A coarse,
    hue-independent sanity check for layout, not a fidelity score.
    """
    def edges(image):
        grey = np.asarray(image.convert("L").resize((320, 240)), dtype=float)
        gx = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1]))
        gy = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :]))
        magnitude = gx + gy
        threshold = max(12.0, np.percentile(magnitude, 88))
        return magnitude > threshold

    def dilate(mask, radius):
        result = mask.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                result |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return result

    rows = []
    for index in range(count):
        original = edges(Image.open(Path(originals) / f"camera{index}-original.png"))
        render = edges(Image.open(out / f"preview-camera{index}.png"))
        precision = (render & dilate(original, tolerance)).sum() / max(1, render.sum())
        recall = (original & dilate(render, tolerance)).sum() / max(1, original.sum())
        rows.append((index, float(precision), float(recall)))
    return rows
