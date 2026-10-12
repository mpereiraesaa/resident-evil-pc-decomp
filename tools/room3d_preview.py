"""Offline OBJ preview for room comparison; no game state or assets are modified.

Projection follows Room3D.cpp (320x240 centre, RDT focal length, near=2*f).
This diagnostic rasterizer is not a replacement for in-game validation.
"""

import numpy as np
from PIL import Image


def face_chart(points):
    """Continuous unit chart shared by all triangles of one planar OBJ face."""
    points = np.asarray(points, float)
    origin = points[0]
    u = points[1] - origin
    normal = np.cross(u, points[2] - origin)
    if min(np.linalg.norm(u), np.linalg.norm(normal)) < 1e-10:
        return None
    u /= np.linalg.norm(u)
    normal /= np.linalg.norm(normal)
    axes = np.array([u, np.cross(normal, u)])
    projected = (points - origin) @ axes.T
    return origin, axes, projected.min(0), np.maximum(np.ptp(projected, axis=0), 1e-10)


def load_obj(path, camera_index, only_groups=None, only_materials=None, with_face_charts=False):
    vertices, uvs, triangles, materials = [], [], [], {}
    charts = []
    hidden = set()
    sidecar = path.with_suffix(".cameras")
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            words = line.split()
            if len(words) == 3 and words[0] == "hide" and int(words[1]) == camera_index:
                hidden.add(words[2])
    groups, material = set(), ""
    for line in path.read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        kind, args = words[0], words[1:]
        if kind == "mtllib":
            current = None
            for mline in (path.parent / " ".join(args)).read_text().splitlines():
                m = mline.split("#", 1)[0].split()
                if not m:
                    continue
                if m[0] == "newmtl":
                    current = materials.setdefault(m[1], [np.ones(3), None])
                elif m[0] == "Kd" and current is not None:
                    current[0] = np.array(m[1:4], dtype=float)
                elif m[0] == "map_Kd" and current is not None:
                    current[1] = (
                        np.asarray(
                            Image.open(path.parent / " ".join(m[1:])).convert("RGB")
                        )
                        / 255.0
                    )
        elif kind == "v":
            v = list(map(float, args))
            vertices.append(v[:3] + (v[3:6] if len(v) >= 6 else [1, 1, 1]))
        elif kind == "vt":
            uvs.append(list(map(float, args[:2])))
        elif kind == "g":
            groups = set(args)
        elif kind == "usemtl":
            material = " ".join(args)
        elif (
            kind == "f"
            and not groups.intersection(hidden)
            and (only_groups is None or groups.intersection(only_groups))
            and (only_materials is None or material in only_materials)
        ):
            face = []
            for ref in args:
                fields = ref.split("/")
                vi = int(fields[0])
                v = vertices[vi - 1 if vi > 0 else vi]
                ti = int(fields[1]) if len(fields) > 1 and fields[1] else None
                uv = uvs[ti - 1 if ti > 0 else ti] if ti is not None else [0, 0]
                face.append(v[:3] + uv + v[3:])
            chart = face_chart([v[:3] for v in face]) if with_face_charts else None
            for i in range(1, len(face) - 1):
                triangles.append((np.array([face[0], face[i], face[i + 1]]), material))
                if with_face_charts:
                    charts.append(chart)
    return (triangles, materials, charts) if with_face_charts else (triangles, materials)


def camera_basis(camera):
    if camera["roll"] != 0:
        raise ValueError("Nonzero camera roll is unsupported, as in Room3D.cpp")
    forward = np.array(camera["target"], dtype=float) - camera["origin"]
    length = np.linalg.norm(forward)
    horizontal = np.linalg.norm(forward[[0, 2]])
    if min(length, horizontal) < 1:
        raise ValueError("Degenerate camera basis")
    right = np.array([forward[2], 0, -forward[0]]) / horizontal
    forward /= length
    return np.array([right, np.cross(forward, right), forward])


def clip_near(poly, near):
    inside = [point[2] >= near for point in poly]
    if all(inside):
        return list(poly)
    if not any(inside):
        return []
    output = []
    for index, a in enumerate(poly):
        following = (index + 1) % len(poly)
        b = poly[following]
        if inside[index]:
            output.append(a)
        if inside[index] != inside[following]:
            output.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return output


def render(path, camera, width=320, height=240, return_depth=False, only_groups=None, only_materials=None):
    triangles, materials = load_obj(path, camera["index"], only_groups, only_materials)
    basis = camera_basis(camera)
    focal = camera["fov"]
    if focal <= 0:
        raise ValueError("Nonpositive focal length")
    pixels = np.zeros((height, width, 3), dtype=float)
    depth = np.full((height, width), np.inf)
    default_material = (np.ones(3), None)
    for triangle, name in triangles:
        triangle[:, :3] = (triangle[:, :3] - camera["origin"]) @ basis.T
        polygon = clip_near(triangle, max(1, focal * 2))
        kd, texture = materials.get(name, default_material)
        for i in range(1, len(polygon) - 1):
            t = np.array([polygon[0], polygon[i], polygon[i + 1]])
            screen = t[:, :2] / t[:, 2:3] * focal
            screen = (screen + [160, 120]) * [width / 320, height / 240]
            lo = np.maximum(np.floor(screen.min(axis=0)).astype(int), [0, 0])
            hi = np.minimum(
                np.ceil(screen.max(axis=0)).astype(int), [width - 1, height - 1]
            )
            if np.any(lo > hi):
                continue
            a, b, c = screen
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-8:
                continue
            yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            x, y = xx + 0.5, yy + 0.5
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            weights = np.stack([w0, w1, 1 - w0 - w1], axis=-1)
            inside = (weights >= -1e-7).all(axis=-1)
            reciprocal = weights / t[:, 2]
            inv_z = reciprocal.sum(axis=-1)
            safe = np.where(inv_z > 0, inv_z, 1)
            z = 1 / safe
            update = inside & (inv_z > 0) & (z < depth[yy, xx])
            if not update.any():
                continue
            attrs = (reciprocal @ t[:, 3:]) / safe[..., None]
            color = attrs[..., 2:] * kd
            if texture is not None:
                tx = np.clip(attrs[..., 0], 0, 1) * (texture.shape[1] - 1)
                ty = np.clip(1 - attrs[..., 1], 0, 1) * (texture.shape[0] - 1)
                color *= texture[np.rint(ty).astype(int), np.rint(tx).astype(int)]
            pixels[yy[update], xx[update]] = color[update]
            depth[yy[update], xx[update]] = z[update]
    image = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype("uint8"))
    if return_depth:
        return image, np.isfinite(depth), depth
    return image, np.isfinite(depth)
