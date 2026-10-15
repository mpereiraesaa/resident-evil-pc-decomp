#!/usr/bin/env python3
"""Paint Room3D meshes with the room's original camera backgrounds.

The fixed-camera backgrounds are the room's real art (and an HD or 4K pack
replaces exactly those images). This tool splits each face of a built room
mesh along the camera frames and occluders and gives every piece the camera
that sees it best - all of it inside the frame, in front of the camera, not
hidden in that camera's depth map, not edge-on, at the highest texel density.
The split mesh replaces roomSRRV.obj (the builder's output is kept as
roomSRRV.base.obj) and a sidecar records the choice:

    roomSRRV.proj    "proj1 <faces>" then one character per OBJ face, in face
                     order: '0'-'9', 'a'-'v' = camera index, '-' = keep the
                     procedural material; then "tint <material> r g b" lines
                     that pull the unprojected faces toward the art's colour.

The game (Room3D.cpp) loads each camera's background through the normal
asset/mod resolution, so an installed texture pack is used automatically,
and maps every projected piece onto it by projecting its corners with that
camera.

    room3d_projection.py assign room1060.obj --rdt ROOM1060.RDT --originals EVIDENCE
    room3d_projection.py preview room1060.obj --rdt ... --originals EVIDENCE --out DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room3d_preview import camera_basis, render  # noqa: E402
from survey_room3d import survey  # noqa: E402

DIGITS = "0123456789abcdefghijklmnopqrstuv"
SPLIT_MARKER = "# room3d_projection: faces split for background projection"


def read_vertices(obj_path):
    """The OBJ's vertex positions (n, 3)."""
    return np.array([[float(v) for v in line.split()[1:4]]
                     for line in Path(obj_path).read_text().splitlines()
                     if line.startswith("v ")], dtype=float)


def read_camera_rules(obj_path):
    """Cameras to leave out (backdrop and shade views do not show the lit
    room) and the groups each camera hides because they stand in front of it."""
    excluded, hidden = set(), {}
    sidecar = Path(obj_path).with_suffix(".cameras")
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            words = line.split("#", 1)[0].split()
            if len(words) >= 2 and words[0] in ("backdrop", "shade"):
                excluded.add(int(words[1]))
            elif len(words) == 3 and words[0] == "hide":
                hidden.setdefault(int(words[1]), set()).add(words[2])
    return excluded, hidden


def project(vertices, camera):
    """Camera-space xyz and screen xy (320x240 game space) of every vertex."""
    view = (vertices - np.array(camera["origin"], dtype=float)) @ camera_basis(camera).T
    z = view[:, 2]
    safe = np.where(np.abs(z) > 1e-6, z, 1e-6)
    screen = view[:, :2] / safe[:, None] * camera["fov"] + [160.0, 120.0]
    return view, screen


def polygon_area(points):
    """Area of a planar polygon (n, 2 or 3)."""
    if points.shape[1] == 2:
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    total = np.zeros(3)
    for i in range(1, len(points) - 1):
        total += np.cross(points[i] - points[0], points[i + 1] - points[0])
    return 0.5 * float(np.linalg.norm(total))


def face_normal(points):
    total = np.zeros(3)
    for i in range(1, len(points) - 1):
        total += np.cross(points[i] - points[0], points[i + 1] - points[0])
    length = np.linalg.norm(total)
    return total / length if length > 0 else total


def clip_polygon(poly, normal, offset):
    """Part of `poly` (n, k; xyz first) where xyz . normal + offset >= 0,
    every attribute interpolated linearly."""
    if len(poly) == 0:
        return poly
    side = poly[:, :3] @ normal + offset
    inside = side >= 0
    if inside.all():
        return poly
    if not inside.any():
        return poly[:0]
    out = []
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if inside[i]:
            out.append(poly[i])
        if inside[i] != inside[j]:
            t = side[i] / (side[i] - side[j])
            out.append(poly[i] + (poly[j] - poly[i]) * t)
    return np.array(out) if len(out) >= 3 else poly[:0]


class View:
    """One original camera prepared for the assignment: basis, frame planes
    in room space, depth map of the mesh as that camera saw it."""

    def __init__(self, camera, depth, hides, margin):
        self.index = camera["index"]
        self.camera = camera
        self.origin = np.array(camera["origin"], dtype=float)
        self.basis = camera_basis(camera)
        self.fov = float(camera["fov"])
        self.depth = depth
        self.hides = hides
        self.near = max(1.0, 2.0 * self.fov)
        planes = []
        for n_view in ((self.fov, 0, 160 - margin), (-self.fov, 0, 160 - margin),
                       (0, self.fov, 120 - margin), (0, -self.fov, 120 - margin)):
            n = self.basis.T @ np.array(n_view, dtype=float)
            n /= np.linalg.norm(n)
            planes.append((n, -n @ self.origin))
        forward = self.basis[2]
        planes.append((forward, -forward @ self.origin - self.near))
        self.planes = planes

    def view(self, xyz):
        return (xyz - self.origin) @ self.basis.T

    def screen(self, view):
        return view[:, :2] / view[:, 2:3] * self.fov + [160.0, 120.0]

    def inside(self, xyz, eps=0.5):
        return all((xyz @ n + d >= -eps).all() for n, d in self.planes)

    def clip(self, poly):
        for n, d in self.planes:
            poly = clip_polygon(poly, n, d)
        return poly

    def visible(self, xyz, slack):
        """Centroid, corners and edge midpoints (pulled 10% inward) all lie
        on the surface this camera's depth map recorded."""
        centre = xyz.mean(axis=0)
        mids = (xyz + np.roll(xyz, -1, axis=0)) * 0.5
        samples = np.vstack([centre[None], centre + (xyz - centre) * 0.9,
                             centre + (mids - centre) * 0.9])
        v = self.view(samples)
        if (v[:, 2] <= 0).any():
            return False
        s = self.screen(v)
        sx, sy = s[:, 0].astype(int), s[:, 1].astype(int)
        if (sx < 0).any() or (sx >= 320).any() or (sy < 0).any() or (sy >= 240).any():
            return False
        d = self.depth[sy, sx]
        return not (np.isfinite(d) & (v[:, 2] > d * (1.0 + slack) + 20.0)).any()

    def density(self, xyz, area):
        return polygon_area(self.screen(self.view(xyz))) / max(area, 1.0)


def split_longest_edge(poly):
    xyz = poly[:, :3]
    edges = np.roll(xyz, -1, axis=0) - xyz
    i = int(np.argmax(np.linalg.norm(edges, axis=1)))
    normal = edges[i] / max(np.linalg.norm(edges[i]), 1e-9)
    mid = xyz[i] + edges[i] * 0.5
    return [clip_polygon(poly, normal, -normal @ mid),
            clip_polygon(poly, -normal, normal @ mid)]


def assign_polygon(poly, groups, views, settings, depth=0):
    """[(piece, camera)] covering `poly`: whole when one camera sees all of
    it, else clipped at the best partial camera's frame edges or halved where
    something occludes it, down to `max_depth` levels."""
    xyz = poly[:, :3]
    area = polygon_area(xyz)
    normal = face_normal(xyz)
    centre = xyz.mean(axis=0)
    best, best_density = -1, settings["min_density"]
    partial, partial_score, occluded = None, 0.0, False
    for view in views:
        if groups & view.hides:
            continue
        ray = centre - view.origin
        ray /= max(np.linalg.norm(ray), 1e-9)
        if abs(ray @ normal) < settings["min_facing"]:
            continue
        if view.inside(xyz):
            density = view.density(xyz, area)
            if density < settings["min_density"]:
                continue
            if view.visible(xyz, settings["depth_slack"]):
                if density > best_density:
                    best, best_density = view.index, density
            else:
                occluded = True
        elif depth < settings["max_depth"]:
            clipped = view.clip(poly)
            if len(clipped):
                piece_area = polygon_area(clipped[:, :3])
                if piece_area < settings["min_piece"]:
                    continue
                density = view.density(clipped[:, :3], piece_area)
                if density < settings["min_density"]:
                    continue
                score = polygon_area(view.screen(view.view(clipped[:, :3])))
                if score > partial_score:
                    partial, partial_score = view, score
    if best >= 0:
        return [(poly, best)]
    if depth >= settings["max_depth"] or area < settings["min_piece"]:
        return [(poly, -1)]
    if occluded and area >= 4 * settings["min_piece"]:
        pieces = split_longest_edge(poly)
    elif partial is not None:
        pieces, rest = [], poly
        for n, d in partial.planes:
            outside = clip_polygon(rest, -n, -d)
            rest = clip_polygon(rest, n, d)
            if len(outside):
                pieces.append(outside)
        pieces.insert(0, rest)
    else:
        return [(poly, -1)]
    result = []
    for piece in pieces:
        if len(piece) >= 3 and polygon_area(piece[:, :3]) > 1.0:
            result += assign_polygon(piece, groups, views, settings, depth + 1)
    return result or [(poly, -1)]


def assign(obj_path, rdt_path, skip=(), min_density=0.000015, depth_slack=0.03, margin=1.0,
           min_facing=0.2, max_depth=6, min_piece=150.0 ** 2):
    """Split the mesh's faces along camera frames and occluders and pick the
    camera that paints each piece. Returns (records, assignment) where
    records are the OBJ lines with every face replaced by its pieces."""
    obj_path = Path(obj_path)
    excluded, hidden = read_camera_rules(obj_path)
    excluded |= set(skip)
    views = []
    for camera in survey(Path(rdt_path))["cameras"]:
        if camera["index"] in excluded or camera["roll"] != 0:
            continue
        try:
            _, _, depth = render(obj_path, camera, return_depth=True)
        except ValueError:
            continue
        views.append(View(camera, depth, hidden.get(camera["index"], set()), margin))
    settings = dict(min_density=min_density, depth_slack=depth_slack, min_facing=min_facing,
                    max_depth=max_depth, min_piece=min_piece)

    lines = obj_path.read_text().splitlines()
    positions, texcoords = [], []
    for line in lines:
        if line.startswith("v "):
            values = [float(x) for x in line.split()[1:]]
            positions.append(values[:6] if len(values) >= 6 else values[:3] + [1.0, 1.0, 1.0])
        elif line.startswith("vt "):
            texcoords.append([float(x) for x in line.split()[1:3]])
    records, assignment, areas, groups = [], [], [], frozenset()
    material, piece_materials, pieces, vertex_records = None, [], [], []
    vertex_count = texcoord_count = 0
    for line in lines:
        if line.startswith(("v ", "vt ")):
            continue
        if line.startswith("g "):
            groups = frozenset(line.split()[1:])
        elif line.startswith("usemtl "):
            material = line.split(None, 1)[1].strip()
        if not line.startswith("f "):
            records.append(line)
            continue
        poly = []
        for ref in line.split()[1:]:
            fields = ref.split("/")
            vi = int(fields[0]) - 1
            ti = int(fields[1]) - 1 if len(fields) > 1 and fields[1] else None
            poly.append(positions[vi] + (texcoords[ti] if ti is not None else [0.0, 0.0]))
        for piece, camera in assign_polygon(np.array(poly, dtype=float), groups, views, settings):
            refs = []
            vertex_records.append([])
            for row in piece:
                vertex_count += 1
                texcoord_count += 1
                vertex_records[-1].append(len(records))
                records.append("v %.6g %.6g %.6g %.4f %.4f %.4f" % tuple(row[:6]))
                records.append("vt %.5f %.5f" % tuple(row[6:8]))
                refs.append(f"{vertex_count}/{texcoord_count}")
            records.append("f " + " ".join(refs))
            assignment.append(camera)
            areas.append(polygon_area(piece[:, :3]))
            piece_materials.append(material)
            pieces.append(piece)
    assignment = np.array(assignment, dtype=int)
    areas = np.array(areas)
    coverage = areas[assignment >= 0].sum() / max(areas.sum(), 1.0)
    split = dict(pieces=pieces, materials=piece_materials, areas=areas,
                 views={view.index: view for view in views}, vertex_records=vertex_records,
                 obj=obj_path)
    return records, assignment, coverage, split


def material_tints(obj_path, split, assignment, originals, min_share=0.05, limits=(0.3, 2.0)):
    """Per material, the colour the original art shows on its projected
    pieces over the colour the procedural material gives the same pieces.
    Applied to the pieces no camera saw, it keeps them from standing out as
    a different wallpaper or floor."""
    obj_path = Path(obj_path)
    kd, texture_mean, current = {}, {}, None
    mtl = obj_path.with_suffix(".mtl")
    for line in (mtl.read_text().splitlines() if mtl.exists() else []):
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        if words[0] == "newmtl":
            current = words[1]
        elif words[0] == "Kd" and current:
            kd[current] = np.array(words[1:4], dtype=float)
        elif words[0] == "map_Kd" and current:
            image = obj_path.parent / " ".join(words[1:])
            if image.exists():
                texture_mean[current] = np.asarray(Image.open(image).convert("RGB"),
                                                   dtype=float).reshape(-1, 3).mean(axis=0) / 255.0
    images = {}
    for k in split["views"]:
        path = Path(originals) / f"camera{k}-original.png"
        if path.exists():
            images[k] = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    art, procedural, projected_area, total_area = {}, {}, {}, {}
    for piece, name, area, camera in zip(split["pieces"], split["materials"], split["areas"],
                                         assignment):
        total_area[name] = total_area.get(name, 0.0) + area
        if camera < 0 or camera not in images or name is None:
            continue
        view = split["views"][camera]
        xyz = piece[:, :3]
        centre = xyz.mean(axis=0)
        samples = np.vstack([centre[None], centre + (xyz - centre) * 0.7])
        screen = view.screen(view.view(samples))
        sx = np.clip(screen[:, 0].astype(int), 0, 319)
        sy = np.clip(screen[:, 1].astype(int), 0, 239)
        image = images[camera]
        sx = sx * image.shape[1] // 320
        sy = sy * image.shape[0] // 240
        colour = image[sy, sx].mean(axis=0)
        base = kd.get(name, np.ones(3)) * texture_mean.get(name, np.ones(3)) * \
            piece[:, 3:6].mean(axis=0)
        art[name] = art.get(name, 0.0) + colour * area
        procedural[name] = procedural.get(name, 0.0) + base * area
        projected_area[name] = projected_area.get(name, 0.0) + area
    tints = {}
    for name, area in projected_area.items():
        if area < min_share * total_area[name]:
            continue
        # A brightness factor with a bounded hue shift: per-channel ratios of
        # dark colours are noise, and a wood that turns cyan is worse than a
        # wood that is merely too light.
        ratio = art[name] / np.maximum(procedural[name], 1e-6)
        brightness = float(np.clip(ratio.mean(), *limits))
        tints[name] = brightness * np.clip(ratio / max(ratio.mean(), 1e-6), 0.75, 1.25)
    return tints


def box_axes(normal):
    """The two world axes a surface's tiled texture runs along (the runtime
    maps unprojected faces the same way): x/z on floors and ceilings, the
    wall's horizontal axis and up on walls."""
    if abs(normal[1]) >= 0.7071:
        return np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    if abs(normal[0]) > abs(normal[2]):
        return np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
    return np.array([1.0, 0, 0]), np.array([0, -1.0, 0])


def inside_polygon(points, polygon):
    """Even-odd test of (k, 2) points against a (n, 2) polygon."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None], polygon[:, 1][None]
    x1, y1 = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    crosses = ((y0 > y) != (y1 > y)) & \
        (x < x0 + (y - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1e-9))
    return crosses.sum(axis=1) % 2 == 1


def fit_square(xyz, normal, view=None, sizes=(1600, 1200, 900, 700, 500, 350), slack=0.03,
               material_depth=None):
    """The largest box-axis-aligned square (side, origin, corner a, corner b
    in room space) on the plane of polygon `xyz`. Without a view it must lie
    inside the polygon; with one it may span neighbouring faces, as long as
    every sample of it is in that camera's frame and on the surface its depth
    map recorded (the builder cuts walls into ~600-unit faces, too small for
    a sharp exemplar on their own)."""
    a, b = box_axes(normal)
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    if abs(normal @ c) < 0.3:
        return None
    flat = np.stack([xyz @ a, xyz @ b], axis=1)
    offset = normal @ xyz[0]
    area = polygon_area(flat)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    grid = (np.arange(8) + 0.5) / 8
    if view is not None and material_depth is None:
        material_depth = view.depth

    def lift(u, v):
        point = u * a + v * b
        return point + c * (offset - normal @ point) / (normal @ c)

    def on_surface(u, v, side):
        uu, vv = np.meshgrid(u + grid * side, v + grid * side)
        points = np.array([lift(x, y) for x, y in zip(uu.ravel(), vv.ravel())])
        if not view.inside(points):
            return False
        local = view.view(points)
        screen = view.screen(local)
        sx = np.clip(screen[:, 0].astype(int), 0, 319)
        sy = np.clip(screen[:, 1].astype(int), 0, 239)
        # On the surface the camera saw, and that surface is this material
        # (`material_depth`: the camera's depth map of this material alone).
        depth = view.depth[sy, sx]
        own = material_depth[sy, sx]
        if not (np.isfinite(depth).all() and np.isfinite(own).all()):
            return False
        return bool(((np.abs(local[:, 2] - depth) <= depth * slack + 20.0) &
                     (np.abs(own - depth) <= depth * slack + 20.0)).all())

    centre = flat.mean(axis=0)
    for side in sizes:
        if view is not None:
            for du, dv in ((0, 0), (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
                u = centre[0] - side * (0.5 + du)
                v = centre[1] - side * (0.5 + dv)
                if on_surface(u, v, side):
                    return side, lift(u, v), lift(u + side, v), lift(u, v + side)
            continue
        if side * side > area:
            continue
        inset = side * 0.02
        for u in np.arange(lo[0] + inset, hi[0] - side - inset + 1e-6, side / 4):
            for v in np.arange(lo[1] + inset, hi[1] - side - inset + 1e-6, side / 4):
                square = np.array([[u, v], [u + side, v], [u + side, v + side], [u, v + side],
                                   [u + side / 2, v + side / 2]])
                if inside_polygon(square, flat).all():
                    return side, lift(u, v), lift(u + side, v), lift(u, v + side)
    return None


def _weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def material_patches(split, assignment, originals, min_unseen=0.05, candidates=40,
                     max_hue=0.08, max_brightness=2.0, spread=600.0, reach=4000.0,
                     min_wall=1.0e6, max_patches=62):
    """Tiles of the room's own art for the pieces no camera saw.

    Every unseen piece of a material that the cameras also show gets a
    square of the original art (an exemplar) to tile: from its own wall where
    a camera sees that wall squarely and sharply - so the pattern's scale,
    colour and lighting carry on across the edge with the projected part -
    or, for small surfaces and walls no camera shows well, the material's
    best square anywhere. The runtime cuts each square out of the camera's
    background as loaded (HD when a pack provides it).

    Returns (patches, tiles, vertex_colours): the exemplars, the patch index
    of every piece (-1 none), and new colours for the tiled vertices (their
    lighting taken from the nearby projected art)."""
    images = {}
    for k in split["views"]:
        path = Path(originals) / f"camera{k}-original.png"
        if path.exists():
            images[k] = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    pieces, areas, materials = split["pieces"], split["areas"], split["materials"]
    by_material, unseen, total = {}, {}, {}
    for index, (name, area, camera) in enumerate(zip(materials, areas, assignment)):
        if name is None:
            continue
        total[name] = total.get(name, 0.0) + area
        if camera < 0:
            unseen[name] = unseen.get(name, 0.0) + area
        elif camera in images:
            by_material.setdefault(name, []).append(index)
    patches, keys, material_depths = [], {}, {}
    tiles = np.full(len(pieces), -1, dtype=int)
    vertex_colours = {}
    grid = (np.arange(16) + 0.5) / 16
    planes = [_plane(pieces[i][:, :3]) for i in range(len(pieces))]

    def evaluate(i, name, art, min_facing):
        """(score, exemplar) for the best square around projected piece i."""
        view = split["views"][assignment[i]]
        xyz = pieces[i][:, :3]
        normal = planes[i][0]
        ray = xyz.mean(axis=0) - view.origin
        if abs(ray @ normal) / max(np.linalg.norm(ray), 1e-9) < min_facing:
            return None
        key = (view.index, name)
        if key not in material_depths:
            _, _, material_depths[key] = render(split["obj"], view.camera, return_depth=True,
                                                only_materials={name})
        fit = fit_square(xyz, normal, view, material_depth=material_depths[key])
        if fit is None:
            return None
        side, origin, corner_a, corner_b = fit
        screen = view.screen(view.view(np.array([origin, corner_a, corner_b])))
        texels = min(np.linalg.norm(screen[1] - screen[0]), np.linalg.norm(screen[2] - screen[0]))
        # Sample the square: its mean colour, and how much its 4x4 block means
        # vary (a lamp, a painting or a hard shadow inside it would repeat
        # across every tile). Reject a square whose art is not the material:
        # its hue or brightness far from the material's typical look.
        uu, vv = np.meshgrid(grid, grid)
        points = origin + uu.reshape(-1, 1) * (corner_a - origin) + \
            vv.reshape(-1, 1) * (corner_b - origin)
        image = images[assignment[i]]
        sx, sy = _pixels(view, points, image)
        colours = image[sy, sx].reshape(16, 16, 3)
        blocks = colours.reshape(4, 4, 4, 4, 3).mean(axis=(1, 3)).mean(axis=2)
        variation = blocks.std() / max(blocks.mean(), 1e-3)
        mean = colours.reshape(-1, 3).mean(axis=0)
        hue = np.abs(mean / max(mean.sum(), 1e-6) - art / max(art.sum(), 1e-6)).sum()
        brightness = mean.sum() / max(art.sum(), 1e-6)
        if hue > max_hue or not (1 / max_brightness < brightness < max_brightness):
            return None
        score = texels * np.sqrt(side / 1000.0) / (1.0 + 4.0 * variation) / (1.0 + 10.0 * hue)
        return score, dict(material=name, camera=int(assignment[i]), side=float(side),
                           origin=origin, a=corner_a, b=corner_b, exemplar=mean)

    def best_of(indices, name, art, min_facing):
        found = [r for r in (evaluate(i, name, art, min_facing) for i in indices) if r is not None]
        return max(found, key=lambda r: r[0])[1] if found else None

    def add(exemplar):
        key = (exemplar["camera"], tuple(np.round(exemplar["origin"], 1)), exemplar["side"])
        if key not in keys:
            if len(patches) >= max_patches:
                return -1
            keys[key] = len(patches)
            patches.append(exemplar)
        return keys[key]

    for name, indices in by_material.items():
        if unseen.get(name, 0.0) < min_unseen * total[name]:
            continue
        piece_colours = np.array([_mean_colour(pieces[i], split["views"][assignment[i]],
                                               images[assignment[i]]) for i in indices])
        centres = np.array([pieces[i][:, :3].mean(axis=0) for i in indices])
        weights = np.array([areas[i] for i in indices])
        # The material's typical look: the area-weighted median of its
        # projected pieces, so the few pieces where the reading and the art
        # disagree (a door painted where the mesh has wallpaper) do not count.
        art = np.array([_weighted_median(piece_colours[:, ch], weights) for ch in range(3)])
        fallback = best_of(sorted(indices, key=lambda i: -areas[i])[:candidates * 2], name, art, 0.5)
        tiled = [i for i, (m, c) in enumerate(zip(materials, assignment)) if m == name and c < 0]
        walls = {}
        for i in tiled:
            walls.setdefault(_plane_key(planes[i]), []).append(i)
        for wall in sorted(walls.values(), key=lambda w: -sum(areas[i] for i in w)):
            exemplar = None
            if sum(areas[i] for i in wall) >= min_wall:
                # From the wall itself, even seen at a slant (as the projection
                # beside it is): matching that part matters more than a
                # square-on view.
                same = [i for i in indices if _same_plane(planes[i], planes[wall[0]])]
                exemplar = best_of(sorted(same, key=lambda i: -areas[i])[:12], name, art, 0.2)
            exemplar = exemplar or fallback
            if exemplar is None:
                continue
            index = add(exemplar)
            if index < 0 and fallback is not None:
                index = add(fallback)
            if index < 0:
                continue
            # Light the tiled faces from the art itself: each vertex takes the
            # colour of the nearby projected pieces of this material (inverse
            # square distance), fading to the material's typical colour
            # further away, over the exemplar's own colour.
            records = [r for i in wall for r in split["vertex_records"][i]]
            points = np.vstack([pieces[i][:, :3] for i in wall])
            light = np.zeros((len(points), 3))
            for start in range(0, len(points), 2048):
                chunk = points[start:start + 2048]
                d2 = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
                w = 1.0 / (d2 + spread ** 2)
                w0 = 1.0 / reach ** 2
                light[start:start + 2048] = (w @ piece_colours + w0 * art) / (w.sum(axis=1) + w0)[:, None]
            factors = np.clip(light / np.maximum(patches[index]["exemplar"], 1e-3), 0.0, 8.0)
            for i in wall:
                tiles[i] = index
            for record, factor in zip(records, factors):
                vertex_colours[record] = (index, factor)
    # Vertex colours must stay in 0..1: each patch carries the scale.
    for index, patch in enumerate(patches):
        mine = [f for (k, f) in vertex_colours.values() if k == index]
        patch["tint"] = np.maximum(np.max(mine, axis=0), 1e-3) if mine else np.ones(3)
    colours = {r: f / patches[k]["tint"] for r, (k, f) in vertex_colours.items()}
    return patches, tiles, colours


def _plane(xyz):
    normal = face_normal(xyz)
    return normal, float(normal @ xyz.mean(axis=0))


def _plane_key(plane):
    normal, offset = plane
    return tuple(np.round(normal * 8).astype(int)) + (int(round(offset / 150.0)),)


def _same_plane(p, q, angle=0.98, distance=150.0):
    return float(p[0] @ q[0]) >= angle and abs(p[1] - q[1]) <= distance


def _pixels(view, points, image):
    screen = view.screen(view.view(points))
    sx = np.clip((screen[:, 0] * image.shape[1] / 320).astype(int), 0, image.shape[1] - 1)
    sy = np.clip((screen[:, 1] * image.shape[0] / 240).astype(int), 0, image.shape[0] - 1)
    return sx, sy


def _mean_colour(piece, view, image):
    xyz = piece[:, :3]
    centre = xyz.mean(axis=0)
    sx, sy = _pixels(view, np.vstack([centre[None], centre + (xyz - centre) * 0.7]), image)
    return image[sy, sx].mean(axis=0)


TILE_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def write_proj(path, assignment, tints=None, patches=None, tiles=None):
    text = "".join(DIGITS[c] if 0 <= c < len(DIGITS) else "-" for c in assignment)
    lines = [f"proj1 {len(assignment)}", text]
    for name, tint in sorted((tints or {}).items()):
        lines.append("tint %s %.3f %.3f %.3f" % (name, *tint))
    for index, patch in enumerate(patches or []):
        lines.append("patch %d %s %d %.0f %s %s %s %.3f %.3f %.3f" % (
            index, patch["material"], patch["camera"], patch["side"],
            *(" ".join("%.1f" % x for x in patch[key]) for key in ("origin", "a", "b")),
            *patch["tint"]))
    if patches:
        lines.append("tiles " + "".join(TILE_DIGITS[t] if t >= 0 else "-" for t in tiles))
    Path(path).write_text("\n".join(lines) + "\n")


def preview_obj(obj_path, rdt_path, assignment, originals, out_dir):
    """A copy of the mesh whose assigned faces carry projected UVs into the
    original backgrounds, for checking with room3d_preview from any camera."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices = read_vertices(obj_path)
    cameras = {c["index"]: c for c in survey(Path(rdt_path))["cameras"]}
    screens = {k: project(vertices, cam)[1] for k, cam in cameras.items()}
    lines = Path(obj_path).read_text().splitlines()
    name = Path(obj_path).stem
    mtl_src = Path(obj_path).with_suffix(".mtl")
    out_lines, face_index, vt_count, material, switched = [], 0, 0, None, False
    for line in lines:
        if line.startswith("usemtl "):
            material, switched = line, False
        if line.startswith("mtllib "):
            out_lines.append(f"mtllib {name}_proj.mtl")
            continue
        if line.startswith("vt "):
            vt_count += 1
        elif line.startswith("f "):
            if switched and material:
                out_lines.append(material)
                switched = False
            cam = assignment[face_index]
            face_index += 1
            if cam >= 0:
                refs = []
                for ref in line.split()[1:]:
                    vi = int(ref.split("/")[0]) - 1
                    sx, sy = screens[cam][vi]
                    vt_count += 1
                    out_lines.append(f"vt {sx / 320.0:.5f} {1.0 - sy / 240.0:.5f}")
                    refs.append(f"{vi + 1}/{vt_count}")
                out_lines.append(f"usemtl proj{cam}")
                out_lines.append("f " + " ".join(refs))
                switched = True
                continue
        out_lines.append(line)
    (out_dir / f"{name}_proj.obj").write_text("\n".join(out_lines) + "\n")
    mtl = mtl_src.read_text() if mtl_src.exists() else ""
    for k in sorted(set(int(c) for c in assignment if c >= 0)):
        src = Path(originals) / f"camera{k}-original.png"
        Image.open(src).convert("RGB").save(out_dir / f"proj{k}.png")
        mtl += f"\nnewmtl proj{k}\nKd 1 1 1\nmap_Kd proj{k}.png\n"
    (out_dir / f"{name}_proj.mtl").write_text(mtl)
    for texture in Path(obj_path).parent.glob("*.png"):
        target = out_dir / texture.name
        if not target.exists() and not texture.name.startswith("preview"):
            target.write_bytes(texture.read_bytes())
    sidecar = Path(obj_path).with_suffix(".cameras")
    if sidecar.exists():
        (out_dir / f"{name}_proj.cameras").write_text(sidecar.read_text())
    return out_dir / f"{name}_proj.obj"


def tint_mtl(path, tints):
    """Multiply the tinted materials' Kd in an MTL (preview only; the game
    reads the tints from the .proj)."""
    if not path.exists():
        return
    out, current, seen_kd = [], None, set()
    def close():
        if current in tints and current not in seen_kd:
            out.append("Kd %.3f %.3f %.3f" % tuple(tints[current]))
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "newmtl":
            close()
            current = words[1]
        elif words and words[0] == "Kd" and current in tints:
            kd = np.array(words[1:4], dtype=float) * tints[current]
            line = "Kd %.3f %.3f %.3f" % tuple(kd)
            seen_kd.add(current)
        out.append(line)
    close()
    path.write_text("\n".join(out) + "\n")


def split_and_assign(obj, rdt, originals=None, skip=(), min_density=0.000015):
    """Split the builder's mesh (kept as roomSRRV.base.obj, so a rerun starts
    from it again) and assign cameras. Returns (OBJ lines, assignment, tints, (patches, tiles))."""
    obj = Path(obj)
    base = obj.with_suffix(".base.obj")
    if not obj.read_text().startswith(SPLIT_MARKER):
        base.write_bytes(obj.read_bytes())
    work = obj.with_name(obj.stem + ".work.obj")
    work.write_bytes(base.read_bytes())
    if obj.with_suffix(".cameras").exists():
        work.with_suffix(".cameras").write_bytes(obj.with_suffix(".cameras").read_bytes())
    try:
        records, assignment, coverage, split = assign(work, rdt, skip, min_density)
        tints = material_tints(work, split, assignment, originals) if originals else {}
        patches, tiles, colours = (material_patches(split, assignment, originals) if originals
                                   else ([], None, {}))
        for index, colour in colours.items():
            words = records[index].split()
            records[index] = "v %s %s %s %.4f %.4f %.4f" % (*words[1:4], *colour)
    finally:
        for suffix in (".obj", ".cameras"):
            work.with_suffix(suffix).unlink(missing_ok=True)
    covered = int((assignment >= 0).sum())
    print(f"{obj.name}: {covered}/{len(assignment)} pieces projected, "
          f"{coverage:.0%} of the surface", flush=True)
    return records, assignment, tints, (patches, tiles)


def project_room(obj, rdt, originals=None, skip=(), min_density=0.000015):
    """Replace the built mesh with its split version and write the .proj."""
    obj = Path(obj)
    records, assignment, tints, patches = split_and_assign(obj, rdt, originals, skip, min_density)
    obj.write_text("\n".join([SPLIT_MARKER] + records) + "\n")
    write_proj(obj.with_suffix(".proj"), assignment, tints, *patches)
    return obj.with_suffix(".proj")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("assign", "preview"):
        p = sub.add_parser(name)
        p.add_argument("obj", type=Path)
        p.add_argument("--rdt", type=Path, required=True)
        p.add_argument("--skip", type=int, nargs="*", default=[])
        p.add_argument("--min-density", type=float, default=0.000015)
        p.add_argument("--originals", type=Path, required=(name == "preview"),
                       help="camera*-original.png of the room: material tints for unseen faces")
        if name == "preview":
            p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assign":
        project_room(args.obj, args.rdt, args.originals, args.skip, args.min_density)
        return
    records, assignment, tints, _ = split_and_assign(args.obj, args.rdt, args.originals,
                                                  args.skip, args.min_density)
    out = args.out / args.obj.name
    args.out.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(records) + "\n")
    for suffix in (".mtl", ".cameras"):
        if args.obj.with_suffix(suffix).exists():
            out.with_suffix(suffix).write_bytes(args.obj.with_suffix(suffix).read_bytes())
    tint_mtl(out.with_suffix(".mtl"), tints)
    for texture in args.obj.parent.glob("*.png"):
        if not texture.name.startswith("preview"):
            (args.out / texture.name).write_bytes(texture.read_bytes())
    print(preview_obj(out, args.rdt, assignment, args.originals, args.out / "proj"))


if __name__ == "__main__":
    main()
