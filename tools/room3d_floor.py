"""Convex visual floor openings; no changes to original gameplay collision data."""
import math

import numpy as np


def area(polygon):
    p = np.asarray(polygon, float)
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def validated_polygon(points):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('Expected a finite convex XZ polygon')
    if area(p) < 0:
        p = p[::-1]
    edges = np.roll(p, -1, axis=0) - p
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    offsets = p[None, :, :] - p[:, None, :]
    side = edges[:, None, 0] * offsets[:, :, 1] - edges[:, None, 1] * offsets[:, :, 0]
    if np.any(side < -1e-8) or area(p) <= 1e-8 or np.any(np.linalg.norm(edges, axis=1) <= 1e-8) or np.any(turns < -1e-8):
        raise ValueError('Floor openings must be nondegenerate convex polygons')
    return p


def clip_halfplane(polygon, a, b, inside):
    direction = b - a
    relative = polygon - a
    distances = direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
    if not inside:
        distances = -distances
    output = []
    for i in range(len(polygon)):
        previous = (i - 1) % len(polygon)
        pa, pb = polygon[previous], polygon[i]
        da, db = distances[previous], distances[i]
        if (da >= 0) != (db >= 0):
            output.append(pa + (pb - pa) * (da / (da - db)))
        if db >= 0:
            output.append(pb)
    if len(output) < 3:
        return np.empty((0, 2))
    result = np.asarray(output)
    keep = np.linalg.norm(result - np.roll(result, 1, axis=0), axis=1) > 1e-8
    result = result[keep]
    return result if len(result) >= 3 and abs(area(result)) > 1e-8 else np.empty((0, 2))


def subtract_convex(polygon, hole):
    remainder = polygon
    result = []
    for a, b in zip(hole, np.roll(hole, -1, axis=0), strict=True):
        outside = clip_halfplane(remainder, a, b, False)
        if len(outside):
            result.append(outside)
        remainder = clip_halfplane(remainder, a, b, True)
        if not len(remainder):
            break
    return result


def floor_patches(bounds, openings, tile_size=1000):
    """Return tiled convex floor patches outside the union of explicit openings."""
    x0, z0, x1, z1 = map(float, bounds)
    if not np.isfinite([x0, z0, x1, z1, tile_size]).all() or x1 <= x0 or z1 <= z0 or tile_size <= 0:
        raise ValueError('Invalid floor bounds or tile size')
    holes = [validated_polygon(opening['polygon']) for opening in openings]
    nx, nz = math.ceil((x1 - x0) / tile_size), math.ceil((z1 - z0) / tile_size)
    if nx * nz > 65536 or len(holes) > 64:
        raise ValueError('Excessive visual floor subdivision')
    xs, zs = np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1)
    result = []
    for i in range(nx):
        for j in range(nz):
            fragments = [np.array([[xs[i], zs[j]], [xs[i+1], zs[j]],
                                   [xs[i+1], zs[j+1]], [xs[i], zs[j+1]]])]
            for hole in holes:
                fragments = [part for polygon in fragments for part in subtract_convex(polygon, hole)]
                if not fragments:
                    break
            # Existing floor winding has a positive Y normal (negative XZ area).
            result.extend(polygon[::-1].tolist() for polygon in fragments)
    return result
