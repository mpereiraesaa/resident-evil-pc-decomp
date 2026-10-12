"""Bounded architectural alternatives for explicitly approximate stair masks."""

import math

import cv2
import numpy as np
from room3d_assemblies import stair_proportions
from room3d_preview import camera_basis
from room3d_shapes import generate
from room3d_stair_edges import normalize_horizontal_axes


def mask_volume_slices(room, group, floor_y=0, resolution=96):
    """Intersect training mask frusta inside room bounds; no model depth used.

    Three-pixel dilation tolerates uncertain mask boundaries. Slices are visual
    volume hypotheses conditioned on a floor plane, not recovered solid geometry.
    """
    if not math.isfinite(floor_y) or not isinstance(resolution, int) or not 2 <= resolution <= 128:
        raise ValueError('Expected finite floor and grid resolution within 2..128')
    if len(group) < 2 or not room.get('boundaries'):
        return []
    bounds = np.array([[b['xmin'], b['xmax'], b['zmin'], b['zmax']] for b in room['boundaries']])
    if not np.isfinite(bounds).all():
        raise ValueError('Room bounds must be finite')
    xs = np.linspace(bounds[:, 0].min(), bounds[:, 1].max(), resolution)
    zs = np.linspace(bounds[:, 2].min(), bounds[:, 3].max(), resolution)
    xx, zz = np.meshgrid(xs, zs)
    supports = [(o['camera_model'], cv2.dilate(o['mask'].astype('uint8'), np.ones((7, 7), 'uint8')))
                for o in group]
    slices = []
    for rise in range(0, 6001, 500):
        points = np.column_stack([xx.ravel(), np.full(xx.size, floor_y-rise), zz.ravel()])
        keep = np.ones(len(points), bool)
        for camera, mask in supports:
            view = (points-camera['origin']) @ camera_basis(camera).T
            xy = view[:, :2]/np.maximum(view[:, 2:], 1e-6)*camera['fov'] + [160, 120]
            ij = np.rint(xy).astype(int)
            inside = ((view[:, 2] > 1) & (ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[1])
                      & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[0]))
            hits = np.zeros(len(points), bool)
            hits[inside] = mask[ij[inside, 1], ij[inside, 0]].astype(bool)
            keep &= hits
        slices.append((rise, points[keep]))
    return slices


def mask_volume_starts(room, group, axes, floor_y=0):
    """Generate finite stair flights from shared floor/upper mask support."""
    slices = mask_volume_slices(room, group, floor_y)
    if not slices or len(slices[0][1]) < 8:
        return
    floor = slices[0][1][:, [0, 2]]
    axes = normalize_horizontal_axes(axes)
    for rise, points in slices[2:]:
        if len(points) < 8:
            continue
        top = points[:, [0, 2]]
        for width_axis in axes:
            lo = max(np.percentile(floor @ width_axis, 5), np.percentile(top @ width_axis, 5))
            hi = min(np.percentile(floor @ width_axis, 95), np.percentile(top @ width_axis, 95))
            if hi-lo < 400:
                continue
            cross = (lo+hi)/2
            for sign in (-1, 1):
                direction = sign*np.array([-width_axis[1], width_axis[0]])
                foot_target = np.percentile(floor @ direction, 2)*direction + cross*width_axis
                top_target = np.percentile(top @ direction, 98)*direction + cross*width_axis
                foot = floor[np.argmin(np.linalg.norm(floor-foot_target, axis=1))]
                end = top[np.argmin(np.linalg.norm(top-top_target, axis=1))]
                run = float((end-foot) @ direction)
                if run <= 0:
                    continue
                yaw = math.atan2(direction[0], direction[1])
                for width in sorted({min(4000., hi-lo), min(2000., hi-lo)}):
                    candidate = generate('staircase', {
                        'width': width, 'run': run, 'rise': rise,
                        'steps': int(np.clip(round(rise/180), 2, 64)),
                        'landing': 0, 'structure': 'suspended', 'slab_depth': 100,
                    }, [foot[0], floor_y, foot[1]], yaw)
                    if not replacement_valid(candidate, axes):
                        continue
                    candidate['initialization'] = f'creative_mask_volume_{rise}'
                    candidate['interpretation'] = {
                        'source': 'approximate_mask_layout_replacement',
                        'qualification': 'Estimated training-mask volume and floor prior; no independent geometry or traversal validation',
                    }
                    yield candidate


def replacement_valid(stairs, axes, maximum_width=4000):
    """An explicit design prior, never a claim about original dimensions."""
    shape = stairs['shape']
    width_axis = np.array([math.cos(shape['yaw']), -math.sin(shape['yaw'])])
    alignment = np.max(np.abs(normalize_horizontal_axes(axes) @ width_axis))
    return bool(shape['parameters']['width'] <= maximum_width
                and alignment >= math.cos(math.radians(12))
                and stair_proportions(stairs)['plausible'])


def replacement_starts(starts, axes, maximum_width=4000):
    """Keep seed centers/base heights, explore both ascent signs on layout axes.

    At most eight input hypotheses and two widths per axis/sign are considered.
    Hidden structure, orientation and endpoint contacts remain estimated.
    Callers must still evaluate all ordinary support/depth/heldout gates.
    """
    axes = normalize_horizontal_axes(axes)
    if not math.isfinite(maximum_width) or maximum_width <= 0:
        raise ValueError('Maximum creative width must be positive and finite')
    seen = set()
    for index, seed in enumerate(starts[:8]):
        shape = seed['shape']
        p = shape['parameters']
        origin = np.asarray(shape['origin'], dtype=float)
        center = origin + p['run']/2 * np.array([math.sin(shape['yaw']), 0, math.cos(shape['yaw'])])
        steps = int(np.clip(round(p['rise']/180), 2, 64))
        run = max(p['run'], p['rise']/math.tan(math.radians(40)) + p.get('landing', 0),
                  steps*200 + p.get('landing', 0))
        for axis in axes:
            for sign in (-1, 1):
                yaw = math.atan2(-axis[1]*sign, axis[0]*sign)
                for width in sorted({min(p['width'], maximum_width), min(p['width'], 2000, maximum_width)}):
                    params = {**p, 'width': width, 'run': run, 'steps': steps,
                              'structure': 'suspended', 'slab_depth': 100}
                    base = center - run/2 * np.array([math.sin(yaw), 0, math.cos(yaw)])
                    key = (tuple(np.round(base, 6)), yaw, width, run, p['rise'], p.get('landing', 0))
                    if key in seen:
                        continue
                    seen.add(key)
                    obj = generate('staircase', params, base, yaw)
                    obj['initialization'] = f'creative_layout_{index}'
                    obj['interpretation'] = {
                        'source': 'approximate_mask_layout_replacement',
                        'qualification': 'Creative estimated flight; not recovered original geometry or verified traversal',
                    }
                    yield obj
