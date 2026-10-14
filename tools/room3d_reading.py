#!/usr/bin/env python3
"""Build a Room3D mesh from a human-written room reading.

A reading (room3d/readings/ROOMxxxx.toml) records what a person read from the
evidence for one room: walkable zones with floor and ceiling heights, wall
finishes, doors, windows, galleries, stairs, arcades, furniture, lights and
per-camera hides. The shell geometry is derived from those zones; everything
else is a named kind from the library below, rendered through
tools/room3d_handmade.py with procedural textures and baked vertex lighting.

    room3d_reading.py survey ROOM1180.RDT --originals DIR   # evidence dump
    room3d_reading.py build room3d/readings/ROOM1180.toml --rdt ... --out DIR

Coordinates are RE1 room units with negative Y up.
"""
import argparse
import math
import struct
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room3d_handmade import (  # noqa: E402
    Mesh,
    balustrade,
    compare_sheet,
    door,
    edge_agreement,
    preview,
    tint,
    to_image,
    value_noise,
    write_mtl,
)
from survey_room3d import survey  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIDES = {"west": (1, 0, 0), "east": (-1, 0, 0), "south": (0, 0, 1), "north": (0, 0, -1)}


# ---------------------------------------------------------------- materials --
# Procedural texture generators keyed by material family. A reading picks a
# family per surface and may tint it; the texture is generated per room.
def _grain(rng, size):
    yy, _ = np.mgrid[0:size, 0:size] / size
    warp = value_noise((size, size), rng, octaves=3, base_freq=3)
    grain = 0.5 + 0.5 * np.sin((yy * 36 + warp * 5) * math.pi)
    streak = value_noise((size, size), rng, octaves=2, base_freq=48, persistence=0.4)
    return grain, streak


def tex_plaster(rng, size, color, stains=0.0):
    yy, xx = np.mgrid[0:size, 0:size] / size
    mottle = value_noise((size, size), rng, octaves=4, base_freq=3)
    fine = value_noise((size, size), rng, octaves=2, base_freq=64, persistence=0.5)
    level = 0.82 + 0.20 * mottle + 0.07 * fine
    if stains:
        level -= stains * np.clip(value_noise((size, size), rng, octaves=3, base_freq=2) - 0.55, 0, 1) * 2.0
    return np.array(color)[None, None, :] * level[..., None]


def tex_wood(rng, size, color):
    grain, streak = _grain(rng, size)
    level = 0.70 + 0.30 * grain + 0.20 * streak
    return np.array(color)[None, None, :] * level[..., None]


def tex_planks(rng, size, color, count=6, gap=(0.07, 0.05, 0.03)):
    yy, _ = np.mgrid[0:size, 0:size] / size
    grain, streak = _grain(rng, size)
    boards = np.floor(yy * count)
    shade = (np.sin(boards * 12.9898 + 7.1) * 43758.5453) % 1
    level = 0.75 + 0.30 * shade + 0.15 * grain + 0.12 * streak
    image = np.array(color)[None, None, :] * level[..., None]
    return tint(((yy * count) % 1 < 0.02).astype(float), image, gap)


def tex_panel(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    wood = tex_wood(rng, size, color)
    frame, bevel = 0.08, 0.03
    inner = (xx > frame) & (xx < 1 - frame) & (yy > frame) & (yy < 1 - frame)
    field = (xx > frame + bevel) & (xx < 1 - frame - bevel) & (yy > frame + bevel) & (yy < 1 - frame - bevel)
    light = inner & ~field & ((yy < frame + bevel) | (xx < frame + bevel))
    panel = tint(((inner & ~field) & ~light).astype(float) * 0.5, wood, (0.05, 0.03, 0.02))
    panel = tint(light.astype(float) * 0.45, panel, tuple(min(1, c * 1.8) for c in color))
    return tint(field.astype(float) * 0.25, panel, tuple(c * 0.5 for c in color))


def tex_marble(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    veins = value_noise((size, size), rng, octaves=5, base_freq=3)
    lines = np.abs(np.sin((xx * 3 + yy * 2 + veins * 4) * math.pi * 1.5))
    level = 0.92 + 0.10 * veins - 0.22 * np.clip(1 - lines * 6, 0, 1)
    image = np.array(color)[None, None, :] * level[..., None]
    grout = ((xx * 2) % 1 < 0.012) | ((yy * 2) % 1 < 0.012)
    return tint(grout.astype(float) * 0.8, image, (0.35, 0.33, 0.30))


def tex_stone(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    mottle = value_noise((size, size), rng, octaves=4, base_freq=4)
    image = np.array(color)[None, None, :] * (0.88 + 0.16 * mottle)[..., None]
    joints = ((yy * 3) % 1 < 0.01) | (((xx + 0.5 * np.floor(yy * 3)) * 2) % 1 < 0.008)
    return tint(joints.astype(float) * 0.6, image, (0.30, 0.27, 0.22))


def tex_parquet(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    columns, plank_w = 6, 1 / 12
    col = np.floor(xx * columns).astype(int)
    band = np.where(col % 2 == 0, np.floor((xx + yy) / plank_w), np.floor((xx - yy + 2) / plank_w))
    shade = (np.sin((band * 31 + col * 17) * 12.9898) * 43758.5453) % 1
    grain = value_noise((size, size), rng, octaves=3, base_freq=32, persistence=0.6)
    edge = np.where(col % 2 == 0, ((xx + yy) % plank_w) < 0.004, ((xx - yy + 2) % plank_w) < 0.004) | ((xx * columns) % 1 < 0.012)
    image = np.array(color)[None, None, :] * (0.72 + 0.36 * shade + 0.16 * grain)[..., None]
    return tint(edge.astype(float) * 0.8, image, (0.10, 0.07, 0.04))


def tex_carpet(rng, size, color, motif=(0.78, 0.62, 0.30)):
    yy, xx = np.mgrid[0:size, 0:size] / size
    weave = value_noise((size, size), rng, octaves=3, base_freq=48, persistence=0.5)
    carpet = np.array(color)[None, None, :] * (0.85 + 0.25 * weave)[..., None]
    r = np.hypot(xx - 0.5, yy - 0.5)
    ang = np.arctan2(yy - 0.5, xx - 0.5)
    petal = 0.5 + 0.5 * np.cos(ang * 8)
    medallion = (r < 0.16 + 0.07 * petal) & (r > 0.06 + 0.04 * petal)
    corners = np.zeros_like(r, dtype=bool)
    for cxr, cyr in ((0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)):
        corners |= np.hypot(xx - cxr, yy - cyr) < 0.06
    carpet = tint(medallion.astype(float) * 0.9, carpet, motif)
    carpet = tint((r < 0.05).astype(float), carpet, (0.85, 0.78, 0.60))
    carpet = tint((np.abs(r - 0.30) < 0.012).astype(float) * 0.8, carpet, (0.85, 0.78, 0.60))
    return tint(corners.astype(float) * 0.8, carpet, motif)


def tex_runner(rng, size, color, motif=(0.78, 0.62, 0.30)):
    yy, xx = np.mgrid[0:size, 0:size] / size
    weave = value_noise((size, size), rng, octaves=3, base_freq=48, persistence=0.5)
    runner = np.array(color)[None, None, :] * (0.85 + 0.25 * weave)[..., None]
    edge = (np.abs(xx - 0.08) < 0.015) | (np.abs(xx - 0.92) < 0.015)
    scroll = (np.abs(xx - 0.5) < 0.06) & (np.abs(np.sin(yy * math.pi * 6)) > 0.8)
    return tint((edge | scroll).astype(float) * 0.85, runner, motif)


def tex_border(rng, size, color, motif=(0.78, 0.62, 0.30)):
    by, bx = np.mgrid[0:size // 2, 0:size] / np.array([size // 2, size])[:, None, None]
    weave = value_noise((size // 2, size), rng, octaves=3, base_freq=48, persistence=0.5)
    border = np.array(color)[None, None, :] * (0.85 + 0.25 * weave)[..., None]
    return tint(((np.abs(by - 0.25) < 0.03) | (np.abs(by - 0.75) < 0.03)).astype(float), border, motif)


def tex_coffer(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    mottle = value_noise((size, size), rng, octaves=3, base_freq=3)
    coffer = np.array(color)[None, None, :] * (0.85 + 0.2 * mottle)[..., None]
    grid = ((xx * 2) % 1 < 0.08) | ((yy * 2) % 1 < 0.08) | ((xx * 2) % 1 > 0.92) | ((yy * 2) % 1 > 0.92)
    coffer = tint(grid.astype(float), coffer, tuple(c * 0.55 for c in color))
    inset = (((xx * 2) % 1 > 0.2) & ((xx * 2) % 1 < 0.8) & ((yy * 2) % 1 > 0.2) & ((yy * 2) % 1 < 0.8))
    return tint(inset.astype(float) * 0.5, coffer, tuple(min(1, c * 1.6) for c in color))


def tex_door(rng, size, color, rows=2, knob=True):
    ground = tuple(int(c * 255) for c in color)
    line = tuple(int(c * 0.55 * 255) for c in color)
    dark = tuple(int(c * 0.38 * 255) for c in color)
    image = Image.new("RGB", (256, 512), ground)
    draw = ImageDraw.Draw(image)
    cell = 460 // rows
    for row in range(rows):
        for col in range(2):
            x0, y0 = 22 + col * 112, 24 + row * cell
            draw.rectangle((x0, y0, x0 + 90, y0 + cell - 24), outline=line, width=4)
            draw.rectangle((x0 + 8, y0 + 8, x0 + 82, y0 + cell - 32), outline=dark, width=2)
    draw.rectangle((0, 0, 255, 511), outline=dark, width=3)
    if knob:
        draw.ellipse((214, 262, 232, 280), fill=(170, 140, 70))
    return np.asarray(image).astype(float) / 255


def tex_double_door(rng, size, color):
    ground = tuple(int(c * 255) for c in color)
    image = Image.new("RGB", (512, 512), ground)
    draw = ImageDraw.Draw(image)
    dark = tuple(int(c * 0.45 * 255) for c in color)
    light = tuple(min(255, int(c * 1.4 * 255)) for c in color)
    for leaf in range(2):
        lx = leaf * 256
        for row in range(3):
            y0 = 20 + row * 160
            draw.rectangle((lx + 24, y0, lx + 232, y0 + 140), outline=dark, width=5)
            draw.rectangle((lx + 36, y0 + 12, lx + 220, y0 + 128), outline=light, width=2)
        draw.line((lx, 0, lx, 511), fill=(20, 12, 6), width=4)
    draw.ellipse((236, 250, 252, 266), fill=(170, 140, 70))
    draw.ellipse((260, 250, 276, 266), fill=(170, 140, 70))
    return np.asarray(image).astype(float) / 255


def tex_painting(rng, size, color):
    w, h = 512, 384
    py, px = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    sky = np.stack([0.22 + 0.25 * (1 - py), 0.26 + 0.22 * (1 - py), 0.36 + 0.18 * (1 - py)], axis=-1)
    clouds = value_noise((h, w), rng, octaves=4, base_freq=3)
    sky = tint(np.clip(clouds - 0.45, 0, 1) * 1.6 * (py < 0.55), sky, (0.55, 0.52, 0.50))
    ridge = 0.48 + 0.12 * value_noise((1, w), rng, octaves=3, base_freq=2)[0]
    hills = py > ridge[None, :]
    land = np.array([0.12, 0.14, 0.10]) + 0.08 * value_noise((h, w), rng, octaves=4, base_freq=6)[..., None]
    scene = np.where(hills[..., None], land, sky)
    frame_w = 0.08
    frame_mask = (px < frame_w) | (px > 1 - frame_w) | (py < frame_w * w / h) | (py > 1 - frame_w * w / h)
    gilt = np.array(color)[None, None, :] * (0.85 + 0.3 * value_noise((h, w), rng, octaves=2, base_freq=48))[..., None]
    scene = np.where(frame_mask[..., None], gilt, scene)
    return scene


def tex_fanlight(rng, size, color):
    """Semicircular stained glass with the chord at the bottom; plaster outside."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    ang = np.arctan2(1.0 - yy, xx - 0.5)
    r = np.hypot(xx - 0.5, 1.0 - yy)
    hue = (np.floor((ang + math.pi) / (math.pi / 8)) + np.floor(r * 4)) % 4
    palette = np.array([(0.55, 0.70, 1.0), (0.95, 0.80, 0.45), (0.80, 0.40, 0.45), (0.50, 0.80, 0.65)])
    glass = palette[hue.astype(int)] * (0.75 + 0.25 * value_noise((size, size), rng, octaves=2, base_freq=16))[..., None]
    tracery = (np.abs(np.sin(ang * 8)) < 0.08) | (np.abs((r * 4) % 1) < 0.05) | (r > 0.96)
    plaster = tex_plaster(rng, size, color)
    rose = np.where((r <= 1)[..., None], glass, plaster)
    return tint((tracery & (r <= 1)).astype(float), rose, (0.10, 0.08, 0.06))


def tex_glass(rng, size, color):
    gy, gx = np.mgrid[0:512, 0:256] / np.array([512, 256])[:, None, None]
    lattice = ((gx * 6) % 1 < 0.06) | ((gy * 12) % 1 < 0.04)
    hue = (np.floor(gx * 6) + np.floor(gy * 12)) % 3
    pal = np.array([(0.80, 0.85, 1.0), (0.95, 0.90, 0.70), (0.70, 0.85, 0.80)])
    return tint(lattice.astype(float), pal[hue.astype(int)] * 0.85, (0.12, 0.10, 0.08))


def tex_window_fan(rng, size, color):
    """Round-arched window with radial mullions in a plaster surround (10B style)."""
    w, h = 512, 256
    py, px = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    r = np.hypot((px - 0.5) * 2, (py - 1.0))
    glow = np.stack([0.70 + 0.25 * (1 - r), 0.80 + 0.18 * (1 - r), 0.98 - 0.02 * r], axis=-1)
    plaster = tex_plaster(rng, size, color)[:h, :w]
    window = np.where((r < 0.9)[..., None], glow, plaster)
    angle = np.arctan2(1.0 - py, (px - 0.5) * 2)
    mullion = np.zeros_like(r, dtype=bool)
    for k in range(1, 6):
        mullion |= np.abs(angle - k * math.pi / 6) < 0.025
    mullion |= np.abs(r - 0.55) < 0.015
    mullion |= (r > 0.86) & (r < 0.9)
    window = tint((mullion & (r < 0.9)).astype(float), window, (0.12, 0.08, 0.05))
    return tint(((r >= 0.9) & (r < 0.98)).astype(float), window, (0.22, 0.14, 0.08))


def tex_chest(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    chest = tex_planks(rng, size, color, count=4, gap=(0.04, 0.03, 0.02))
    straps = ((xx > 0.10) & (xx < 0.16)) | ((xx > 0.84) & (xx < 0.90)) | (yy < 0.06) | (yy > 0.94)
    return tint(straps.astype(float), chest, (0.10, 0.10, 0.11))


def tex_crate(rng, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    crate = tex_planks(rng, size, color, count=6, gap=(0.10, 0.07, 0.04))
    frame = (xx < 0.08) | (xx > 0.92) | (yy < 0.08) | (yy > 0.92)
    return tint(frame.astype(float) * 0.8, crate, tuple(min(1, c * 1.5) for c in color))


def tex_wallpaper(rng, size, color, motif=(0.30, 0.28, 0.20), repeats=8):
    """Damask-style wallpaper: a lattice of lobed motifs on a toned ground."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    grain = value_noise((size, size), rng, octaves=3, base_freq=24, persistence=0.5)
    paper = np.array(color)[None, None, :] * (0.88 + 0.18 * grain)[..., None]
    u, v = (xx * repeats) % 1, (yy * repeats) % 1
    lobes = (np.abs(np.sin(u * math.pi * 2) * np.sin(v * math.pi * 2)) > 0.72)
    u2, v2 = ((xx * repeats) + 0.5) % 1, ((yy * repeats) + 0.5) % 1
    diamonds = (np.abs(u2 - 0.5) + np.abs(v2 - 0.5)) < 0.08
    pattern = lobes | diamonds
    return tint(pattern.astype(float) * 0.55, paper, motif)


def tex_concrete(rng, size, color, stains=0.5, panels=2):
    """Poured concrete with stains and panel joints."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    mottle = value_noise((size, size), rng, octaves=5, base_freq=4)
    fine = value_noise((size, size), rng, octaves=3, base_freq=48, persistence=0.5)
    level = 0.78 + 0.22 * mottle + 0.08 * fine
    level -= stains * np.clip(value_noise((size, size), rng, octaves=3, base_freq=3) - 0.5, 0, 1) * 1.6
    image = np.array(color)[None, None, :] * level[..., None]
    joints = ((xx * panels) % 1 < 0.008) | ((yy * panels) % 1 < 0.008)
    return tint(joints.astype(float) * 0.5, image, tuple(c * 0.5 for c in color))


def tex_grating(rng, size, color):
    """Steel mesh grating."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    bars = ((xx * 16) % 1 < 0.25) | ((yy * 16) % 1 < 0.25)
    dark = np.array((0.06, 0.06, 0.06))[None, None, :] * np.ones((size, size, 1))
    return np.where(bars[..., None], np.array(color)[None, None, :] * np.ones((size, size, 1)), dark)


def tex_rock(rng, size, color, slabs=34, moss=0.6):
    """Irregular rock slabs (Voronoi cells) with dark joints and moss tint."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    points = rng.random((slabs, 2))
    best = np.full((size, size), 9.0)
    second = np.full((size, size), 9.0)
    owner = np.zeros((size, size), dtype=int)
    for i, (px, py) in enumerate(points):
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                d = np.hypot(xx - px - ox, yy - py - oy)
                closer = d < best
                second = np.where(closer, best, np.minimum(second, d))
                owner = np.where(closer, i, owner)
                best = np.where(closer, d, best)
    shade = 0.70 + 0.45 * ((np.sin(owner * 12.9898) * 43758.5453) % 1)
    grain = value_noise((size, size), rng, octaves=4, base_freq=16, persistence=0.5)
    level = shade * (0.85 + 0.3 * grain)
    image = np.array(color)[None, None, :] * level[..., None]
    joint = np.clip(1 - (second - best) * 40, 0, 1)
    image = tint(joint * 0.92, image, tuple(c * 0.12 for c in color))
    green = value_noise((size, size), rng, octaves=3, base_freq=4)
    return tint(np.clip(green - 0.45, 0, 1) * 2 * moss, image, (0.26, 0.34, 0.08))


def tex_checker(rng, size, color, dark=(0.22, 0.17, 0.13)):
    """Two-tone marble tiles, two by two per texture."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    veins = value_noise((size, size), rng, octaves=5, base_freq=3)
    level = 0.92 + 0.10 * veins
    light = np.array(color)[None, None, :] * level[..., None]
    shade = np.array(dark)[None, None, :] * level[..., None]
    check = ((np.floor(xx * 2) + np.floor(yy * 2)) % 2) == 0
    image = np.where(check[..., None], light, shade)
    grout = ((xx * 2) % 1 < 0.01) | ((yy * 2) % 1 < 0.01)
    return tint(grout.astype(float) * 0.7, image, (0.30, 0.28, 0.25))


def tex_stripes(rng, size, color, stripe=(0.72, 0.40, 0.34), count=10):
    """Vertical striped wallpaper with a fine floral band on each stripe."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    grain = value_noise((size, size), rng, octaves=3, base_freq=24, persistence=0.5)
    paper = np.array(color)[None, None, :] * (0.90 + 0.14 * grain)[..., None]
    u = (xx * count) % 1
    band = (u > 0.30) & (u < 0.70)
    paper = tint(band.astype(float) * 0.6, paper, stripe)
    dots = ((u > 0.44) & (u < 0.56)) & (np.abs(np.sin(yy * math.pi * count * 2)) > 0.85)
    return tint(dots.astype(float) * 0.7, paper, tuple(c * 0.7 for c in stripe))


# family -> (generator, default colour, default tile)
FAMILIES = {
    "plaster": (tex_plaster, (0.70, 0.60, 0.46), (1400, 1400)),
    "wood": (tex_wood, (0.30, 0.20, 0.13), (900, 900)),
    "planks": (tex_planks, (0.42, 0.30, 0.19), (1600, 1600)),
    "panel": (tex_panel, (0.26, 0.17, 0.10), (1200, 2400)),
    "marble": (tex_marble, (0.80, 0.78, 0.75), (2400, 2400)),
    "stone": (tex_stone, (0.70, 0.66, 0.58), (1600, 1600)),
    "parquet": (tex_parquet, (0.48, 0.28, 0.13), (1200, 1200)),
    "carpet": (tex_carpet, (0.50, 0.12, 0.10), (4333, 4475)),
    "runner": (tex_runner, (0.48, 0.11, 0.09), (2400, 1200)),
    "border": (tex_border, (0.32, 0.08, 0.07), (1200, 600)),
    "coffer": (tex_coffer, (0.22, 0.16, 0.10), (3000, 3000)),
    "door": (tex_door, (0.28, 0.19, 0.11), None),
    "double_door": (tex_double_door, (0.24, 0.16, 0.09), None),
    "painting": (tex_painting, (0.60, 0.48, 0.22), None),
    "fanlight": (tex_fanlight, (0.70, 0.60, 0.46), None),
    "glass": (tex_glass, (1, 1, 1), None),
    "window_fan": (tex_window_fan, (0.70, 0.60, 0.46), None),
    "chest": (tex_chest, (0.22, 0.14, 0.08), None),
    "wallpaper": (tex_wallpaper, (0.55, 0.52, 0.38), (1200, 1200)),
    "stripes": (tex_stripes, (0.86, 0.80, 0.68), (1600, 1600)),
    "concrete": (tex_concrete, (0.58, 0.58, 0.54), (3000, 3000)),
    "rock": (tex_rock, (0.62, 0.60, 0.40), (2200, 2200)),
    "grating": (tex_grating, (0.50, 0.52, 0.52), (1200, 1200)),
    "checker": (tex_checker, (0.78, 0.72, 0.62), (2400, 2400)),
    "crate": (tex_crate, (0.40, 0.28, 0.16), None),
}
SOLIDS = {
    "glow": {"kd": (1.0, 0.96, 0.84), "emissive": True},
    "iron": {"kd": (0.16, 0.16, 0.17)},
    "brass": {"kd": (0.52, 0.42, 0.21)},
    "paper": {"kd": (0.92, 0.90, 0.82)},
    "enamel": {"kd": (0.12, 0.12, 0.13)},
    "drum": {"kd": (0.62, 0.55, 0.22)},
    "velvet": {"kd": (0.45, 0.10, 0.12)},
    "soot": {"kd": (0.06, 0.05, 0.05)},
    "water": {"kd": (0.18, 0.32, 0.28)},
    "night": {"kd": (0.05, 0.08, 0.06), "emissive": True},
    "hedge": {"kd": (0.16, 0.26, 0.12)},
    "porcelain": {"kd": (0.52, 0.58, 0.46)},
    "steel": {"kd": (0.60, 0.62, 0.64)},
    "glass_dark": {"kd": (0.18, 0.24, 0.24)},
    "rust": {"kd": (0.52, 0.28, 0.14)},
    "tank": {"kd": (0.72, 1.0, 0.80), "emissive": True},
    "mirror": {"kd": (0.34, 0.38, 0.42)},
    "shade": {"kd": (0.86, 0.80, 0.62), "emissive": True},
    "armor": {"kd": (0.46, 0.48, 0.50)},
    "plastic": {"kd": (0.78, 0.78, 0.72)},
    "copper": {"kd": (0.48, 0.26, 0.16)},
    "lacquer": {"kd": (0.05, 0.05, 0.06)},
    "linen": {"kd": (0.62, 0.62, 0.56)},
    "marble_dark": {"kd": (0.16, 0.15, 0.14)},
}


class Palette:
    """Materials used by one reading: solids plus textured families with tints."""

    def __init__(self, reading):
        self.spec = dict(SOLIDS)
        self.tints = {}
        self.extra = {}
        for name, value in reading.get("materials", {}).items():
            if name in self.spec and "kd" in self.spec[name] and not isinstance(value, dict):
                self.spec[name] = dict(self.spec[name], kd=tuple(value))
                continue
            if isinstance(value, dict):
                value = dict(value)
                if "color" in value:
                    self.tints[name] = value.pop("color")
                self.extra[name] = value
            else:
                self.tints[name] = value

    def use(self, name):
        """Register `name` (family or family.variant) and return the material id."""
        if name in self.spec:
            return name
        family = name.split(".")[0]
        if family not in FAMILIES:
            raise KeyError(f"unknown material {name}")
        _, _, tile = FAMILIES[family]
        extra = self.extra.get(name, {})
        self.spec[name] = {"map": name.replace(".", "-") + ".png", "tile": tuple(extra.get("tile", tile)) if (extra.get("tile", tile)) else None}
        if family in ("fanlight", "glass", "window_fan"):
            self.spec[name]["emissive"] = True
        return name

    def write_textures(self, out, seed):
        rng = np.random.default_rng(seed)
        for name, spec in self.spec.items():
            if "map" not in spec:
                continue
            family = name.split(".")[0]
            generator, default, _ = FAMILIES[family]
            color = tuple(self.tints.get(name, self.tints.get(family, default)))
            kwargs = {k: v for k, v in self.extra.get(name, {}).items() if k not in ("tile",)}
            image = generator(rng, 512, color, **kwargs)
            to_image(image).save(out / spec["map"])


# ------------------------------------------------------------------ lights --
class Rig:
    def __init__(self, reading):
        lighting = reading.get("lighting", {})
        self.ambient = tuple(lighting.get("ambient", (0.14, 0.12, 0.10)))
        self.up_boost = lighting.get("up_boost", 0.35)
        self.lights = []
        for light in reading.get("light", []):
            self.add(light["at"], light.get("color", (1.0, 0.9, 0.72)), light.get("strength", 0.3), light.get("reach", 5000))

    def add(self, at, color, strength, reach):
        self.lights.append((tuple(float(v) for v in at), tuple(color), float(strength), float(reach)))

    def __call__(self, point, normal):
        nx, ny, nz = normal
        up = max(0.0, -ny)
        color = [a * (0.8 + self.up_boost * up) for a in self.ambient]
        for (lx, ly, lz), tint_, strength, reach in self.lights:
            dx, dy, dz = lx - point[0], ly - point[1], lz - point[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            lambert = (nx * dx + ny * dy + nz * dz) / dist
            wrap = max(0.0, lambert + 0.35) / 1.35
            falloff = 1.0 / (1.0 + (dist / reach) ** 2)
            for k in range(3):
                color[k] += tint_[k] * strength * wrap * falloff
        return tuple(min(1.0, max(0.0, c)) for c in color)


# ------------------------------------------------------------------- shell --
def exposed_edges(zones):
    """Wall segments of a union of axis-aligned zones: (side, fixed, a, b, zone)."""
    walls = []
    for i, zone in enumerate(zones):
        x0, z0, x1, z1 = zone["box"]
        for side in ("west", "east", "south", "north"):
            if side in ("west", "east"):
                fixed = x0 if side == "west" else x1
                span = [(z0, z1)]
                for j, other in enumerate(zones):
                    if i == j:
                        continue
                    ox0, oz0, ox1, oz1 = other["box"]
                    touching = (ox1 == fixed) if side == "west" else (ox0 == fixed)
                    if touching:
                        span = _subtract(span, (oz0, oz1))
            else:
                fixed = z0 if side == "south" else z1
                span = [(x0, x1)]
                for j, other in enumerate(zones):
                    if i == j:
                        continue
                    ox0, oz0, ox1, oz1 = other["box"]
                    touching = (oz1 == fixed) if side == "south" else (oz0 == fixed)
                    if touching:
                        span = _subtract(span, (ox0, ox1))
            for a, b in span:
                if b - a > 1:
                    walls.append((side, fixed, a, b, zone))
    return walls


def _subtract(intervals, cut):
    c0, c1 = cut
    result = []
    for a, b in intervals:
        if c1 <= a or c0 >= b:
            result.append((a, b))
            continue
        if c0 > a:
            result.append((a, c0))
        if c1 < b:
            result.append((c1, b))
    return result


def wall_rect(mesh, side, fixed, a, b, bottom, top, material, flip=False):
    """Vertical rectangle on a zone edge facing into the zone."""
    height = bottom - top
    if height <= 0:
        return
    if side == "west":
        mesh.rect((fixed, bottom, a), (0, 0, 1), (0, -1, 0), b - a, height, material, flip)
    elif side == "east":
        mesh.rect((fixed, bottom, b), (0, 0, -1), (0, -1, 0), b - a, height, material, flip)
    elif side == "south":
        mesh.rect((b, bottom, fixed), (-1, 0, 0), (0, -1, 0), b - a, height, material, flip)
    else:
        mesh.rect((a, bottom, fixed), (1, 0, 0), (0, -1, 0), b - a, height, material, flip)


def skirting_box(mesh, side, fixed, a, b, height, depth, material, floor=0):
    if side == "west":
        mesh.box((fixed, floor - height, a), (fixed + depth, floor, b), material, omit=("bottom", "west"))
    elif side == "east":
        mesh.box((fixed - depth, floor - height, a), (fixed, floor, b), material, omit=("bottom", "east"))
    elif side == "south":
        mesh.box((a, floor - height, fixed), (b, floor, fixed + depth), material, omit=("bottom", "south"))
    else:
        mesh.box((a, floor - height, fixed - depth), (b, floor, fixed), material, omit=("bottom", "north"))


# ------------------------------------------------------------------- kinds --
class Builder:
    def __init__(self, reading, rdt):
        self.reading = reading
        self.room = survey(rdt)
        self.palette = Palette(reading)
        self.rig = Rig(reading)
        self.zones = reading["zone"]
        self._raise_ceilings()
        self.floor = reading.get("shell", {}).get("floor", 0)
        header = (
            f"# Built by tools/room3d_reading.py from a room reading; RE1 room units, -Y up.",
            f"# {reading.get('title', reading['room'])}: shell from the reading's zones, kinds from the library.",
        )
        self.mesh = Mesh(reading["room"].lower(), self.palette.spec, self.rig, header=header, cell=reading.get("shell", {}).get("cell", 900.0))
        self.hides = []
        self.auto_hides = []   # (camera, group) pairs derived from camera positions
        self.cameras = self.room["cameras"]
        self._lights_from_kinds()

    def _cameras_behind(self, side, fixed, a, b, margin=2500):
        """Cameras on the far side of a wall segment, which would see its back."""
        behind = []
        for cam in self.cameras:
            ox, oy, oz = cam["origin"]
            along, across = (oz, ox) if side in ("west", "east") else (ox, oz)
            outside = (across < fixed) if side in ("west", "south") else (across > fixed)
            if outside and a - margin <= along <= b + margin:
                behind.append(cam["index"])
        return behind

    def _cameras_above(self, zone, ceiling):
        x0, z0, x1, z1 = zone["box"]
        return [cam["index"] for cam in self.cameras
                if cam["origin"][1] < ceiling and x0 - 2500 <= cam["origin"][0] <= x1 + 2500 and z0 - 2500 <= cam["origin"][2] <= z1 + 2500]

    def _raise_ceilings(self):
        """Lift a zone's ceiling above any camera standing over it.

        The original views from such cameras show walls rising toward the
        lens; a ceiling below the camera would be hidden and leave the upper
        frame black. A reading opts out per zone with fixed_ceiling = true.
        """
        shell = self.reading.get("shell", {})
        if shell.get("auto_ceiling", True) is False:
            return
        default = shell.get("ceiling", -6000)
        for zone in self.zones:
            if zone.get("fixed_ceiling") or zone.get("open_floor"):
                continue
            ceiling = zone.get("ceiling", default)
            if ceiling is None:
                continue
            x0, z0, x1, z1 = zone["box"]
            highest = min((cam["origin"][1] for cam in self.room["cameras"]
                           if x0 - 1500 <= cam["origin"][0] <= x1 + 1500
                           and z0 - 1500 <= cam["origin"][2] <= z1 + 1500), default=0)
            if highest < ceiling + 400:
                zone["ceiling"] = int(highest - 900)

    # Lights that fixtures imply are added before any geometry is lit.
    def _lights_from_kinds(self):
        for sconce in self.reading.get("sconce", []):
            x, y, z = self._wall_point(sconce["wall"], sconce["at"])
            d = SIDES[sconce["wall"]]
            self.rig.add((x + d[0] * 420, y, z + d[2] * 420), sconce.get("color", (1.0, 0.90, 0.72)),
                         sconce.get("strength", 0.28), sconce.get("reach", 5000))
        for lamp in self.reading.get("wall_lamp", []):
            x, y, z = self._wall_point(lamp["wall"], lamp["at"])
            d = SIDES[lamp["wall"]]
            self.rig.add((x + d[0] * 260, y + 150, z + d[2] * 260), lamp.get("color", (1.0, 0.86, 0.62)),
                         lamp.get("strength", 0.9), lamp.get("reach", 4200))
        for ch in self.reading.get("chandelier", []):
            for gx, gy, gz in self._globes(ch):
                self.rig.add((gx, gy, gz), ch.get("color", (1.0, 0.92, 0.74)), ch.get("strength", 0.19), ch.get("reach", 6000))
        for o in self.reading.get("object", []):
            if o.get("kind") == "lamp":
                x, z = o["at"]
                self.rig.add((x, o.get("base", self.floor) - o.get("height", 650) + 100, z), (1.0, 0.92, 0.75),
                             o.get("strength", 0.35), o.get("reach", 3500))
        for post in self.reading.get("lamp_post", []):
            self.rig.add((post["at"][0], post.get("top", -4600), post["at"][1]), (1.0, 0.92, 0.75), post.get("strength", 0.4), post.get("reach", 7000))
        for win in self.reading.get("window", []):
            x, y, z = self._wall_point(win["wall"], (win.get("sill", -8000) - win.get("height", 3000) / 2, win["at"]))
            d = SIDES[win["wall"]]
            self.rig.add((x + d[0] * 400, y, z + d[2] * 400), win.get("color", (0.75, 0.80, 1.0)), win.get("strength", 0.3), win.get("reach", 9000))

    # Helpers -------------------------------------------------------------
    def _wall_line(self, side, along=None):
        """Wall coordinate for a named side, from the zone that spans `along`."""
        candidates = []
        for zone in self.zones:
            x0, z0, x1, z1 = zone["box"]
            if side in ("west", "east"):
                if along is None or z0 <= along <= z1:
                    candidates.append(x0 if side == "west" else x1)
            else:
                if along is None or x0 <= along <= x1:
                    candidates.append(z0 if side == "south" else z1)
        if not candidates:
            raise ValueError(f"no zone has a {side} wall at {along}")
        return min(candidates) if side in ("west", "south") else max(candidates)

    def _wall_point(self, side, at):
        """(x, y, z) on a wall: `at` is (y, coordinate along the wall)."""
        y, along = at
        line = self._wall_line(side, along)
        if side in ("west", "east"):
            return (line, y, along)
        return (along, y, line)

    def _globes(self, ch):
        cx, cz = ch["at"]
        r, phase, n = ch.get("radius", 650), ch.get("phase", 0.0), ch.get("globes", 4)
        return [(cx + r * math.cos(phase + k * math.tau / n), ch["y"], cz + r * math.sin(phase + k * math.tau / n)) for k in range(n)]

    def use(self, name):
        return self.palette.use(name)

    # Shell ---------------------------------------------------------------
    def shell(self):
        mesh, floor = self.mesh, self.floor
        shell = self.reading.get("shell", {})
        skirt = shell.get("skirting", 240)
        dado = shell.get("dado", 0)
        dado_material = self.use(shell.get("dado_material", "panel")) if dado else None
        rail = shell.get("chair_rail", False)
        trim = self.use(shell.get("trim", "wood"))
        for zone in self.zones:
            x0, z0, x1, z1 = zone["box"]
            group = zone.get("group", "floor")
            below = [cam["index"] for cam in self.cameras if cam["origin"][1] > floor]
            if below and "group" not in zone:
                group = f"floor_{self.zones.index(zone)}"
                self.auto_hides += [(c, group) for c in below]
            mesh.set_group(group)
            fm = self.use(zone.get("floor_material", shell.get("floor_material", "planks")))
            if zone.get("open_floor"):
                pass  # a void or stairwell: walls still stop at its edges, a slab or stair fills it
            elif zone.get("floor_along") == "z":
                mesh.rect((x0, floor, z0), (0, 0, 1), (1, 0, 0), z1 - z0, x1 - x0, fm, flip=True)
            else:
                mesh.rect((x0, floor, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, fm)
            ceiling = zone.get("ceiling", shell.get("ceiling", -6000))
            if ceiling is not None:
                group = zone.get("ceiling_group", "ceiling")
                above = self._cameras_above(zone, ceiling)
                if above and "ceiling_group" not in zone:
                    group = f"ceiling_{self.zones.index(zone)}"
                    self.auto_hides += [(c, group) for c in above]
                mesh.set_group(group)
                cm = self.use(zone.get("ceiling_material", shell.get("ceiling_material", "plaster")))
                mesh.rect((x0, ceiling, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, cm, flip=True)
        mesh.set_group("walls")
        wm_default = shell.get("wall_material", "plaster")
        for index, (side, fixed, a, b, zone) in enumerate(exposed_edges(self.zones)):
            ceiling = zone.get("ceiling", shell.get("ceiling", -6000))
            wm = self.use(zone.get("wall_materials", {}).get(side, zone.get("wall_material", wm_default)))
            group = zone.get("wall_groups", {}).get(side, zone.get("wall_group"))
            behind = self._cameras_behind(side, fixed, a, b)
            if behind and not group:
                group = f"wall_{index}"
                self.auto_hides += [(c, group) for c in behind]
            if group:
                mesh.set_group(group)
            base = floor
            if dado:
                wall_rect(mesh, side, fixed, a, b, floor, floor - dado, dado_material)
                base = floor - dado
                if rail:
                    skirting_box(mesh, side, fixed, a, b, 80, 40, trim, floor=floor - dado + 80)
            wall_rect(mesh, side, fixed, a, b, base, ceiling, wm)
            if skirt:
                skirting_box(mesh, side, fixed, a, b, skirt, 55, trim, floor)
            # Cornice where the wall meets the ceiling (interiors: skirting set).
            cornice = zone.get("cornice", shell.get("cornice", 200 if skirt else 0))
            if cornice and ceiling is not None:
                skirting_box(mesh, side, fixed, a, b, cornice, 70, trim, floor=ceiling + cornice)
            if group:
                mesh.set_group("walls")
        # Soffits between zones with different ceilings.
        for i, zone in enumerate(self.zones):
            ci = zone.get("ceiling", shell.get("ceiling", -6000))
            x0, z0, x1, z1 = zone["box"]
            for other in self.zones[i + 1:]:
                cj = other.get("ceiling", shell.get("ceiling", -6000))
                if ci == cj:
                    continue
                ox0, oz0, ox1, oz1 = other["box"]
                lo, hi = (ci, cj) if ci > cj else (cj, ci)  # lo is the lower ceiling (less negative)
                cm = self.use(shell.get("ceiling_material", "plaster"))
                if x1 == ox0 or x0 == ox1:
                    fixed = x1 if x1 == ox0 else x0
                    a, b = max(z0, oz0), min(z1, oz1)
                    if b > a:
                        side = "east" if (ci > cj) == (x1 == ox0) else "west"
                        wall_rect(mesh, side, fixed, a, b, lo, hi, cm, flip=True)
                if z1 == oz0 or z0 == oz1:
                    fixed = z1 if z1 == oz0 else z0
                    a, b = max(x0, ox0), min(x1, ox1)
                    if b > a:
                        side = "north" if (ci > cj) == (z1 == oz0) else "south"
                        wall_rect(mesh, side, fixed, a, b, lo, hi, cm, flip=True)
        mesh.set_group(None)

    # Kinds -----------------------------------------------------------------
    def doors(self):
        for index, d in enumerate(self.reading.get("door", [])):
            side = d["wall"]
            a, b = d["span"]
            line = d.get("line", self._wall_line(side, (a + b) / 2))
            material = self.use(d.get("material", "door"))
            frame = self.use(d.get("frame", "wood"))
            top = d.get("top", -3300)
            group = d.get("group")
            behind = self._cameras_behind(side, line, a, b)
            if behind and not group:
                group = f"door_{index}"
                self.auto_hides += [(c, group) for c in behind]
            self.mesh.set_group(group)
            if side in ("south", "north"):
                door(self.mesh, a, b, line, top, material, +1 if side == "south" else -1, frame_depth=d.get("frame_depth", 110), floor=self.floor, frame=frame)
            else:
                self._door_on_x_wall(side, line, a, b, top, material, frame, d.get("bottom", self.floor))
            self.mesh.set_group(None)

    def _door_on_x_wall(self, side, line, z0, z1, top, material, frame, floor):
        mesh = self.mesh
        d = 1 if side == "west" else -1
        inner = line + d * 70
        if d > 0:
            mesh.rect((inner, floor, z0), (0, 0, 1), (0, -1, 0), z1 - z0, floor - top, material)
        else:
            mesh.rect((inner, floor, z1), (0, 0, -1), (0, -1, 0), z1 - z0, floor - top, material)
        xa, xb = (line, line + 110) if d > 0 else (line - 110, line)
        for za in (z0 - 90, z1):
            mesh.box((xa, top - 110, za), (xb, floor, za + 90), frame, omit=("bottom", "west" if d > 0 else "east"))
        mesh.box((xa, top - 110, z0 - 90), (xb, top, z1 + 90), frame, omit=("west" if d > 0 else "east",))

    def windows(self):
        mesh = self.mesh
        for w in self.reading.get("window", []):
            side, kind = w["wall"], w.get("kind", "arch")
            along = w["at"]
            line = self._wall_line(side, along)
            d = SIDES[side]
            stone = self.use(w.get("frame", "stone"))
            if kind == "fanlight":
                glass = self.use("fanlight")
                r, sill = w["radius"], w["sill"]
                z = line + d[2] * 60
                n = 16
                arc = [(along + r * math.cos(t * math.pi / n), sill - r * math.sin(t * math.pi / n), z) for t in range(n + 1)]
                mesh.poly(list(reversed(arc)) if d[2] > 0 else arc, glass)
                mesh.box((along - r - 380, sill, min(line, z)), (along + r + 380, sill + 300, max(line, z) + 200), stone)
                for t in range(n):
                    a0, a1 = t * math.pi / n, (t + 1) * math.pi / n
                    inner = [(along + r * math.cos(a), sill - r * math.sin(a)) for a in (a0, a1)]
                    outer = [(along + (r + 380) * math.cos(a), sill - (r + 380) * math.sin(a)) for a in (a0, a1)]
                    pts = [(inner[0][0], inner[0][1], z), (outer[0][0], outer[0][1], z), (outer[1][0], outer[1][1], z), (inner[1][0], inner[1][1], z)]
                    mesh.poly(pts if d[2] > 0 else list(reversed(pts)), stone)
            elif kind == "fan":  # 10B-style round arched window drawn in one texture
                glass = self.use("window_fan")
                r, sill = w["radius"], w["sill"]
                if side in ("north", "south"):
                    z = line + d[2] * 40
                    origin = (along - r - 150, sill, z) if d[2] < 0 else (along + r + 150, sill, z)
                    eu = (1, 0, 0) if d[2] < 0 else (-1, 0, 0)
                    mesh.rect(origin, eu, (0, -1, 0), 2 * r + 300, r + 150, glass)
                    mesh.box((along - r - 150, sill, min(line, z) - 50), (along + r + 150, sill + 90, max(line, z)), self.use("wood"))
            else:  # tall arched window
                glass = self.use("glass")
                half, sill, height = w.get("half", 1200), w["sill"], w.get("height", 4200)
                top = sill - height
                z = line + d[2] * 50 if side in ("north", "south") else None
                if side in ("north", "south"):
                    origin = (along + half, sill, z) if d[2] < 0 else (along - half, sill, z)
                    eu = (-1, 0, 0) if d[2] < 0 else (1, 0, 0)
                    mesh.rect(origin, eu, (0, -1, 0), 2 * half, height - half, glass)
                    n = 10
                    arch = [(along + half * math.cos(math.pi * t / n), top + half - half * math.sin(math.pi * t / n), z) for t in range(n + 1)]
                    mesh.poly(list(reversed(arch)) if d[2] < 0 else arch, glass)
                    for xa in (along - half - 160, along + half):
                        mesh.box((xa, top + half, min(line, z) - 90), (xa + 160, sill, max(line, z)), stone)
                    mesh.box((along - half - 260, sill, min(line, z) - 170), (along + half + 260, sill + 180, max(line, z)), stone)
                else:
                    x = line + d[0] * 40
                    origin = (x, sill, along - half) if d[0] > 0 else (x, sill, along + half)
                    eu = (0, 0, 1) if d[0] > 0 else (0, 0, -1)
                    mesh.rect(origin, eu, (0, -1, 0), 2 * half, height, glass)
                    for za in (along - half - 120, along + half):
                        mesh.box((min(line, x) - 100, top - 120, za), (max(line, x) + 100, sill, za + 120), stone)

    def paintings(self):
        for p in self.reading.get("painting", []):
            side = p["wall"]
            a, b = p["span"]
            top, bottom = p["y"]
            line = self._wall_line(side, (a + b) / 2)
            d = SIDES[side]
            canvas = self.use(p.get("material", "painting"))
            frame = self.use(p.get("frame", "brass"))
            if side in ("west", "east"):
                x = line + d[0] * 70
                origin = (x, bottom, a) if d[0] > 0 else (x, bottom, b)
                eu = (0, 0, 1) if d[0] > 0 else (0, 0, -1)
                self.mesh.rect(origin, eu, (0, -1, 0), b - a, bottom - top, canvas)
                xa, xb = (line, line + 110) if d[0] > 0 else (line - 110, line)
                for za, zb in ((a - 110, a), (b, b + 110)):
                    self.mesh.box((xa, top - 110, za), (xb, bottom + 110, zb), frame)
                for ya, yb in ((top - 110, top), (bottom, bottom + 110)):
                    self.mesh.box((xa, ya, a - 110), (xb, yb, b + 110), frame)
            else:
                z = line + d[2] * 70
                origin = (b, bottom, z) if d[2] > 0 else (a, bottom, z)
                eu = (-1, 0, 0) if d[2] > 0 else (1, 0, 0)
                self.mesh.rect(origin, eu, (0, -1, 0), b - a, bottom - top, canvas)
                za, zb = (line, line + 110) if d[2] > 0 else (line - 110, line)
                for xa, xb in ((a - 110, a), (b, b + 110)):
                    self.mesh.box((xa, top - 110, za), (xb, bottom + 110, zb), frame)
                for ya, yb in ((top - 110, top), (bottom, bottom + 110)):
                    self.mesh.box((a - 110, ya, za), (b + 110, yb, zb), frame)

    def sconces(self):
        mesh = self.mesh
        brass, glow = self.use("brass"), self.use("glow")
        for s in self.reading.get("sconce", []):
            x, y, z = self._wall_point(s["wall"], s["at"])
            dx, dz = SIDES[s["wall"]][0], SIDES[s["wall"]][2]
            candles = s.get("candles", 3)
            mesh.box((x + min(0, dx * 120) - 70 * abs(dz), y - 220, z + min(0, dz * 120) - 70 * abs(dx)),
                     (x + max(0, dx * 120) + 70 * abs(dz), y + 220, z + max(0, dz * 120) + 70 * abs(dx)), brass)
            offsets = (-1, 0, 1) if candles == 3 else (0,)
            for side in offsets:
                ax, az = x + dx * 300 - dz * side * 260, z + dz * 300 + dx * side * 260
                mesh.box((ax - 25, y - 30 + (0 if side else -120), az - 25), (ax + 25, y + 30, az + 25), brass)
                mesh.sphere((ax, y - 160 + (0 if side else -120), az), 95, glow, rings=3, segments=6)
        for lamp in self.reading.get("wall_lamp", []):
            x, y, z = self._wall_point(lamp["wall"], lamp["at"])
            dx, dz = SIDES[lamp["wall"]][0], SIDES[lamp["wall"]][2]
            iron = self.use("iron")
            mesh.box((x + min(0, dx * 110) - 100 * abs(dz), y - 60, z + min(0, dz * 110) - 100 * abs(dx)),
                     (x + max(0, dx * 110) + 100 * abs(dz), y + 60, z + max(0, dz * 110) + 100 * abs(dx)), iron)
            ax, az = x + dx * 260, z + dz * 260
            mesh.cylinder(ax, az, 160, y - 20, y + 120, iron, segments=10, caps=False)
            mesh.cylinder(ax, az, 160, y - 20, y, iron, segments=10)
            mesh.sphere((ax, y + 60, az), 95, glow, rings=4, segments=8)

    def chandeliers(self):
        mesh = self.mesh
        iron, brass, glow = self.use("iron"), self.use("brass"), self.use("glow")
        for ch in self.reading.get("chandelier", []):
            cx, cz = ch["at"]
            ceiling = ch["ceiling"]
            gy = ch["y"]
            mesh.cylinder(cx, cz, 22, ceiling, gy - 700, iron, segments=6, caps=False)
            mesh.cylinder(cx, cz, 180, ceiling, ceiling + 60, iron, segments=10)
            hub_top, hub_bottom = gy - 700, gy - 420
            mesh.cylinder(cx, cz, 120, hub_top, hub_bottom, iron, segments=8)
            mesh.sphere((cx, hub_bottom + 90, cz), 110, brass, rings=4, segments=8)
            for gx, _, gz in self._globes(ch):
                ax, az = gx - cx, gz - cz
                for i in range(5):
                    f0, f1 = i / 5, (i + 1) / 5
                    drop0 = 260 * math.sin(math.pi * f0) - 120 * f0
                    drop1 = 260 * math.sin(math.pi * f1) - 120 * f1
                    a = (cx + ax * f0, hub_bottom - 60 + drop0, cz + az * f0)
                    b = (cx + ax * f1, hub_bottom - 60 + drop1, cz + az * f1)
                    mesh.box((min(a[0], b[0]) - 18, min(a[1], b[1]) - 18, min(a[2], b[2]) - 18),
                             (max(a[0], b[0]) + 18, max(a[1], b[1]) + 18, max(a[2], b[2]) + 18), iron)
                mesh.cylinder(gx, gz, 70, gy + 180, gy + 300, brass, segments=8)
                mesh.sphere((gx, gy, gz), ch.get("globe_radius", 230), glow, rings=6, segments=10)

    def column(self, x, z, top, base=None, radius=520, material="stone", square=False):
        stone = self.use(material)
        if square:
            base = self.floor if base is None else base
            self.mesh.box((x - radius, top, z - radius), (x + radius, base, z + radius), stone, omit=("bottom", "top"))
            return
        base = self.floor if base is None else base
        self.mesh.box((x - radius - 180, base - 300, z - radius - 180), (x + radius + 180, base, z + radius + 180), stone, omit=("bottom",))
        self.mesh.cylinder(x, z, radius, top + 450, base - 300, stone, segments=14, caps=False)
        self.mesh.box((x - radius - 120, top, z - radius - 120), (x + radius + 120, top + 450, z + radius + 120), stone)

    def arcade_bay(self, axis, line, a, b, spring, top, thickness=300, segments=12):
        stone = self.use("stone")
        mesh = self.mesh
        radius, centre = (b - a) / 2, (a + b) / 2
        arc = [(centre - radius * math.cos(t * math.pi / segments), spring - radius * math.sin(t * math.pi / segments)) for t in range(segments + 1)]
        for face in (-1, 1):
            off = line + face * thickness / 2
            for (s0, y0), (s1, y1) in zip(arc, arc[1:]):
                pts = [(off, y0, s0), (off, y1, s1), (off, top, s1), (off, top, s0)] if axis == "z" else [(s0, y0, off), (s1, y1, off), (s1, top, off), (s0, top, off)]
                mesh.poly(pts if face > 0 else list(reversed(pts)), stone)
        lo, hi = line - thickness / 2, line + thickness / 2
        for (s0, y0), (s1, y1) in zip(arc, arc[1:]):
            pts = [(lo, y0, s0), (hi, y0, s0), (hi, y1, s1), (lo, y1, s1)] if axis == "z" else [(s0, y0, lo), (s0, y0, hi), (s1, y1, hi), (s1, y1, lo)]
            mesh.poly(pts, stone, flip=True)

    def arcades(self):
        mesh = self.mesh
        stone = self.use("stone")
        for arc in self.reading.get("arcade", []):
            axis, line = arc["axis"], arc["line"]
            cols = arc["columns"]
            spring, top, r = arc.get("spring", -3500), arc["top"], arc.get("radius", 520)
            mesh.set_group(arc.get("group", "gallery"))
            for c in cols:
                self.column(*((line, c) if axis == "z" else (c, line)), spring, radius=r)
                for face in (-1, 1):
                    off = line + face * 150
                    pts = ([(off, spring, c - r), (off, spring, c + r), (off, top, c + r), (off, top, c - r)] if axis == "z"
                           else [(c - r, spring, off), (c + r, spring, off), (c + r, top, off), (c - r, top, off)])
                    mesh.poly(pts if (face > 0) == (axis == "z") else list(reversed(pts)), stone)
            ends = arc.get("ends")
            bays = [tuple(b) for b in arc.get("bays", [])]
            if not bays:
                if ends and ends[0] is not None:
                    bays.append((ends[0], cols[0] - r))
                bays += [(p + r, q - r) for p, q in zip(cols, cols[1:])]
                if ends and ends[1] is not None:
                    bays.append((cols[-1] + r, ends[1]))
            for a, b in bays:
                if b - a > 2 * r:
                    self.arcade_bay(axis, line, a, b, spring, top, arc.get("thickness", 300))
            for pil in arc.get("pilasters", []):
                for face in (-1, 1):
                    off = line + face * 150
                    pts = ([(off, spring, pil - 300), (off, spring, pil + 300), (off, top, pil + 300), (off, top, pil - 300)] if axis == "z"
                           else [(pil - 300, spring, off), (pil + 300, spring, off), (pil + 300, top, off), (pil - 300, top, off)])
                    mesh.poly(pts if (face > 0) == (axis == "z") else list(reversed(pts)), stone)
            mesh.set_group(None)
        for col in self.reading.get("column", []):
            mesh.set_group(col.get("group"))
            self.column(col["at"][0], col["at"][1], col.get("top", -3500), col.get("base"), col.get("radius", 520),
                        col.get("material", "stone"), col.get("square", False))
            mesh.set_group(None)

    def slabs(self):
        """Galleries and landings: a floor plate at a height with edges and rails."""
        mesh = self.mesh
        for s in self.reading.get("slab", []):
            x0, z0, x1, z1 = s["box"]
            top, thick = s["top"], s.get("thickness", 400)
            mesh.set_group(s.get("group", "gallery"))
            mesh.rect((x0, top, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, self.use(s.get("top_material", "planks")))
            mesh.rect((x0, top + thick, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, self.use(s.get("bottom_material", "plaster")), flip=True)
            edge = self.use(s.get("edge_material", "stone"))
            for side in s.get("edges", []):
                fixed = {"west": x0, "east": x1, "south": z0, "north": z1}[side]
                a, b = (z0, z1) if side in ("west", "east") else (x0, x1)
                # edge faces outward: away from the slab
                outward = {"west": "east", "east": "west", "south": "north", "north": "south"}[side]
                wall_rect(mesh, outward, fixed, a, b, top + thick, top, edge)
            for band in s.get("edge_band", []):
                side, width = band["side"], band.get("width", 450)
                mat = self.use(band.get("material", "stone"))
                if side == "west":
                    mesh.rect((x0, top - 8, z0), (1, 0, 0), (0, 0, 1), width, z1 - z0, mat)
                elif side == "east":
                    mesh.rect((x1 - width, top - 8, z0), (1, 0, 0), (0, 0, 1), width, z1 - z0, mat)
                elif side == "south":
                    mesh.rect((x0, top - 8, z0), (1, 0, 0), (0, 0, 1), x1 - x0, width, mat)
                else:
                    mesh.rect((x0, top - 8, z1 - width), (1, 0, 0), (0, 0, 1), x1 - x0, width, mat)
            for rail in s.get("rails", []):
                side = rail["side"]
                inset = rail.get("inset", 90)
                a, b = rail.get("span", (z0, z1) if side in ("west", "east") else (x0, x1))
                pad = rail.get("pad", 100)
                if side == "west":
                    start, end = (x0 + inset, top, a + pad), (x0 + inset, top, b - pad)
                elif side == "east":
                    start, end = (x1 - inset, top, a + pad), (x1 - inset, top, b - pad)
                elif side == "south":
                    start, end = (a + pad, top, z0 + inset), (b - pad, top, z0 + inset)
                else:
                    start, end = (a + pad, top, z1 - inset), (b - pad, top, z1 - inset)
                balustrade(mesh, start, end, height=rail.get("height", 950), spacing=rail.get("spacing", 420),
                           newels=tuple(rail.get("newels", (True, True))), material=self.use(rail.get("material", "wood")))
            mesh.set_group(None)

    def stairs(self):
        mesh, floor = self.mesh, self.floor
        for st in self.reading.get("stair", []):
            x0, x1 = st["x"]
            z0, z1 = st["z"]
            top = st["top"]
            base = st.get("base", floor)             # height of the foot (landing flights start higher)
            steps = st.get("steps", 20)
            direction = st.get("direction", "north")  # climbing direction
            rise = (base - top) / steps
            tread_mat = self.use(st.get("tread_material", "stone"))
            runner = st.get("runner")
            runner_mat = self.use(st.get("runner_material", "runner")) if runner else None
            parapet = st.get("parapet", 0)
            stone = self.use("stone")
            mesh.set_group(st.get("group", "stairs"))
            along = (z0, z1) if direction in ("north", "south") else (x0, x1)
            run = (along[1] - along[0]) / steps
            for i in range(steps):
                s0 = along[0] + i * run if direction in ("north", "east") else along[1] - (i + 1) * run
                s1 = s0 + run
                t, b = base - (i + 1) * rise, base - i * rise
                if direction in ("north", "south"):
                    spans = [(x0, x1, tread_mat)] if not runner else [(x0, runner[0], tread_mat), (runner[0], runner[1], runner_mat), (runner[1], x1, tread_mat)]
                    for a, c, m in spans:
                        mesh.box((a, t, s0), (c, b, s1), m, omit=("bottom", "west", "east", "north" if direction == "north" else "south"))
                else:
                    spans = [(z0, z1, tread_mat)] if not runner else [(z0, runner[0], tread_mat), (runner[0], runner[1], runner_mat), (runner[1], z1, tread_mat)]
                    for a, c, m in spans:
                        mesh.box((s0, t, a), (s1, b, c), m, omit=("bottom", "south", "north", "east" if direction == "east" else "west"))
            if parapet and direction in ("north", "south"):
                for px0, px1 in ((x0 - parapet, x0), (x1, x1 + parapet)):
                    for i in range(steps):
                        s0 = z0 + i * run if direction == "north" else z1 - (i + 1) * run
                        mesh.box((px0, base - (i + 1) * rise - 350, s0), (px1, floor, s0 + run), stone, omit=("bottom",))
                    rail_x = (px0 + px1) / 2
                    foot, head = (z0 + 150, z1 - 150) if direction == "north" else (z1 - 150, z0 + 150)
                    balustrade(mesh, (rail_x, base - rise - 350, foot), (rail_x, top - 350, head), height=800, material=self.use("wood"))
                if st.get("under", True):
                    panel = self.use(st.get("under_material", "panel"))
                    for px, flip in ((x0 - parapet, True), (x1 + parapet, False)):
                        pts = [(px, floor, z0), (px, floor, z1), (px, top + 400, z1), (px, floor, z0)]
                        mesh.poly(list(reversed(pts)) if flip else pts, panel)
                    back = z1 if direction == "north" else z0
                    wall_rect(mesh, "south" if direction == "north" else "north", back, x0 - parapet, x1 + parapet, floor, top + 400, panel, flip=True)
            # Open-string rail along one side of an east/west flight.
            for rail in st.get("rails", []):
                side = rail["side"]
                zr = z0 + rail.get("inset", 60) if side == "south" else z1 - rail.get("inset", 60)
                foot_x, head_x = (x0 + 60, x1 - 60) if direction == "east" else (x1 - 60, x0 + 60)
                balustrade(mesh, (foot_x, base, zr), (head_x, top, zr), height=rail.get("height", 950),
                           newels=tuple(rail.get("newels", (True, True))), material=self.use("wood"))
            # Sloped string boards and soffit for open east/west flights.
            if st.get("string") and direction in ("east", "west"):
                wood = self.use("wood")
                plaster = self.use(st.get("soffit_material", "plaster"))
                for i in range(steps):
                    xa = x0 + i * run if direction == "east" else x1 - (i + 1) * run
                    xb = xa + run
                    def nose(x):
                        f = (x - x0) / (x1 - x0) if direction == "east" else (x1 - x) / (x1 - x0)
                        return base - f * (base - top) - 250
                    def soffit(x):
                        f = (x - x0) / (x1 - x0) if direction == "east" else (x1 - x) / (x1 - x0)
                        return min(floor, base - f * (base - top) + rise + 20)
                    for zs, flip in ((z0, False), (z1, True)):
                        pts = [(xa, soffit(xa), zs), (xa, nose(xa), zs), (xb, nose(xb), zs), (xb, soffit(xb), zs)]
                        mesh.poly(list(reversed(pts)) if flip else pts, wood)
                    mesh.poly([(xa, soffit(xa), z0), (xa, soffit(xa), z1), (xb, soffit(xb), z1), (xb, soffit(xb), z0)], plaster)
            mesh.set_group(None)

    def objects(self):
        mesh, floor = self.mesh, self.floor
        for o in self.reading.get("object", []):
            kind = o["kind"]
            mesh.set_group(o.get("group"))
            if kind == "box":
                x0, z0, x1, z1 = o["box"]
                mesh.box((x0, o["top"], z0), (x1, o.get("bottom", floor), z1), self.use(o.get("material", "crate")), omit=tuple(o.get("omit", ("bottom",))))
            elif kind == "cylinder":
                x, z = o["at"]
                mesh.cylinder(x, z, o["radius"], o["top"], o.get("bottom", floor), self.use(o.get("material", "drum")), segments=o.get("segments", 14))
                for hoop in o.get("hoops", []):
                    mesh.cylinder(x, z, o["radius"] + 16, hoop, hoop + 70, self.use(o.get("hoop_material", "iron")), segments=o.get("segments", 14), caps=False)
            elif kind == "chest":
                self._chest(o)
            elif kind == "table_round":
                self._table(o)
            elif kind == "lamp_post":
                self._lamp_post(o)
            elif kind == "rug":
                x0, z0, x1, z1 = o["box"]
                mat = self.use(o.get("material", "carpet"))
                mesh.rect((x0, floor - 12, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, mat)
                bw = o.get("border", 0)
                if bw:
                    bm = self.use(o.get("border_material", "border"))
                    mesh.rect((x0 - bw, floor - 12, z0 - bw), (1, 0, 0), (0, 0, 1), x1 - x0 + 2 * bw, bw, bm)
                    mesh.rect((x0 - bw, floor - 12, z1), (1, 0, 0), (0, 0, 1), x1 - x0 + 2 * bw, bw, bm)
                    mesh.rect((x0 - bw, floor - 12, z0), (0, 0, 1), (1, 0, 0), z1 - z0, bw, bm, flip=True)
                    mesh.rect((x1, floor - 12, z0), (0, 0, 1), (1, 0, 0), z1 - z0, bw, bm, flip=True)
            elif kind == "wall_band":
                side = o["wall"]
                a, b = o["span"]
                top, bottom = o["y"]
                line = self._wall_line(side, (a + b) / 2)
                d = SIDES[side]
                inset = o.get("inset", 20)
                wall_rect(mesh, side, line + (d[0] + d[2]) * inset, a, b, bottom, top, self.use(o.get("material", "panel")))
                if o.get("rail", True):
                    skirting_box(mesh, side, line, a, b, 90, 60, self.use("wood"), floor=top + 90)
            elif kind == "table":
                x0, z0, x1, z1 = o["box"]
                top, thick = o["top"], o.get("thickness", 120)
                mat = self.use(o.get("material", "wood"))
                mesh.box((x0, top, z0), (x1, top + thick, z1), mat)
                leg = o.get("leg", 160)
                for lx, lz in ((x0 + 200, z0 + 200), (x1 - 200 - leg, z0 + 200), (x0 + 200, z1 - 200 - leg), (x1 - 200 - leg, z1 - 200 - leg)):
                    mesh.box((lx, top + thick, lz), (lx + leg, floor, lz + leg), mat, omit=("bottom", "top"))
                if o.get("with") == "typewriter":
                    tx, tz = o.get("typewriter_at", ((x0 + x1) / 2, (z0 + z1) / 2))
                    enamel, paper, iron = self.use("enamel"), self.use("paper"), self.use("iron")
                    mesh.box((tx - 300, top - 90, tz - 260), (tx + 290, top, tz + 200), enamel)
                    mesh.box((tx - 280, top - 220, tz + 10), (tx + 270, top - 90, tz + 190), enamel)
                    for row in range(3):
                        for column in range(9):
                            mesh.cylinder(tx - 230 + column * 55 + (row % 2) * 14, tz - 230 + row * 70, 18, top - 115 - row * 12, top - 90, paper, segments=6)
                    mesh.box((tx - 300, top - 250, tz + 90), (tx + 300, top - 195, tz + 160), self.use("wood"))
                    mesh.poly([(tx - 190, top - 180, tz + 40), (tx - 190, top - 560, tz + 80), (tx + 160, top - 560, tz + 80), (tx + 160, top - 180, tz + 40)], paper)
            elif kind == "chair":
                x, z = o["at"]
                facing = o.get("facing", "north")
                wood, seat = self.use(o.get("material", "wood")), self.use(o.get("seat", "velvet"))
                half, seat_y = o.get("half", 240), o.get("seat_y", -480)
                mesh.box((x - half, seat_y, z - half), (x + half, seat_y + 60, z + half), seat)
                for lx, lz in ((x - half + 20, z - half + 20), (x + half - 60, z - half + 20), (x - half + 20, z + half - 60), (x + half - 60, z + half - 60)):
                    mesh.box((lx, seat_y + 60, lz), (lx + 40, floor, lz + 40), wood, omit=("bottom", "top"))
                back_top = o.get("back", -1050)
                if facing == "north":
                    mesh.box((x - half, back_top, z - half), (x + half, seat_y, z - half + 50), wood)
                elif facing == "south":
                    mesh.box((x - half, back_top, z + half - 50), (x + half, seat_y, z + half), wood)
                elif facing == "east":
                    mesh.box((x - half, back_top, z - half), (x - half + 50, seat_y, z + half), wood)
                else:
                    mesh.box((x + half - 50, back_top, z - half), (x + half, seat_y, z + half), wood)
            elif kind == "fireplace":
                side = o["wall"]
                a, b = o["span"]
                depth, mantel = o.get("depth", 700), o.get("mantel", -1500)
                line = self._wall_line(side, (a + b) / 2)
                d = SIDES[side]
                stone, soot = self.use(o.get("material", "stone")), self.use("soot")
                ow, oh = o.get("opening", (1400, 1100))
                if side in ("west", "east"):
                    x0, x1 = (line, line + depth) if d[0] > 0 else (line - depth, line)
                    mesh.box((x0, mantel, a), (x1, floor, b), stone, omit=("bottom", side))
                    mesh.box((x0 - 60 * (d[0] < 0), mantel - 150, a - 120), (x1 + 60 * (d[0] > 0), mantel, b + 120), stone)
                    cz = (a + b) / 2
                    fx = x1 if d[0] > 0 else x0
                    mesh.box((min(fx, fx - d[0] * 300) , -oh, cz - ow / 2), (max(fx, fx - d[0] * 300) + 1, floor, cz + ow / 2), soot)
                else:
                    z0, z1 = (line, line + depth) if d[2] > 0 else (line - depth, line)
                    mesh.box((a, mantel, z0), (b, floor, z1), stone, omit=("bottom", side))
                    mesh.box((a - 120, mantel - 150, z0 - 60 * (d[2] < 0)), (b + 120, mantel, z1 + 60 * (d[2] > 0)), stone)
                    cx = (a + b) / 2
                    fz = z1 if d[2] > 0 else z0
                    mesh.box((cx - ow / 2, -oh, min(fz, fz - d[2] * 300)), (cx + ow / 2, floor, max(fz, fz - d[2] * 300) + 1), soot)
            elif kind == "urn":
                x, z = o["at"]
                r, top = o.get("radius", 500), o.get("top", -1600)
                ped = o.get("pedestal", 600)
                stone, mat = self.use("stone"), self.use(o.get("material", "iron"))
                ped_top = o.get("pedestal_top", -900)
                mesh.box((x - ped / 2, ped_top, z - ped / 2), (x + ped / 2, floor, z + ped / 2), stone, omit=("bottom",))
                mesh.cylinder(x, z, r * 0.55, ped_top - 120, ped_top, mat, segments=10)
                mesh.cylinder(x, z, r, top + (ped_top - top) * 0.35, ped_top - 120, mat, segments=12)
                mesh.cylinder(x, z, r * 0.6, top, top + (ped_top - top) * 0.35, mat, segments=12)
            elif kind == "sofa":
                x0, z0, x1, z1 = o["box"]
                back = o.get("back", "north")
                seat_y, back_top = o.get("seat_y", -480), o.get("back_top", -1000)
                fabric, wood = self.use(o.get("material", "velvet")), self.use("wood")
                mesh.box((x0, seat_y, z0), (x1, floor, z1), fabric, omit=("bottom",))
                t = 160
                if back == "north":
                    mesh.box((x0, back_top, z1 - t), (x1, seat_y, z1), fabric)
                elif back == "south":
                    mesh.box((x0, back_top, z0), (x1, seat_y, z0 + t), fabric)
                elif back == "west":
                    mesh.box((x0, back_top, z0), (x0 + t, seat_y, z1), fabric)
                else:
                    mesh.box((x1 - t, back_top, z0), (x1, seat_y, z1), fabric)
                for ax0, ax1 in ((x0, x0 + t), (x1 - t, x1)):
                    if back in ("north", "south"):
                        mesh.box((ax0, seat_y - 250, z0), (ax1, seat_y, z1), wood)
            elif kind == "lamp":
                x, z = o["at"]
                base = o.get("base", floor)
                h = o.get("height", 650)
                brass, shade = self.use("brass"), self.use("shade")
                mesh.cylinder(x, z, 90, base - 40, base, brass, segments=8)
                mesh.cylinder(x, z, 18, base - h + 180, base - 40, brass, segments=6, caps=False)
                mesh.cylinder(x, z, o.get("shade_radius", 150), base - h, base - h + 200, shade, segments=10)
            elif kind == "armor":
                x, z = o["at"]
                base = o.get("base", floor)
                steel, iron = self.use("armor"), self.use("iron")
                for dx in (-110, 110):
                    mesh.box((x + dx - 70, base - 900, z - 70), (x + dx + 70, base, z + 70), steel, omit=("bottom",))
                mesh.box((x - 230, base - 1550, z - 140), (x + 230, base - 900, z + 140), steel)
                mesh.box((x - 330, base - 1500, z - 110), (x - 230, base - 1000, z + 110), steel)
                mesh.box((x + 230, base - 1500, z - 110), (x + 330, base - 1000, z + 110), steel)
                mesh.sphere((x, base - 1720, z), 150, steel, rings=4, segments=8)
                if o.get("spear", True):
                    mesh.box((x + 360, base - 2600, z - 20), (x + 400, base, z + 20), iron)
                    mesh.box((x + 345, base - 2850, z - 40), (x + 415, base - 2600, z + 40), steel)
            elif kind == "rail":
                (ax, az), (bx, bz) = o["from"], o["to"]
                y = o.get("y", floor)
                balustrade(mesh, (ax, y, az), (bx, y, bz), height=o.get("height", 950), spacing=o.get("spacing", 420),
                           newels=tuple(o.get("newels", (True, True))), material=self.use(o.get("material", "wood")))
            elif kind == "bed":
                x0, z0, x1, z1 = o["box"]
                top = o.get("top", -620)
                head = o.get("head", "north")
                linen, frame = self.use(o.get("material", "linen")), self.use(o.get("frame", "iron"))
                mesh.box((x0, top, z0), (x1, floor, z1), linen, omit=("bottom", head))
                hb = o.get("headboard", -1100)
                if head == "north":
                    mesh.box((x0, hb, z1 - 60), (x1, top, z1), frame)
                elif head == "south":
                    mesh.box((x0, hb, z0), (x1, top, z0 + 60), frame)
                elif head == "west":
                    mesh.box((x0, hb, z0), (x0 + 60, top, z1), frame)
                else:
                    mesh.box((x1 - 60, hb, z0), (x1, top, z1), frame)
            elif kind == "canopy_bed":
                x0, z0, x1, z1 = o["box"]
                top = o.get("top", -700)
                linen, wood, cloth = self.use("linen"), self.use("wood"), self.use(o.get("drape", "velvet"))
                mesh.box((x0, top, z0), (x1, floor, z1), linen, omit=("bottom",))
                canopy = o.get("canopy", -3200)
                for px, pz in ((x0, z0), (x1 - 120, z0), (x0, z1 - 120), (x1 - 120, z1 - 120)):
                    mesh.box((px, canopy, pz), (px + 120, floor, pz + 120), wood, omit=("bottom",))
                mesh.box((x0 - 60, canopy - 200, z0 - 60), (x1 + 60, canopy, z1 + 60), cloth)
                for px in (x0, x1 - 120):
                    mesh.box((px - 40, canopy, z0 - 60), (px + 160, top - 200, z0 + 260), cloth)
                    mesh.box((px - 40, canopy, z1 - 260), (px + 160, top - 200, z1 + 60), cloth)
            elif kind == "newel":
                x, z = o["at"]
                wood = self.use("wood")
                mesh.box((x - 100, o.get("top", -1200), z - 100), (x + 100, floor, z + 100), wood, omit=("bottom",))
                mesh.sphere((x, o.get("top", -1200) - 100, z), 110, wood, rings=4, segments=8)
            else:
                raise ValueError(f"unknown object kind {kind}")
            mesh.set_group(None)
        for post in self.reading.get("lamp_post", []):
            mesh.set_group(post.get("group"))
            self._lamp_post(post)
            mesh.set_group(None)

    def _chest(self, o):
        mesh, floor = self.mesh, self.floor
        x0, z0, x1, z1 = o["box"]
        top = o["top"]
        back = o.get("back", "south")
        chest, iron, brass = self.use(o.get("material", "chest")), self.use("iron"), self.use("brass")
        mesh.box((x0, top, z0), (x1, floor, z1), chest, omit=("bottom", back))
        front_z = z1 if back == "south" else z0
        dz = 25 if back == "south" else -25
        for y in (top + 40, -120):
            mesh.box((x0 - 10, y, min(front_z, front_z + dz)), (x1 + 10, y + 70, max(front_z, front_z + dz)), iron)
            for x in (x0 - 10, x1 - 20):
                mesh.box((x, y, z0), (x + 30, y + 70, z1), iron)
        for x in (x0 + 260, x1 - 320):
            mesh.box((x, top, min(front_z, front_z + dz)), (x + 60, floor, max(front_z, front_z + dz)), iron)
        cx = (x0 + x1) / 2
        mesh.box((cx - 120, -1000, min(front_z, front_z + 1.6 * dz)), (cx + 120, -760, max(front_z, front_z + 1.6 * dz)), brass)

    def _table(self, o):
        mesh, floor = self.mesh, self.floor
        x, z = o["at"]
        r, top = o.get("radius", 450), o.get("top", -1550)
        wood = self.use("wood")
        mesh.cylinder(x, z, 330, -110, floor, wood, segments=12)
        mesh.cylinder(x, z, 60, top + 70, -110, wood, segments=8, caps=False)
        mesh.cylinder(x, z, r, top, top + 70, wood, segments=14)
        if o.get("with") == "typewriter":
            enamel, paper, iron = self.use("enamel"), self.use("paper"), self.use("iron")
            mesh.box((x - 300, top - 90, z - 260), (x + 290, top, z + 200), enamel)
            mesh.box((x - 280, top - 220, z + 10), (x + 270, top - 90, z + 190), enamel)
            for row in range(3):
                for column in range(9):
                    mesh.cylinder(x - 230 + column * 55 + (row % 2) * 14, z - 230 + row * 70, 18, top - 115 - row * 12, top - 90, paper, segments=6)
            mesh.box((x - 160, top - 115, z - 258), (x + 140, top - 90, z - 232), paper)
            mesh.box((x - 300, top - 250, z + 90), (x + 300, top - 195, z + 160), wood)
            for kx in (x - 340, x + 300):
                mesh.box((kx, top - 258, z + 80), (kx + 40, top - 187, z + 170), iron)
            mesh.poly([(x - 190, top - 180, z + 40), (x - 190, top - 560, z + 80), (x + 160, top - 560, z + 80), (x + 160, top - 180, z + 40)], paper)

    def _lamp_post(self, o):
        mesh, floor = self.mesh, self.floor
        x, z = o["at"]
        top = o.get("top", -4600)
        stone, brass, glow = self.use("stone"), self.use("brass"), self.use("glow")
        mesh.box((x - 330, -500, z - 330), (x + 330, floor, z + 330), stone, omit=("bottom",))
        mesh.box((x - 220, top + 1000, z - 220), (x + 220, -500, z + 220), stone)
        mesh.box((x - 300, top + 800, z - 300), (x + 300, top + 1000, z + 300), stone)
        mesh.cylinder(x, z, 40, top + 300, top + 800, brass, segments=6)
        mesh.sphere((x, top, z), 230, glow, rings=4, segments=8)

    def beams(self):
        wood = self.use("wood")
        for b in self.reading.get("beam", []):
            x0, z0, x1, z1 = b["box"]
            self.mesh.set_group(b.get("group", "ceiling"))
            self.mesh.box((x0, b["top"], z0), (x1, b["top"] + b.get("depth", 170), z1), wood, omit=("top",))
            self.mesh.set_group(None)

    def hides_text(self):
        lines = [f"hide {c} {g}" for c, g in self.auto_hides]
        for h in self.reading.get("hide", []):
            for g in h["groups"]:
                lines.append(f"hide {h['camera']} {g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def build(self):
        self.shell()
        self.beams()
        self.doors()
        self.windows()
        self.paintings()
        self.arcades()
        self.slabs()
        self.stairs()
        self.objects()
        self.sconces()
        self.chandeliers()
        return self.mesh


# ------------------------------------------------------------------ survey --
def survey_report(rdt, originals=None):
    """Evidence dump for writing a reading: cameras, collision, doors, masks."""
    from room3d_preview import camera_basis
    room = survey(rdt)
    data = rdt.read_bytes()
    print(f"# {rdt.name}: {len(room['cameras'])} cameras")
    for c in room["cameras"]:
        print(f"camera {c['index']}: origin {c['origin']} target {c['target']} fov {c['fov']}")
    for b in room["boundaries"]:
        kind = {1: "block", 3: "circle", 4: "stair?", 5: "slope?"}.get(b["type"], str(b["type"]))
        extra = f" centre ({(b['xmin'] + b['xmax']) // 2}, {(b['zmin'] + b['zmax']) // 2}) r~{(b['xmax'] - b['xmin']) // 2}" if b["type"] == 3 else ""
        print(f"collision {b['index']:2d} {kind:7s} x[{b['xmin']},{b['xmax']}] z[{b['zmin']},{b['zmax']}] flags {b['flags']}{extra}")
    # mask sprites floor-cast
    for c, cam in enumerate(room["cameras"]):
        base = 0x94 + c * 0x2C
        mask_ptr = struct.unpack_from("<i", data, base)[0]
        if not mask_ptr:
            continue
        count = struct.unpack_from("<i", data, mask_ptr)[0]
        g = mask_ptr + 4
        groups = []
        for _ in range(count):
            groups.append(struct.unpack_from("<HHhh", data, g))
            g += 8
        rec = g
        B = camera_basis(cam)
        o = np.array(cam["origin"], float)
        inv = np.linalg.inv(B)
        f = cam["fov"]
        for gi, (sc, bits, ox, oy) in enumerate(groups):
            rows = []
            for _ in range(sc):
                w0, w1, posdata, flags = struct.unpack_from("<HHHH", data, rec)
                rec += 8
                if (flags & 0xF000) == 0:
                    w, h = struct.unpack_from("<HH", data, rec)
                    rec += 4
                else:
                    w = h = (flags & 0xF1FF) >> 9
                rows.append(((w1 & 0xFF) + ox - 160, (w1 >> 8) + oy - 120, w, h))
            xs0, xs1 = min(r[0] for r in rows), max(r[0] + r[2] for r in rows)
            ys0, ys1 = min(r[1] for r in rows), max(r[1] + r[3] for r in rows)
            bottom = [r for r in rows if r[1] + r[3] == ys1]
            bx = (min(r[0] for r in bottom) + max(r[0] + r[2] for r in bottom)) / 2
            ray = inv @ np.array([bx / f, ys1 / f, 1.0])
            where = ""
            if ray[1] > 1e-6:
                p = o + ray * (-o[1] / ray[1])
                depth = (B @ (p - o))[2]
                where = f" -> floor ({p[0]:.0f}, {p[2]:.0f}) width~{(xs1 - xs0) / f * depth:.0f} top~{-(ys1 - ys0) / f * depth:.0f}"
            print(f"mask cam{c} grp{gi + 1} px x[{xs0 + 160},{xs1 + 160}] y[{ys0 + 120},{ys1 + 120}]{where}")
    if originals:
        from batch_room3d import background
        originals = Path(originals)
        originals.mkdir(parents=True, exist_ok=True)
        stage_dir = rdt.parent
        for c in room["cameras"]:
            name = f"rc{rdt.stem[4:7].lower()}{c["index"]}.pak"
            hits = [p for p in stage_dir.iterdir() if p.name.lower() == name]
            if hits:
                image = background(hits[0]).resize((960, 720), Image.NEAREST)
                draw = ImageDraw.Draw(image)
                for x in range(0, 320, 20):
                    draw.line((x * 3, 0, x * 3, 719), fill=(0, 90, 90), width=1)
                    draw.text((x * 3 + 2, 2), str(x), fill=(255, 255, 0))
                for y in range(0, 240, 20):
                    draw.line((0, y * 3, 959, y * 3), fill=(0, 90, 90), width=1)
                    draw.text((2, y * 3 + 2), str(y), fill=(255, 255, 0))
                image.save(originals / f"camera{c['index']}-grid.png")
                background(hits[0]).save(originals / f"camera{c['index']}-original.png")
        print(f"originals and grids written to {originals}")


def build_reading(path, rdt, out, originals=None):
    reading = tomllib.loads(Path(path).read_text())
    builder = Builder(reading, rdt)
    mesh = builder.build()
    out.mkdir(parents=True, exist_ok=True)
    name = reading["room"].lower()
    (out / f"{name}.obj").write_text("\n".join(mesh.lines) + "\n", encoding="ascii")
    (out / f"{name}.cameras").write_text(builder.hides_text())
    write_mtl(out, name, builder.palette.spec)
    builder.palette.write_textures(out, reading.get("seed", 1))
    size = (out / f"{name}.obj").stat().st_size
    print(f"{out / (name + '.obj')}: {mesh.triangles} triangles, {mesh.vertex_count} vertices, {size} bytes")
    if size > 4 * 1024 * 1024 or mesh.triangles > 50000 or mesh.vertex_count > 100000:
        raise SystemExit("exceeds the Room3D loader limits")
    cameras = len(builder.room["cameras"])
    preview(out, name, rdt)
    if originals:
        print(compare_sheet(out, name, originals, cameras))
        for camera, precision, recall in edge_agreement(out, name, originals, cameras):
            print(f"camera {camera}: edge precision {precision:.2f} recall {recall:.2f}")
    return out / f"{name}.obj"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("survey", help="print the evidence for a room and write gridded originals")
    s.add_argument("rdt", type=Path)
    s.add_argument("--originals", type=Path, help="write camera*-original.png and camera*-grid.png here")
    b = sub.add_parser("build", help="build a mesh from a reading")
    b.add_argument("reading", type=Path)
    b.add_argument("--rdt", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--originals", type=Path, help="camera*-original.png for compare.png and edge scores")
    args = parser.parse_args()
    if args.command == "survey":
        survey_report(args.rdt, args.originals)
    else:
        build_reading(args.reading, args.rdt, args.out, args.originals)


if __name__ == "__main__":
    main()
