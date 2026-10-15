"""Compare dense view union and free-space filtering, keeping original cameras.

This is a diagnostic experiment, not a watertight scene or runtime-ready export.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_preview import camera_basis, render
from room3d_refine import load_depth


def view_surface(depth, camera, stride=4):
    ys = np.unique(np.r_[np.arange(0, 240, stride), 239])
    xs = np.unique(np.r_[np.arange(0, 320, stride), 319])
    xx, yy = np.meshgrid(xs, ys)
    z = depth[yy, xx]
    rays = np.stack(
        ((xx - 160) / camera["fov"], (yy - 120) / camera["fov"], np.ones_like(z)), -1
    )
    points = (rays * z[..., None]) @ camera_basis(camera) + camera["origin"]
    a = (np.arange(len(ys) - 1)[:, None] * len(xs) + np.arange(len(xs) - 1)).ravel()
    faces = np.concatenate(
        (
            np.stack((a, a + 1, a + len(xs)), 1),
            np.stack((a + 1, a + len(xs) + 1, a + len(xs)), 1),
        )
    )
    ds = z.ravel()[faces]
    faces = faces[ds.max(1) / ds.min(1) <= 1.08]
    return (
        points.reshape(-1, 3),
        np.column_stack((xx.ravel() / 319, 1 - yy.ravel() / 239)),
        faces,
    )


def export_union(room, depths, images, output, prune=False, exclude=None):
    lines = ["mtllib combined.mtl"]
    materials = []
    offset = 0
    counts = []
    for i, (camera, (depth, _), image) in enumerate(
        zip(room["cameras"], depths, images, strict=True)
    ):
        if i == exclude:
            continue
        points, uv, faces = view_surface(depth, camera)
        before = len(faces)
        if prune:
            centers = points[faces].mean(1)
            rejected = np.zeros(len(faces), bool)
            for j, (other, (target, valid)) in enumerate(
                zip(room["cameras"], depths, strict=True)
            ):
                if j == i or j == exclude:
                    continue
                view = (centers - other["origin"]) @ camera_basis(other).T
                z = view[:, 2]
                xy = view[:, :2] / np.maximum(z[:, None], 1e-6) * other["fov"] + [
                    160,
                    120,
                ]
                pixels = np.rint(xy).astype(int)
                inside = (
                    (z > 2 * other["fov"])
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < 320)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < 240)
                )
                ids = np.flatnonzero(inside)
                x, y = pixels[ids].T
                # A surface behind another view's foreground is allowed.
                # A surface in that view's confident free space is inconsistent.
                rejected[ids] |= valid[y, x] & (z[ids] < target[y, x] * 0.92)
            faces = faces[~rejected]
        lines.extend([f"g view{i}", f"usemtl view{i}"])
        lines.extend("v " + " ".join(f"{x:.5f}" for x in point) for point in points)
        lines.extend(f"vt {u:.8f} {v:.8f}" for u, v in uv)
        lines.extend(
            "f "
            + " ".join(f"{int(v) + offset + 1}/{int(v) + offset + 1}" for v in face)
            for face in faces
        )
        offset += len(points)
        materials.extend([f"newmtl view{i}", "Kd 1 1 1", f"map_Kd source{i}.png"])
        image.save(output.parent / f"source{i}.png")
        counts.append(
            {
                "camera": camera["index"],
                "triangles_before": before,
                "triangles_after": len(faces),
            }
        )
    output.write_text("\n".join(lines) + "\n")
    (output.parent / "combined.mtl").write_text("\n".join(materials) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        "<h1>Prueba multivista de superficies densas</h1>",
        "<p>Original → método anterior → unión de vistas → unión filtrada. Todas conservan las cámaras originales. La unión no implica una malla cerrada; puede contener superficies duplicadas.</p>",
        "<p>La columna excluida omite la superficie y textura de esa cámara. DA3 estimó profundidad conjuntamente: tampoco es una validación independiente completa.</p>",
    ]
    records = []
    for key in ("room1180", "room10b0"):
        source = args.input / key
        out = args.out / key
        out.mkdir(exist_ok=True)
        room = json.loads((source / "survey.json").read_text())
        images = [
            Image.open(source / f"camera{c['index']}-original.png").convert("RGB")
            for c in room["cameras"]
        ]
        depths = load_depth(
            args.input / "depth" / key, source / "survey.json", room, images
        )
        raw = out / "union.obj"
        filtered = out / "filtered.obj"
        export_union(room, depths, images, raw)
        stats = export_union(room, depths, images, filtered, True)
        record = {"room": key, "surfaces": stats, "views": []}
        rows.append(f"<h2>{key.upper()}</h2>")
        for i, camera in enumerate(room["cameras"]):
            excluded = out / f"excluded{i}.obj"
            export_union(room, depths, images, excluded, True, exclude=i)
            # Restore complete material declarations after excluded-view export.
            # All meshes share identical material names and unchanged source PNGs.
            export_union(room, depths, images, filtered, True)
            panels = [("original", images[i], np.ones((240, 320), bool))]
            for label, mesh in [
                ("anterior", source / "assisted.obj"),
                ("union", raw),
                ("filtrada", filtered),
                ("excluida", excluded),
            ]:
                image, mask = render(mesh, camera)
                panels.append((label, image, mask))
            rows.append(f'<h3>Cámara {i}</h3><div class="row">')
            for label, image, mask in panels:
                name = f"camera{i}-{label}.png"
                image.save(out / name)
                rows.append(
                    f'<figure><figcaption>{label}</figcaption><img src="{key}/{name}"></figure>'
                )
                if label != "original":
                    difference = np.abs(
                        np.asarray(image, float) - np.asarray(images[i], float)
                    )
                    record["views"].append(
                        {
                            "camera": i,
                            "method": label,
                            "coverage": float(mask.mean()),
                            "mae_all_pixels": float(difference.mean()),
                            "qualification": "Appearance error; own-view textures reused except excluded surface. Not geometry accuracy.",
                        }
                    )
            rows.append("</div>")
        shifted = copy.deepcopy(room["cameras"][1])
        offset = camera_basis(shifted)[0] * np.median(depths[1][0]) * 0.15
        shifted["origin"] = (np.asarray(shifted["origin"]) + offset).tolist()
        shifted["target"] = (np.asarray(shifted["target"]) + offset).tolist()
        shifted["index"] = -1
        rows.append(
            '<h3>Cámara desplazada: sin imagen de referencia</h3><div class="row">'
        )
        for label, mesh in [
            ("anterior", source / "assisted.obj"),
            ("union", raw),
            ("filtrada", filtered),
        ]:
            image, mask = render(mesh, shifted)
            name = f"shifted-{label}.png"
            image.save(out / name)
            rows.append(
                f'<figure><figcaption>{label}</figcaption><img src="{key}/{name}"></figure>'
            )
        rows.append("</div>")
        record["shifted_camera"] = shifted
        records.append(record)
    (args.out / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Prueba multivista</title><style>body{background:#151920;color:#eee;font:16px system-ui;margin:24px}.row{display:flex;gap:8px;overflow:auto}figure{margin:4px;flex:none}img{width:320px;image-rendering:pixelated}</style>'
        + "".join(rows)
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
