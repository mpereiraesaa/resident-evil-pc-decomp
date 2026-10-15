"""Small CPU depth-mesh experiment; no hidden-surface or generative-model claim."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_preview import camera_basis, render
from room3d_refine import load_depth


def export_dense(image, depth, camera, out, stride=2):
    """Keep source texture detail; reject triangles across depth discontinuities."""
    xs, ys = np.arange(0, 320, stride), np.arange(0, 240, stride)
    x, y = np.meshgrid(xs, ys)
    z = depth[y, x]
    view = np.stack(
        ((x - 160) * z / camera["fov"], (y - 120) * z / camera["fov"], z), -1
    )
    world = view @ camera_basis(camera) + camera["origin"]
    lines = ["mtllib dense.mtl", "g dense", "usemtl source"]
    for point, px, py in zip(world.reshape(-1, 3), x.flat, y.flat):
        lines.append("v " + " ".join(f"{v:.5f}" for v in point))
        lines.append(f"vt {px / 319:.8f} {1 - py / 239:.8f}")
    faces = rejected = 0
    width = len(xs)
    for row in range(len(ys) - 1):
        for col in range(width - 1):
            a = row * width + col
            for ids in ((a, a + 1, a + width), (a + 1, a + width + 1, a + width)):
                ds = z.flat[list(ids)]
                if ds.max() / ds.min() > 1.08:
                    rejected += 1
                    continue
                lines.append("f " + " ".join(f"{i + 1}/{i + 1}" for i in ids))
                faces += 1
    out.write_text("\n".join(lines) + "\n")
    (out.parent / "dense.mtl").write_text(
        "newmtl source\nKd 1 1 1\nmap_Kd source.png\n"
    )
    image.save(out.parent / "source.png")
    return {"faces": faces, "rejected_depth_edges": rejected, "stride": stride}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        "<h1>Prueba pequeña: malla densa desde profundidad</h1>",
        "<p>Esta prueba de malla densa corre en CPU y no ejecuta modelos generativos. Conserva superficies visibles; no reconstruye reversos ocultos. La prueba posterior de TRELLIS.2 cuantizado está en el informe de alternativas.</p>",
        "<p>La vista de partida reutiliza su propia imagen: NO es validación independiente. La otra cámara original y el desplazamiento muestran los límites.</p>",
    ]
    summaries = []
    for key in ["room1180", "room10b0"]:
        source = args.input / key
        dest = args.out / key
        dest.mkdir(exist_ok=True)
        room = json.loads((source / "survey.json").read_text())
        originals = [
            Image.open(source / f"camera{c['index']}-original.png").convert("RGB")
            for c in room["cameras"]
        ]
        depths = load_depth(
            args.input / "depth" / key, source / "survey.json", room, originals
        )
        index = 1
        camera = room["cameras"][index]
        depth = depths[index][0]
        mesh = dest / "dense.obj"
        stats = export_dense(originals[index], depth, camera, mesh)
        shifted = copy.deepcopy(camera)
        offset = camera_basis(camera)[0] * float(np.median(depth)) * 0.15
        shifted["origin"] = (np.asarray(camera["origin"]) + offset).tolist()
        shifted["target"] = (np.asarray(camera["target"]) + offset).tolist()
        shifted["index"] = -1
        dense, covered = render(mesh, camera)
        old, _ = render(source / "assisted.obj", camera)
        novel, _ = render(mesh, shifted)
        old_novel, _ = render(source / "assisted.obj", shifted)
        other = room["cameras"][0]
        heldout, _ = render(mesh, other)
        old_other, _ = render(source / "assisted.obj", other)
        panels = {
            "original": originals[index],
            "anterior": old,
            "densa": dense,
            "anterior-desplazada": old_novel,
            "densa-desplazada": novel,
            "otra-original": originals[0],
            "otra-anterior": old_other,
            "otra-densa": heldout,
        }
        for name, panel in panels.items():
            panel.save(dest / f"{name}.png")
        stats["same_view_mae"] = float(
            np.abs(np.asarray(dense, float) - np.asarray(originals[index], float))[
                covered
            ].mean()
        )
        stats["qualification"] = (
            "Same-view reconstruction error only; source appearance reused, not geometry accuracy"
        )
        stats["shifted_camera"] = shifted
        summaries.append(dict(room=key, **stats))
        rows.append(f'<h2>{key.upper()}</h2><div class="grid">')
        for name in panels:
            rows.append(
                f'<figure><figcaption>{name}</figcaption><img src="{key}/{name}.png"></figure>'
            )
        rows.append("</div>")
        crops = (
            [("baul", (65, 125, 175, 215)), ("esquina", (35, 0, 145, 130))]
            if key == "room1180"
            else [("lampara", (125, 10, 190, 85))]
        )
        for label, box in crops:
            rows.append(
                f'<h3>{label}: original / anterior / densa (misma cámara)</h3><div class="crops">'
            )
            for name in ["original", "anterior", "densa"]:
                filename = f"{label}-{name}.png"
                panels[name].crop(box).resize(
                    (330, 270), Image.Resampling.NEAREST
                ).save(dest / filename)
                rows.append(f'<img src="{key}/{filename}">')
            rows.append("</div>")
    (args.out / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Prueba densa</title><style>body{background:#151920;color:#eee;font:16px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr)}figure{margin:8px}img{max-width:100%;image-rendering:pixelated}.crops{display:flex;gap:12px}</style>'
        + "".join(rows)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
