"""Export predicted camera-space surfaces at RDT poses and render a comparison."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_preview import camera_basis, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()
    out = args.predictions
    room = json.loads((args.source / "survey.json").read_text())
    lines, materials, count = ["mtllib mapanything.mtl"], [], 0
    counts = []
    for i, camera in enumerate(room["cameras"]):
        with np.load(out / f"camera{i}.npz", allow_pickle=False) as data:
            points = data["pts3d_cam"][0]
            valid = data["mask"][0, ..., 0] > 0
            rgb = data["img_no_norm"][0]
        height, width = valid.shape
        Image.fromarray(np.uint8(np.clip(rgb * 255, 0, 255))).save(
            out / f"texture{i}.png"
        )
        # Match baseline sampling density; retain raw full-resolution predictions separately.
        yy, xx = np.mgrid[0:height:4, 0:width:4]
        sampled = points[yy, xx]
        h, w = yy.shape
        a = (np.arange(h - 1)[:, None] * w + np.arange(w - 1)).ravel()
        faces = np.concatenate(
            [np.stack([a, a + 1, a + w], 1), np.stack([a + 1, a + w + 1, a + w], 1)]
        )
        depth = sampled[..., 2].ravel()
        finite = np.isfinite(sampled).all(-1).ravel() & (depth > 0)
        keep = (valid[yy, xx].ravel() & finite)[faces].all(1)
        ds = depth[faces]
        keep &= ds.max(1) <= ds.min(1) * 1.08
        faces = faces[keep]
        world = sampled.reshape(-1, 3) @ camera_basis(camera) * 1000 + camera["origin"]
        world = np.nan_to_num(world)
        lines += [f"g camera{i}", f"usemtl camera{i}"]
        lines += ["v " + " ".join(f"{v:.6f}" for v in p) for p in world]
        lines += [
            f"vt {x / (width - 1):.9f} {1 - y / (height - 1):.9f}"
            for x, y in zip(xx.ravel(), yy.ravel(), strict=True)
        ]
        lines += [
            "f " + " ".join(f"{v + count + 1}/{v + count + 1}" for v in face)
            for face in faces
        ]
        count += len(world)
        counts.append(len(faces))
        materials += [f"newmtl camera{i}", "Kd 1 1 1", f"map_Kd texture{i}.png"]
    mesh = out / "mapanything.obj"
    mesh.write_text("\n".join(lines) + "\n")
    (out / "mapanything.mtl").write_text("\n".join(materials) + "\n")
    html = [
        "<!doctype html><meta charset='utf-8'><title>ROOM10B MapAnything</title>",
        "<style>body{background:#181818;color:#eee;font-family:sans-serif} .row{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:480px;max-width:95vw}</style>",
        "<h1>ROOM10B: MapAnything local</h1>",
        (
            "<p>Geometría estimada con imágenes y cámaras, sin profundidad DA3. "
            "Superficies colocadas con poses RDT originales. Texturas reutilizadas de las entradas: "
            "una buena vista original no demuestra exactitud 3D. No es una habitación cerrada.</p>"
        ),
        "<p><a href='manifest.json'>Mediciones de inferencia</a> · <a href='mapanything.obj'>OBJ</a></p>",
    ]
    records = []
    cameras = list(room["cameras"])
    moved = copy.deepcopy(cameras[1])
    offset = camera_basis(moved)[0] * 1000
    for k in ["origin", "target"]:
        moved[k] = (np.asarray(moved[k]) + offset).tolist()
    moved["index"] = -1
    cameras.append(moved)
    for i, camera in enumerate(cameras):
        html.append(
            f"<h2>{'Cámara desplazada, sin referencia' if camera['index'] == -1 else 'Cámara ' + str(i)}</h2><div class='row'>"
        )
        if camera["index"] >= 0:
            Image.open(args.source / f"camera{i}-original.png").save(
                out / f"original{i}.png"
            )
            html.append(
                f"<figure><figcaption>Original</figcaption><img src='original{i}.png'></figure>"
            )
        for label, path in [("DA3-TSDF", args.baseline), ("MapAnything", mesh)]:
            image, mask = render(path, camera)
            name = f"view{i}-{label}.png"
            image.save(out / name)
            html.append(
                f"<figure><figcaption>{label}</figcaption><img src='{name}'></figure>"
            )
            records.append({"view": i, "method": label, "coverage": float(mask.mean())})
        html.append("</div>")
    (out / "index.html").write_text("\n".join(html))
    (out / "comparison.json").write_text(
        json.dumps({"triangles_per_view": counts, "renders": records}, indent=2)
    )
    print(out / "index.html")


if __name__ == "__main__":
    main()
