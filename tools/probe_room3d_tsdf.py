"""Local TSDF fusion experiment on estimated depth, with original RDT cameras."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import marching_cubes
from room3d_preview import camera_basis, render
from room3d_refine import load_depth


def project(points, camera):
    view = (points - camera["origin"]) @ camera_basis(camera).T
    z = view[:, 2]
    xy = view[:, :2] / np.maximum(z[:, None], 1e-6) * camera["fov"] + [160, 120]
    pixels = np.rint(xy).astype(int)
    valid = (
        (z > 2 * camera["fov"])
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < 320)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < 240)
    )
    return z, xy, pixels, valid


def fuse(room, depths, voxel=150):
    bounds = room["boundaries"]
    origin = np.array(
        [
            min(b["xmin"] for b in bounds) - voxel,
            -12000,
            min(b["zmin"] for b in bounds) - voxel,
        ]
    )
    upper = np.array(
        [
            max(b["xmax"] for b in bounds) + voxel,
            voxel,
            max(b["zmax"] for b in bounds) + voxel,
        ]
    )
    shape = np.ceil((upper - origin) / voxel).astype(int) + 1
    points = np.indices(shape, dtype=np.float32).reshape(3, -1).T * voxel + origin
    total = np.zeros(len(points), np.float32)
    weight = np.zeros(len(points), np.float32)
    trunc = voxel * 3
    for camera, (depth, confident) in zip(room["cameras"], depths, strict=True):
        z, _, pixels, visible = project(points, camera)
        ids = np.flatnonzero(visible)
        x, y = pixels[ids].T
        signed = depth[y, x] - z[ids]
        keep = signed >= -trunc
        ids, x, y, signed = ids[keep], x[keep], y[keep], signed[keep]
        weights = np.where(confident[y, x], 1.0, 0.15)
        total[ids] += np.clip(signed / trunc, -1, 1) * weights
        weight[ids] += weights
    field = np.ones(len(points), np.float32)
    known = weight > 0
    field[known] = total[known] / weight[known]
    vertices, faces, _, _ = marching_cubes(
        field.reshape(shape), level=0, spacing=(voxel,) * 3, mask=known.reshape(shape)
    )
    return (
        vertices + origin,
        faces,
        {
            "voxel": voxel,
            "truncation": trunc,
            "grid_shape": shape.tolist(),
            "known_voxels": int(known.sum()),
        },
    )


def export_mesh(vertices, faces, room, depths, images, out, voxel):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-8)
    centers = triangles.mean(1)
    scores = []
    coordinates = []
    for i, (camera, (depth, _), image) in enumerate(
        zip(room["cameras"], depths, images, strict=True)
    ):
        z, xy, pixels, valid = project(vertices, camera)
        error = np.full(len(vertices), np.inf)
        ids = np.flatnonzero(valid)
        x, y = pixels[ids].T
        error[ids] = np.abs(z[ids] - depth[y, x])
        face_error = error[faces].max(1)
        ray = centers - camera["origin"]
        ray /= np.maximum(np.linalg.norm(ray, axis=1)[:, None], 1e-8)
        angle = np.abs((normals * ray).sum(1))
        score = np.where(face_error < voxel * 3, angle / (1 + face_error / voxel), -1)
        scores.append(score)
        coordinates.append(xy / [319, 239])
        image.save(out.parent / f"source{i}.png")
    scores = np.asarray(scores)
    chosen = scores.argmax(0)
    supported = scores.max(0) >= 0
    materials = ["newmtl unknown", "Kd 0.25 0.3 0.35"]
    for i in range(len(images)):
        materials += [f"newmtl source{i}", "Kd 1 1 1", f"map_Kd source{i}.png"]
    out.with_suffix(".mtl").write_text("\n".join(materials) + "\n")
    lines = [f"mtllib {out.stem}.mtl"]
    for n, face in enumerate(faces):
        i = chosen[n]
        lines.append(f"usemtl source{i}" if supported[n] else "usemtl unknown")
        for vertex in face:
            lines.append("v " + " ".join(f"{x:.5f}" for x in vertices[vertex]))
            u, v = coordinates[i][vertex]
            lines.append(f"vt {u:.8f} {1 - v:.8f}")
        lines.append(
            "f " + " ".join(f"{n * 3 + j + 1}/{n * 3 + j + 1}" for j in range(3))
        )
    out.write_text("\n".join(lines) + "\n")
    return {
        "faces": len(faces),
        "textured_faces": int(supported.sum()),
        "unknown_faces": int((~supported).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        "<h1>Fusión volumétrica TSDF</h1><p>Profundidad estimada, cámaras originales. Resuelve algunas superposiciones mediante consenso; no recupera geometría verdadera. Las superficies sin textura compatible se muestran grises.</p>"
    ]
    records = []
    for key in ["room1180", "room10b0"]:
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
        vertices, faces, stats = fuse(room, depths)
        mesh = out / "fused.obj"
        stats.update(
            export_mesh(vertices, faces, room, depths, images, mesh, stats["voxel"])
        )
        rows.append(f"<h2>{key.upper()}</h2>")
        views = []
        for i, camera in enumerate(room["cameras"]):
            image, mask = render(mesh, camera)
            image.save(out / f"camera{i}.png")
            images[i].save(out / f"original{i}.png")
            rows.append(
                f'<h3>Cámara {i}: original / fusión</h3><img src="{key}/original{i}.png"><img src="{key}/camera{i}.png">'
            )
            views.append(
                {
                    "camera": i,
                    "coverage": float(mask.mean()),
                    "mae_all_pixels": float(
                        np.abs(
                            np.asarray(image, float) - np.asarray(images[i], float)
                        ).mean()
                    ),
                    "qualification": "Appearance uses these input views; not independent geometry accuracy",
                }
            )
        shifted = copy.deepcopy(room["cameras"][1])
        shifted["index"] = -1
        offset = camera_basis(shifted)[0] * np.median(depths[1][0]) * 0.15
        shifted["origin"] = (np.asarray(shifted["origin"]) + offset).tolist()
        shifted["target"] = (np.asarray(shifted["target"]) + offset).tolist()
        image, _ = render(mesh, shifted)
        image.save(out / "shifted.png")
        rows.append(f'<h3>Desplazada: sin referencia</h3><img src="{key}/shifted.png">')
        records.append(dict(room=key, views=views, shifted_camera=shifted, **stats))
    (args.out / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{background:#151920;color:#eee;font:16px system-ui;margin:24px}img{width:480px;image-rendering:pixelated}</style>'
        + "".join(rows)
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
