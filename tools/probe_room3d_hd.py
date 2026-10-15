"""Controlled DA3 original/HD x processing-resolution experiment for ROOM10B."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image
from probe_room3d_multiview import export_union
from room3d_depth import MODEL, REVISION, camera_inputs
from room3d_preview import camera_basis, render

HD_NAMES = ["8CA9A873", "196082CE", "83C8E167", "C12E6487"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    survey = args.source / "survey.json"
    room = json.loads(survey.read_text())
    if room["source"].upper() != "ROOM10B0.RDT" or len(room["cameras"]) != 4:
        raise ValueError("This HD mapping is specific to ROOM10B0")
    original = [
        Image.open(args.source / f"camera{i}-original.png").convert("RGB")
        for i in range(4)
    ]
    hd = [Image.open(args.hd / (name + ".webp")).convert("RGB") for name in HD_NAMES]
    if any(im.size != (320, 240) for im in original) or any(
        im.size != (1280, 960) for im in hd
    ):
        raise ValueError("Unexpected dimensions")
    for i in range(4):
        original[i].save(args.out / f"original{i}.png")
        hd[i].save(args.out / f"hd{i}.png")
    ext, intr = camera_inputs(room)
    model = DepthAnything3.from_pretrained(MODEL, revision=REVISION).to("cpu").eval()
    experiments = {}
    for kind, images in [("original", original), ("hd", hd)]:
        for resolution in [336, 672]:
            name = f"{kind}-{resolution}"
            dest = args.out / name
            dest.mkdir()
            scaled_intr = intr.copy()
            scaled_intr[:, :2] *= images[0].width / 320
            torch.manual_seed(42)
            start = time.monotonic()
            pred = model.inference(
                image=images,
                extrinsics=ext.copy(),
                intrinsics=scaled_intr,
                align_to_input_ext_scale=True,
                process_res=resolution,
                process_res_method="upper_bound_resize",
            )
            depths = []
            for i in range(4):
                depth = np.asarray(pred.depth[i], np.float32)
                confidence = np.asarray(pred.conf[i], np.float32)
                if not np.isfinite(depth).all() or not (depth > 0).all():
                    raise ValueError("Invalid depth")
                np.savez_compressed(
                    dest / f"camera{i}.npz",
                    depth=depth,
                    confidence=confidence,
                    intrinsics=pred.intrinsics[i],
                    extrinsics=pred.extrinsics[i],
                )
                normalized = cv2.resize(
                    depth, (320, 240), interpolation=cv2.INTER_LINEAR
                )
                conf = cv2.resize(
                    confidence, (320, 240), interpolation=cv2.INTER_NEAREST
                )
                depths.append((normalized, conf >= np.percentile(conf, 40)))
            # Identical original textures isolate geometry differences from HD appearance.
            stats = export_union(
                room, depths, original, dest / "filtered.obj", prune=True
            )
            experiments[name] = {
                "seconds": time.monotonic() - start,
                "shape": list(pred.depth.shape),
                "surfaces": stats,
                "depths": depths,
            }
            print(name, experiments[name]["seconds"], flush=True)
    html = [
        "<!doctype html><meta charset='utf-8'><title>DA3: original vs HD</title>",
        "<style>body{background:#181818;color:#eee;font-family:sans-serif}.row{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:400px;max-width:94vw}</style>",
        "<h1>ROOM10B: ¿ayudan los fondos HD a DA3?</h1>",
        "<p>4 vistas juntas, mismas cámaras y modelo DA3-SMALL. Original/HD × resolución de proceso 336/672. El pack HD también cambia iluminación y color.</p>",
        "<p>Todos los renders usan las texturas originales para aislar geometría. Son superficies filtradas por visibilidad, no una habitación cerrada ni el resultado completo del batcher.</p>",
    ]
    cameras = list(room["cameras"])
    moved = copy.deepcopy(cameras[1])
    offset = camera_basis(moved)[0] * 1000
    for key in ["origin", "target"]:
        moved[key] = (np.asarray(moved[key]) + offset).tolist()
    moved["index"] = -1
    cameras.append(moved)
    for i, camera in enumerate(cameras):
        html.append(
            f"<h2>{'Cámara desplazada (sin referencia)' if i == 4 else 'Cámara ' + str(i)}</h2>"
        )
        if i < 4:
            html.append(
                f"<div class='row'><figure>Entrada original<br><img src='original{i}.png'></figure><figure>Entrada HD<br><img src='hd{i}.png'></figure></div>"
            )
        html.append("<div class='row'>")
        for name, record in experiments.items():
            dest = args.out / name
            image, mask = render(dest / "filtered.obj", camera)
            image.save(dest / f"view{i}.png")
            record.setdefault("coverage", []).append(float(mask.mean()))
            html.append(f"<figure>{name}<br><img src='{name}/view{i}.png'></figure>")
        html.append("</div>")
        if i < 4:
            # Shared per-camera range across all runs; independently normalized plots mislead.
            arrays = [v["depths"][i][0] for v in experiments.values()]
            lo, hi = np.percentile(np.stack(arrays), [2, 98])
            html.append(
                "<details><summary>Profundidad, misma escala entre variantes</summary><div class='row'>"
            )
            for name, depth in zip(experiments, arrays, strict=True):
                gray = np.uint8(np.clip((depth - lo) / (hi - lo), 0, 1) * 255)
                Image.fromarray(gray).save(args.out / name / f"depth{i}.png")
                html.append(
                    f"<figure>{name}<br><img src='{name}/depth{i}.png'></figure>"
                )
            html.append("</div></details>")
    records = {
        name: {k: v for k, v in value.items() if k != "depths"}
        for name, value in experiments.items()
    }
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "seed": 42,
        "survey_sha256": hashlib.sha256(survey.read_bytes()).hexdigest(),
        "input_rgb_sha256": {
            kind: [hashlib.sha256(im.tobytes()).hexdigest() for im in ims]
            for kind, ims in [("original", original), ("hd", hd)]
        },
        "runs": records,
        "qualification": "Coverage is not geometry accuracy; no 3D ground truth.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    html.append("<p><a href='manifest.json'>Mediciones y procedencia</a></p>")
    (args.out / "index.html").write_text("\n".join(html))
    print(args.out / "index.html", flush=True)


if __name__ == "__main__":
    main()
