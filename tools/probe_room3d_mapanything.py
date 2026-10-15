"""Local MapAnything diagnostic with known RE1 cameras; no DA3 depth input."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from room3d_preview import camera_basis


def main():
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=322)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    room = json.loads((args.source / "survey.json").read_text())
    views = []
    for camera in room["cameras"]:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = camera_basis(camera).T
        pose[:3, 3] = np.asarray(camera["origin"]) * 0.001
        f = camera["fov"]
        views.append(
            {
                "img": Image.open(
                    args.source / f"camera{camera['index']}-original.png"
                ).convert("RGB"),
                "intrinsics": torch.tensor(
                    [[f, 0, 160], [0, f, 120], [0, 0, 1]], dtype=torch.float32
                ),
                "camera_poses": torch.from_numpy(pose),
                "is_metric_scale": True,
            }
        )
    inputs = preprocess_inputs(views, resize_mode="longest_side", size=args.size)
    print("Loading model", flush=True)
    start = time.monotonic()
    model = MapAnything.from_pretrained(str(args.model)).eval()
    # Keep weights FP32: the upstream geometric-input path mixes indexed FP32
    # assignments with model dtype. AMP still reduces activation memory.
    model = model.to(device=args.device)
    print("Model loaded", flush=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Avoid retaining a second BF16 copy of every FP32 weight in the AMP cache.
    torch.set_autocast_cache_enabled(False)
    with torch.inference_mode():
        predictions = model.infer(
            inputs,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
    records = []
    for i, pred in enumerate(predictions):
        arrays = {
            k: v.detach().float().cpu().numpy()
            for k, v in pred.items()
            if torch.is_tensor(v)
        }
        np.savez_compressed(args.out / f"camera{i}.npz", **arrays)
        records.append({k: list(v.shape) for k, v in arrays.items()})
    manifest = {
        "model": str(args.model),
        "room": room["source"],
        "size": args.size,
        "device": args.device,
        "weight_dtype": "float32",
        "seed": 42,
        "depth_input": False,
        "camera_input": "RDT OpenCV cam2world; game units / 1000",
        "elapsed_seconds_including_model_load": time.monotonic() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if args.device == "cuda"
        else None,
        "outputs": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
