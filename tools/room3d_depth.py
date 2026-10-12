"""Local DA3 proposals: baseline-aligned scale, or an explicit monocular prior."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from PIL import Image
from room3d_preview import camera_basis

MODEL = "depth-anything/DA3-SMALL"
REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


def camera_inputs(room):
    extrinsics, intrinsics = [], []
    for camera in room["cameras"]:
        rotation = camera_basis(camera)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ camera["origin"]
        extrinsics.append(extrinsic)
        f = camera["fov"]
        intrinsics.append([[f, 0, 160], [0, f, 120], [0, 0, 1]])
    return np.asarray(extrinsics, np.float32), np.asarray(intrinsics, np.float32)


def single_view_scale(prediction, camera, extrinsics, intrinsics):
    """Apply a creative camera-target distance prior; never claim metric evidence.

    A single pose cannot establish trajectory scale. Its look-at target supplies
    a deterministic editing scale, not a measured surface or floor location.
    Keep relative depth and the original conditioned camera projection intact.
    """
    depth = np.asarray(prediction.depth, np.float32)
    if depth.ndim != 3 or len(depth) != 1 or not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("Expected one finite positive relative depth map")
    distance = float(np.linalg.norm(np.asarray(camera['target'], float) - camera['origin']))
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Camera target distance prior must be positive and finite")
    median = float(np.median(depth))
    factor = distance / median
    scaled = depth * factor
    if not np.isfinite(scaled).all() or not (scaled > 0).all():
        raise ValueError("Invalid scaled monocular proposal")
    prediction.depth = scaled
    prediction.extrinsics = np.asarray(extrinsics)[..., :3, :].copy()
    prediction.intrinsics = np.asarray(intrinsics).copy()
    return {
        "method": "camera_target_distance_prior", "factor": factor,
        "relative_depth_median": median, "target_distance": distance,
        "metric_scale_verified": False,
        "qualification": "Estimated editing scale only; camera target is not a measured surface. Hidden geometry, navigation and world-scale fidelity unverified",
    }


def run(survey, images_root, out):
    import torch
    from depth_anything_3.api import DepthAnything3

    torch.set_num_threads(4)
    room = json.loads(survey.read_text())
    paths = [images_root / f"camera{c['index']}-original.png" for c in room["cameras"]]
    images = [Image.open(p).convert("RGB") for p in paths]
    if any(im.size != (320, 240) for im in images):
        raise ValueError("Expected native 320x240 backgrounds")
    ext, intr = camera_inputs(room)

    class RoomDepthModel(DepthAnything3):
        def _align_to_input_extrinsics_intrinsics(self, extrinsics, intrinsics, prediction,
                                                 align_to_input_ext_scale=True, **kwargs):
            if extrinsics is not None and len(extrinsics) == 1:
                self.single_view_evidence = single_view_scale(
                    prediction, room['cameras'][0], extrinsics.numpy(), intrinsics.numpy(),
                )
                return prediction
            return super()._align_to_input_extrinsics_intrinsics(
                extrinsics, intrinsics, prediction, align_to_input_ext_scale, **kwargs,
            )

    # The pinned upstream API attempts Umeyama even with its scale flag off.
    # Override only the undefined singleton alignment, retaining camera
    # conditioning in the actual forward pass. No installed library is patched.
    model = RoomDepthModel.from_pretrained(MODEL, revision=REVISION).to("cpu").eval()
    started = time.monotonic()
    prediction = model.inference(
        image=images,
        extrinsics=ext,
        intrinsics=intr,
        align_to_input_ext_scale=True,
        process_res=336,
        process_res_method="upper_bound_resize",
    )
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (camera, image) in enumerate(zip(room["cameras"], images, strict=True)):
        depth = np.asarray(prediction.depth[i], np.float32)
        if not np.isfinite(depth).all() or not (depth > 0).all():
            raise ValueError("DA3 produced non-finite/non-positive depth")
        name = f"camera{camera['index']}"
        np.save(out / (name + "-depth.npy"), depth, allow_pickle=False)
        conf = getattr(prediction, "conf", None)
        if conf is not None:
            np.save(out / (name + "-confidence.npy"), conf[i], allow_pickle=False)
        lo, hi = np.percentile(depth, [2, 98])
        display = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
        Image.fromarray((display * 255).astype(np.uint8)).save(
            out / (name + "-depth.png")
        )
        records.append(
            {
                "camera": camera["index"],
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "shape": list(depth.shape),
                "percentiles": [float(lo), float(hi)],
                "intrinsics": prediction.intrinsics[i].tolist(),
                "extrinsics": prediction.extrinsics[i].tolist(),
            }
        )
    record = {
        "model": MODEL,
        "revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "device": "cpu",
        "seconds": time.monotonic() - started,
        "status": "unverified_depth_proposal",
        "scale": "aligned_to_input_camera_baselines",
        "survey_sha256": hashlib.sha256(survey.read_bytes()).hexdigest(),
        "views": records,
    }
    if len(room['cameras']) == 1:
        record['scale'] = 'estimated_from_camera_target_distance'
        record['scale_evidence'] = model.single_view_evidence
    (out / "depth.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.survey, args.images, args.out)
