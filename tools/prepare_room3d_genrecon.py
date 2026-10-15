"""Export known RE1 cameras and estimated DA3 depth for GenRecon's Iphone loader.

The cloud is estimated, not SfM or ground truth. No network or job submission.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_preview import camera_basis
from room3d_refine import load_depth
from scipy.spatial.transform import Rotation

# Proper rotation: (X, Y, Z) -> (X, Z, -Y), with Z up.
WORLD_ROTATION = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def colmap_pose(camera, scale):
    rotation = camera_basis(camera) @ WORLD_ROTATION.T
    center = WORLD_ROTATION @ np.asarray(camera["origin"]) * scale
    translation = -rotation @ center
    quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    return quaternion, translation


def prepare(source, depth_root, output, scale=0.001, stride=4):
    if scale <= 0 or not np.isfinite(scale) or stride < 1:
        raise ValueError("Scale and stride must be positive")
    survey = source / "survey.json"
    room = json.loads(survey.read_text())
    images = [
        Image.open(source / f"camera{c['index']}-original.png").convert("RGB")
        for c in room["cameras"]
    ]
    if any(image.size != (320, 240) for image in images):
        raise ValueError("Expected original 320x240 backgrounds")
    depths = load_depth(depth_root, survey, room, images)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    (output / "colmap").mkdir()
    cameras, poses, clouds, colors = [], [], [], []
    max_error = 0.0
    for image_id, (camera, image, (depth, valid)) in enumerate(
        zip(room["cameras"], images, depths, strict=True), 1
    ):
        name = f"camera{camera['index']}.png"
        image.save(output / "rgb" / name)
        f = camera["fov"]
        cameras.append(f"{image_id} PINHOLE 320 240 {f} {f} 160 120")
        quaternion, translation = colmap_pose(camera, scale)
        values = " ".join(f"{v:.17g}" for v in [*quaternion, *translation])
        poses.append(f"{image_id} {values} {image_id} {name}\n")
        yy, xx = np.mgrid[0:240:stride, 0:320:stride]
        keep = valid[yy, xx]
        x, y = xx[keep], yy[keep]
        rays = np.column_stack(((x - 160) / f, (y - 120) / f, np.ones(len(x))))
        game_points = (rays * depth[y, x, None]) @ camera_basis(camera) + camera[
            "origin"
        ]
        points = game_points @ WORLD_ROTATION.T * scale
        projected = (
            points @ Rotation.from_quat(quaternion, scalar_first=True).as_matrix().T
            + translation
        )
        pixels = projected[:, :2] / projected[:, 2, None] * f + [160, 120]
        max_error = max(
            max_error, float(np.abs(pixels - np.column_stack((x, y))).max())
        )
        clouds.append(points)
        colors.append(np.asarray(image)[y, x])
    if max_error > 1e-6:
        raise ValueError(f"Camera round-trip failed: {max_error} pixels")
    (output / "colmap/cameras.txt").write_text("\n".join(cameras) + "\n")
    # COLMAP requires an empty POINTS2D row after each pose.
    (output / "colmap/images.txt").write_text("\n".join(poses) + "\n")
    points, rgb = np.concatenate(clouds), np.concatenate(colors)
    with (output / "colmap/points3D.txt").open("w") as stream:
        stream.write(
            "# Estimated DA3 samples; ERROR=0 is a format placeholder, no SfM tracks.\n"
        )
        for index, (point, color) in enumerate(zip(points, rgb, strict=True), 1):
            stream.write(
                f"{index} "
                + " ".join(f"{v:.17g}" for v in point)
                + " "
                + " ".join(str(int(v)) for v in color)
                + " 0\n"
            )
    manifest = {
        "room": room["source"],
        "views": len(images),
        "points": len(points),
        "world_rotation": WORLD_ROTATION.tolist(),
        "units_scale": scale,
        "scale_note": "Assumed 1000 game units per metre; not measured physical scale.",
        "cloud_source": "DA3 estimated depth, confidence top 60%, sampled; not ground truth",
        "max_camera_roundtrip_error_px": max_error,
        "files": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "survey_sha256": hashlib.sha256(survey.read_bytes()).hexdigest(),
        "depth_manifest_sha256": hashlib.sha256(
            (depth_root / "depth.json").read_bytes()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.depth, args.out), indent=2))
