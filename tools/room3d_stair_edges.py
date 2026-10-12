"""Propose stair-flight regions from repeated horizontal tread edges.

These are image/depth hypotheses, not ground-truth segmentations. Floorboards,
wall moldings and rail posts must fail the height/run/horizontal-edge tests.
"""

import cv2
import numpy as np
from room3d_preview import camera_basis

# Exact category phrases only: merged detector labels such as "staircase
# railing" or "stair treads staircase" do not establish a stair-flight category.
STAIR_LABELS = {"staircase", "carpeted staircase", "wooden staircase", "stone staircase"}


def backproject(xy, depth, camera):
    ij = np.rint(xy).astype(int)
    ij = np.clip(ij, [0, 0], [depth.shape[1] - 1, depth.shape[0] - 1])
    z = depth[ij[:, 1], ij[:, 0]]
    view = np.column_stack(
        ((xy[:, 0] - 160) * z / camera["fov"], (xy[:, 1] - 120) * z / camera["fov"], z)
    )
    return view @ camera_basis(camera) + camera["origin"]


def normalize_horizontal_axes(axes):
    """Validate explicit XZ layout axes; keep them separate from model confidence."""
    axes = np.asarray(axes, dtype=float)
    if (axes.ndim != 2 or axes.shape[1] != 2 or not len(axes)
            or not np.isfinite(axes).all()):
        raise ValueError('Horizontal axes must be finite nonzero XZ vectors')
    lengths = np.linalg.norm(axes, axis=1)
    if not np.isfinite(lengths).all() or (lengths <= 1e-8).any():
        raise ValueError('Horizontal axes must be finite nonzero XZ vectors')
    return axes / lengths[:, None]


def horizontal_axis_edges(lines, camera, axes, tolerance_degrees=12):
    """Check an explicit horizontal layout prior using image planes, not depth.

    This cannot prove an edge is horizontal or a tread. It only rejects lines
    incompatible with the supplied XZ axes. Diagonal flights need their own
    axes or no prior; callers must not silently assume a Manhattan layout.
    """
    axes = normalize_horizontal_axes(axes)
    if not np.isfinite(tolerance_degrees) or not 0 < tolerance_degrees < 90:
        raise ValueError('Axis tolerance must be within 0..90 degrees')
    lines = np.asarray(lines, dtype=float).reshape(-1, 2, 2)
    homogeneous = np.concatenate([lines, np.ones((*lines.shape[:2], 1))], axis=2)
    image_planes = np.cross(homogeneous[:, 0], homogeneous[:, 1])
    normals = np.column_stack((image_planes[:, :2] * camera['fov'],
                               image_planes @ [160, 120, 1])) @ camera_basis(camera)
    widths = np.cross(normals, [0, 1, 0])[:, [0, 2]]
    lengths = np.linalg.norm(widths, axis=1)
    widths = widths / np.maximum(lengths[:, None], 1e-8)
    aligned = np.max(np.abs(widths @ axes.T), axis=1)
    return (lengths > 1e-8) & (aligned >= np.cos(np.deg2rad(tolerance_degrees)))


def flight_regions(image, support, depth, camera, horizontal_axes=None):
    """Search both ordinary and tightly projected tread spacing.

    Distant flights can have less than three pixels between tread edges. Keep
    the same orientation/depth tests at both spacings, and deduplicate identical
    line sets so a second scale cannot manufacture independent evidence.
    """
    candidates, seen = [], set()
    for spacing in (3.0, 1.5):
        for region in _flight_regions(image, support, depth, camera, spacing, horizontal_axes):
            key = tuple(sorted(region["line_indices"]))
            if key in seen:
                continue
            seen.add(key)
            region["minimum_edge_spacing"] = spacing
            candidates.append(region)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def _flight_regions(image, support, depth, camera, minimum_spacing, horizontal_axes=None):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 80)
    edges[~support] = 0
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 12, minLineLength=12, maxLineGap=5)
    if raw is None:
        return []
    lines = np.asarray(raw).reshape(-1, 2, 2).astype(float)
    delta = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(delta, axis=1)
    direction = delta / length[:, None]
    mid = lines.mean(1)
    world = backproject(lines.reshape(-1, 2), depth, camera).reshape(-1, 2, 3)
    world_delta = world[:, 1] - world[:, 0]
    horizontal = (
        np.abs(world_delta[:, 1])
        < np.linalg.norm(world_delta[:, [0, 2]], axis=1) * 0.45
    )
    # Camera orientation supplies vertical vanishing directions independently
    # of predicted depth, which often flattens railing posts into the background.
    rotation = camera_basis(camera)
    normalized = (mid - [160, 120]) / camera["fov"]
    vertical = rotation[:2, 1] - normalized * rotation[2, 1]
    vertical /= np.maximum(np.linalg.norm(vertical, axis=1, keepdims=True), 1e-9)
    horizontal &= np.abs(np.sum(direction * vertical, axis=1)) < np.cos(np.deg2rad(20))
    if horizontal_axes is not None:
        horizontal &= horizontal_axis_edges(lines, camera, horizontal_axes)
    candidates = []
    seen = set()
    for i, tangent in enumerate(direction):
        normal = np.array([-tangent[1], tangent[0]])
        along = mid @ tangent
        across = mid @ normal
        selected = (np.abs(direction @ tangent) > np.cos(np.deg2rad(10))) & horizontal
        selected &= (length >= length[i] * 0.5) & (length <= length[i] * 2)
        selected &= np.abs(along - along[i]) < (length + length[i]) * 0.3
        indices = np.flatnonzero(selected)
        indices = indices[np.argsort(across[indices])]
        # Split distant parallel structures and collapse duplicate edge pairs.
        runs, run = [], []
        for idx in indices:
            if run and across[idx] - across[run[-1]] > 22:
                runs.append(run)
                run = []
            if not run or across[idx] - across[run[-1]] >= minimum_spacing:
                run.append(int(idx))
        if run:
            runs.append(run)
        for run in runs:
            key = tuple(sorted(run))
            if len(run) < 5 or key in seen or i not in run:
                continue
            seen.add(key)
            centers = world[run].mean(1)
            xz = centers[:, [0, 2]]
            _, _, basis = np.linalg.svd(xz - xz.mean(0), full_matrices=False)
            positions = (xz - xz.mean(0)) @ basis[0]
            span = float(np.ptp(positions))
            height = float(np.ptp(centers[:, 1]))
            if span < 400 or height < 300:
                continue
            slope, intercept = np.polyfit(positions, centers[:, 1], 1)
            residual = float(
                np.median(np.abs(centers[:, 1] - (positions * slope + intercept)))
            )
            correlation = float(abs(np.corrcoef(positions, centers[:, 1])[0, 1]))
            if not (
                0.2 < abs(slope) < 1.8
                and correlation > 0.75
                and residual < max(150, height * 0.2)
            ):
                continue
            polygon = cv2.convexHull(
                lines[run].reshape(-1, 2).astype("float32")
            ).reshape(-1, 2)
            mask = np.zeros(support.shape, "uint8")
            cv2.fillConvexPoly(mask, np.rint(polygon).astype("int32"), 1)
            mask = cv2.dilate(mask, np.ones((3, 3), "uint8")).astype(bool) & support
            if mask.sum() < 150:
                continue
            candidates.append(
                {
                    "mask": mask,
                    "polygon": polygon.tolist(),
                    "line_indices": run,
                    "line_segments": lines[run].reshape(-1, 2, 2).tolist(),
                    "tread_edges": len(run),
                    "height_span": height,
                    "run_span": span,
                    "slope": float(slope),
                    "correlation": correlation,
                    "residual": residual,
                    "score": float(
                        len(run) * correlation / (1 + residual / max(height, 1))
                    ),
                }
            )
            if horizontal_axes is not None:
                candidates[-1]['horizontal_axis_prior'] = {
                    'axes_xz': np.asarray(horizontal_axes, dtype=float).tolist(),
                    'tolerance_degrees': 12,
                    'qualification': 'Explicit estimated layout prior, not recovered tread identity',
                }
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def supported_original(detection, original, regions):
    """Retain a moderate-confidence original mask with strong tread evidence.

    This does not increase SAM's confidence or call the mask corrected. It only
    supplies structural evidence when prompted SAM proposals fragment the flight.
    """
    if detection["score"] < 0.3 or detection["predicted_iou"] < 0.65:
        return None
    for region in regions:
        coverage = float((region["mask"] & original).sum() / max(1, original.sum()))
        if (
            region["tread_edges"] >= 5
            and region["correlation"] >= 0.9
            and 0.15 <= coverage <= 0.9
        ):
            return region
    return None


def refine_segments(room, images, segments, depths, out, horizontal_axes=None):
    """Local SAM proposals prompted by detected tread regions, with provenance.

    Low SAM confidence is retained explicitly. Consumers must apply geometric
    acceptance gates rather than silently treating these as reliable labels.
    """
    import hashlib
    import json

    from PIL import Image
    from room3d_segment import MODEL, REVISION, Segmenter

    out.mkdir(parents=True, exist_ok=True)
    if horizontal_axes is not None:
        horizontal_axes = normalize_horizontal_axes(horizontal_axes).tolist()
    refined = []
    records = []
    model = None
    for camera, image, entries, (depth, _) in zip(
        room["cameras"], images, segments, depths, strict=True
    ):
        output = []
        for detection, original in entries:
            if detection["label"].strip().casefold() not in STAIR_LABELS:
                output.append((detection, original))
                continue
            regions = flight_regions(image, original, depth, camera,
                                     **({'horizontal_axes': horizontal_axes}
                                        if horizontal_axes is not None else {}))
            best = None
            proposal_method = "sam_tread_prompt"
            attempts = []
            for region in regions[:2]:
                if model is None:
                    model = Segmenter()
                point = np.mean(region["polygon"], axis=0).tolist()
                inputs = model.processor(
                    images=image,
                    input_points=[[[point]]],
                    input_labels=[[[1]]],
                    return_tensors="pt",
                )
                with model.torch.inference_mode():
                    prediction = model.model(**inputs, multimask_output=True)
                masks = (
                    model.processor.post_process_masks(
                        prediction.pred_masks.cpu(), inputs["original_sizes"]
                    )[0][0]
                    .numpy()
                    .astype(bool)
                )
                scores = prediction.iou_scores[0, 0].tolist()
                for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
                    mask &= original
                    support = float(
                        (mask & region["mask"]).sum() / max(1, region["mask"].sum())
                    )
                    bounded = (
                        region["mask"].sum() * 0.8 <= mask.sum() <= original.sum() * 0.8
                    )
                    eligible = bool(score >= 0.2 and support >= 0.7 and bounded)
                    attempts.append(
                        {
                            "point": point,
                            "candidate": index,
                            "predicted_iou": float(score),
                            "mask_pixels": int(mask.sum()),
                            "tread_region_coverage": support,
                            "eligible": eligible,
                        }
                    )
                    if eligible and (best is None or score > best[0]):
                        best = (float(score), mask, region, point)
            if best is None:
                region = supported_original(detection, original, regions)
                if region is not None:
                    best = (
                        float(detection["predicted_iou"]),
                        original,
                        region,
                        np.mean(region["polygon"], axis=0).tolist(),
                    )
                    proposal_method = "original_mask_with_tread_evidence"
            record = {
                "camera": camera["index"],
                "source_label": detection["label"],
                "source_mask": detection["mask"],
                "source_mask_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                "model": MODEL,
                "revision": REVISION,
                "attempts": attempts,
                "accepted_proposal": best is not None,
                "proposal_method": proposal_method if best is not None else None,
            }
            if horizontal_axes is not None:
                record['horizontal_axes'] = horizontal_axes
            if best is None:
                output.append((detection, original))
            else:
                score, mask, region, point = best
                name = f"camera{camera['index']}-{len(records)}-stairs.png"
                Image.fromarray(mask.astype("uint8") * 255).save(out / name)
                record.update(
                    mask=name,
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    predicted_iou=score,
                    structural_evidence={
                        k: v for k, v in region.items() if k != "mask"
                    },
                    point=point,
                )
                item = dict(detection)
                item.update(
                    predicted_iou=score,
                    structural_refinement=record,
                    original_predicted_iou=detection["predicted_iou"],
                )
                output.append((item, mask))
            records.append(record)
        refined.append(output)
    (out / "refinements.json").write_text(json.dumps(records, indent=2) + "\n")
    return refined


def load_refinements(root, room, images, segments, depths, horizontal_axes=None):
    """Load only proposals bound to these exact image, mask and depth inputs."""
    import hashlib
    import json
    from pathlib import Path

    from PIL import Image

    records = json.loads((root / "refinements.json").read_text())
    if horizontal_axes is not None:
        horizontal_axes = normalize_horizontal_axes(horizontal_axes).tolist()
    for record in records:
        stored = record.get('horizontal_axes')
        if stored is None or horizontal_axes is None:
            matches = stored is None and horizontal_axes is None
        else:
            stored = np.asarray(stored, dtype=float)
            matches = (stored.shape == np.asarray(horizontal_axes).shape
                       and np.allclose(stored, horizontal_axes, rtol=0, atol=1e-12))
        if not matches:
            raise ValueError('Stair refinement horizontal-axis prior differs')
    indexed = {(r["camera"], r["source_mask"]): r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate refinement record")
    output = []
    for camera, image, entries, (depth, _) in zip(
        room["cameras"], images, segments, depths, strict=True
    ):
        items = []
        for detection, mask in entries:
            record = indexed.pop((camera["index"], detection["mask"]), None)
            if record is not None:
                if "source_label" in record and record["source_label"] != detection.get("label"):
                    raise ValueError("Stair refinement source label differs")
                for key, data in (
                    ("source_mask_sha256", mask.tobytes()),
                    ("rgb_sha256", image.tobytes()),
                    ("depth_sha256", depth.tobytes()),
                ):
                    if hashlib.sha256(data).hexdigest() != record[key]:
                        raise ValueError(f"Stair refinement {key} differs")
                if record["accepted_proposal"]:
                    name = record["mask"]
                    if Path(name).name != name or "\\" in name:
                        raise ValueError("Unsafe refinement mask path")
                    refined = np.asarray(Image.open(root / name).convert("L")) > 0
                    if (
                        refined.shape != mask.shape
                        or hashlib.sha256(refined.tobytes()).hexdigest()
                        != record["mask_sha256"]
                    ):
                        raise ValueError("Refinement mask differs")
                    detection = {
                        **detection,
                        "predicted_iou": record["predicted_iou"],
                        "original_predicted_iou": detection["predicted_iou"],
                        "structural_refinement": record,
                    }
                    mask = refined
            items.append((detection, mask))
        output.append(items)
    if indexed:
        raise ValueError("Refinement references unknown camera/mask")
    return output


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from PIL import Image
    from room3d_refine import load_depth
    from room3d_segment import load_masks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument('--horizontal-axis', nargs=2, type=float, action='append',
                        help='Optional explicit XZ tread-width layout prior; repeat for multiple axes')
    args = parser.parse_args()
    room = json.loads(args.survey.read_text())
    images = [
        Image.open(args.images / f"camera{c['index']}-original.png").convert("RGB")
        for c in room["cameras"]
    ]
    depths = load_depth(args.depth, args.survey, room, images)
    segments = load_masks(args.segments, room, images)
    refine_segments(room, images, segments, depths, args.out, args.horizontal_axis)
    print(args.out / "refinements.json")
