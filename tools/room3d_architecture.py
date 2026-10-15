"""Experimental architectural correspondences; no automatic geometry edits."""

import numpy as np
from room3d_levels import floor_anchors
from room3d_reconstruct import project


def projected_object_evidence(part, scene, observed, valid, matching_mask=None):
    """Distinguish foreground occlusion from depth conflict and missing evidence.

    Predicted depth is model evidence, not ground truth. A projection over a
    different visible object is not a contradiction when that object is closer.
    Unknown pixels and inferred-scene occlusion cannot validate object identity.
    """
    part, scene, observed = (np.asarray(a, float) for a in (part, scene, observed))
    valid = np.asarray(valid, bool)
    if part.ndim != 2 or any(a.shape != part.shape for a in (scene, observed, valid)):
        raise ValueError("Expected matching two-dimensional depth and validity arrays")
    if matching_mask is not None:
        matching_mask = np.asarray(matching_mask, bool)
        if matching_mask.shape != part.shape:
            raise ValueError("Matching mask dimensions differ")
    projected = np.isfinite(part) & (part > 0)
    known = projected & valid & np.isfinite(observed) & (observed > 0)
    scene_hidden = projected & np.isfinite(scene) & (part > scene + 50)
    states = np.zeros(part.shape, np.uint8)
    states[projected] = 1
    states[scene_hidden] = 2
    # Compute only on finite samples, avoiding inf-inf and invalid-depth math.
    delta = observed[known] - part[known]
    tolerance = np.maximum(180, observed[known] * 0.12)
    classifications = np.full(delta.shape, 4, np.uint8)
    classifications[delta < -tolerance] = 3
    classifications[delta > tolerance] = 5
    states[known] = classifications
    if matching_mask is not None:
        states[(states == 4) & matching_mask] = 6
    labels = {1: "unknown", 2: "inferred_scene_occluded", 3: "observed_foreground",
              4: "depth_agreement_without_identity", 5: "model_depth_conflict",
              6: "depth_agreement_with_matching_mask"}
    return {"projected_pixels": int(projected.sum()), "reliable_pixels": int(known.sum()),
            "counts": {label: int((states == value).sum()) for value, label in labels.items()},
            "qualification": "Model/scene evidence categories, not fidelity or independent object identity approval; unknown and occluded pixels are not contradictions"}


def fit_column_axis(room, hypothesis, segments, floor_y, out, generate_mesh):
    """Fit one shared procedural shape on the two contributing masks only.

    This isolated silhouette fit does not accept a room edit. Scene occlusion,
    reliable depth and independent-view checks must follow before adoption.
    """
    from room3d_preview import render
    from room3d_shape_fit import compare, isolated_recipe
    from room3d_shapes import generate

    if not hypothesis.get("eligible") or hypothesis.get("ambiguous"):
        raise ValueError("Expected an eligible, unambiguous axis hypothesis")
    if len(hypothesis["observations"]) != 2 or len(hypothesis["checks"]) != 2:
        raise ValueError("Expected two contributing views")
    cameras = {c["index"]: c for c in room["cameras"]}
    masks = {(c["index"], d["mask"]): mask
             for c, view in zip(room["cameras"], segments, strict=True) for d, mask in view}
    views = [(cameras[o["camera"]], masks[o["camera"], o["mask"]])
             for o in hypothesis["observations"]]
    if views[0][0]["index"] == views[1][0]["index"]:
        raise ValueError("Expected distinct contributing cameras")
    origin = [hypothesis["xz"][0], floor_y, hypothesis["xz"][1]]
    height = float(np.mean([c["height"] for c in hypothesis["checks"]]))
    # Estimate shaft width from central rows, avoiding wider base/capital.
    widths = []
    for camera, mask in views:
        y, _ = np.nonzero(mask)
        low, high = np.quantile(y, [0.3, 0.7])
        spans = [np.ptp(np.flatnonzero(mask[j])) + 1
                 for j in range(int(low), int(high) + 1) if mask[j].any()]
        _, z = project(np.array([[origin[0], floor_y - height / 2, origin[2]]]), camera)
        widths.append(float(np.median(spans) * z[0] / camera["fov"]))
    width = float(np.mean(widths))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trial.obj"
    trials, best = [], None
    for height_scale in (0.95, 1.0, 1.05):
        for width_scale in (0.7, 0.9, 1.1, 1.3):
            for base_fraction, capital_fraction in ((0.05, 0.05), (0.08, 0.08), (0.12, 0.05)):
                obj = generate("column", {"height": height * height_scale,
                                          "width": width * width_scale,
                                          "base_height": height * height_scale * base_fraction,
                                          "capital_height": height * height_scale * capital_fraction}, origin)
                generate_mesh(room, path, isolated_recipe(room, {"floor_y": floor_y}, obj))
                scores = []
                for camera, mask in views:
                    depth = render(path, camera, return_depth=True,
                                   only_groups={"primitive_0"})[2]
                    score = compare(depth, depth, mask, np.zeros(mask.shape),
                                    np.zeros(mask.shape, bool))
                    score.pop("depth_error")
                    # No observed depth contributes to this isolated shape fit.
                    score["loss"] = 0.8 * (1 - score["silhouette_iou"]) + 0.2 * score["edge_error"]
                    scores.append({"camera": camera["index"], **score})
                loss = max(s["loss"] for s in scores)
                trials.append({"shape": obj["shape"], "scores": scores, "worst_view_loss": loss})
                if best is None or loss < best[0]:
                    best = (loss, obj, len(trials) - 1)
    _, obj, selected = best
    generate_mesh(room, out / "selected.obj", isolated_recipe(room, {"floor_y": floor_y}, obj))
    return obj, {"trials": trials, "selected": selected,
                 "training_cameras": [c["index"] for c, _ in views], "heldout_cameras": [],
                 "accepted_room_edit": False,
                 "qualification": "Shared isolated silhouette fit on contributing views only; scene, depth, object identity and independent-view quality unverified"}


def column_axis_hypotheses(room, segments, floor_y=0, *, excluded_cameras=()):
    """Intersect vertical interpretation planes without matching visible feet.

    Mask centerlines supply an axis plane per camera. Two planes always admit
    a vertical intersection: axis agreement alone cannot establish identity.
    Independent height and floor-extent checks reject incompatible proposals;
    remaining proposals still need rendered shape and unused-view evaluation.
    """
    from room3d_levels import rays

    if not np.isfinite(floor_y):
        raise ValueError("Expected finite floor height")
    excluded = set(excluded_cameras)
    observations = []
    for camera, entries in zip(room["cameras"], segments, strict=True):
        if camera["index"] in excluded:
            continue
        for detection, mask in entries:
            if (not any(w in detection["label"] for w in ("column", "pillar"))
                    or detection["score"] < 0.28 or detection["predicted_iou"] < 0.85):
                continue
            y, x = np.nonzero(mask)
            if len(y) < 24 or np.ptp(y) < 8 or mask[0].any() or mask[-1].any():
                continue
            low, high = np.quantile(y, [0.2, 0.8])
            pixels = np.array([[np.median(np.flatnonzero(mask[j])), j]
                               for j in range(int(low), int(high) + 1) if mask[j].any()])
            if len(pixels) < 4:
                continue
            direction = rays(camera, pixels)[:, [0, 2]]
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            if (norms < 1e-9).any():
                continue
            direction /= norms
            normal = np.median(np.column_stack([-direction[:, 1], direction[:, 0]]), axis=0)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            observations.append({"camera": camera, "mask": detection["mask"],
                                 "pixels": pixels, "normal": normal,
                                 "offset": float(normal @ np.asarray(camera["origin"])[[0, 2]]),
                                 "top": int(y.min()), "bottom": int(y.max()),
                                 "center_x": float(np.median(x))})
    trials = []
    for i, a in enumerate(observations):
        for b in observations[i + 1:]:
            if a["camera"]["index"] == b["camera"]["index"]:
                continue
            matrix = np.array([a["normal"], b["normal"]])
            condition = float(np.linalg.cond(matrix))
            trial = {"observations": [{"camera": o["camera"]["index"], "mask": o["mask"]}
                                      for o in (a, b)], "condition": condition}
            if condition > 20:
                trials.append({**trial, "eligible": False, "reason": "weak_baseline"})
                continue
            point = np.linalg.solve(matrix, [a["offset"], b["offset"]])
            checks = []
            for obs in (a, b):
                camera = obs["camera"]
                origin = np.asarray(camera["origin"])
                delta = point - origin[[0, 2]]
                directions = rays(camera, obs["pixels"])
                t = (directions[:, [0, 2]] @ delta) / (directions[:, [0, 2]] ** 2).sum(1)
                axis = origin + t[:, None] * directions
                axis[:, 0], axis[:, 2] = point
                uv, z = project(axis, camera)
                endpoint_rays = rays(camera, [[obs["center_x"], obs["top"]],
                                             [obs["center_x"], obs["bottom"]]])
                endpoint_t = ((endpoint_rays[:, [0, 2]] @ delta)
                              / (endpoint_rays[:, [0, 2]] ** 2).sum(1))
                endpoints = origin + endpoint_t[:, None] * endpoint_rays
                floor_uv, _ = project(np.array([[point[0], floor_y, point[1]]]), camera)
                checks.append({"camera": camera["index"],
                               "axis_error": float(np.linalg.norm(uv - obs["pixels"], axis=1).max()),
                               "front": bool((z > 2 * camera["fov"]).all()),
                               "height": float(floor_y - endpoints[0, 1]),
                               "floor_extent_error": float(abs(floor_uv[0, 1] - obs["bottom"])),
                               "floor_extent_limit": 3 + 0.2 * (obs["bottom"] - obs["top"])})
            heights = [c["height"] for c in checks]
            difference = float(abs(heights[0] - heights[1]) / max(1, np.mean(np.abs(heights))))
            gates = {"axis": all(c["axis_error"] <= 3 and c["front"] for c in checks),
                     "height": all(200 <= h <= 12000 for h in heights) and difference <= 0.1,
                     "floor_extent": all(c["floor_extent_error"] <= c["floor_extent_limit"] for c in checks)}
            trials.append({**trial, "xz": point.tolist(), "checks": checks, "gates": gates,
                           "relative_height_difference": difference,
                           "eligible": all(gates.values())})
    eligible = [t for t in trials if t["eligible"]]
    for trial in trials:
        trial["ambiguous"] = False
        if not trial["eligible"]:
            continue
        ids = {(o["camera"], o["mask"]) for o in trial["observations"]}
        views = {o["camera"] for o in trial["observations"]}
        trial["ambiguous"] = any(
            other is not trial
            and {o["camera"] for o in other["observations"]} == views
            and bool(ids & {(o["camera"], o["mask"]) for o in other["observations"]})
            for other in eligible
        )
    return {"trials": trials, "excluded_cameras": sorted(excluded), "floor_y": floor_y,
            "qualification": "Provisional shared vertical axes; identity, dimensions, rendered shape and unused-view consistency unverified"}


def column_foot_correspondences(room, segments, floor_y=0, *, excluded_cameras=(),
                                max_pixel_error=3):
    """Compare visible column bases using a shared floor-plane hypothesis.

    Reciprocal projection checks reject different pillars. A positive match
    remains provisional: the floor, mask completeness and identity are assumed.
    Ambiguous alternatives are reported rather than merged.
    """
    if not np.isfinite(floor_y) or not np.isfinite(max_pixel_error) or max_pixel_error <= 0:
        raise ValueError("Expected finite floor and positive pixel tolerance")
    excluded = set(excluded_cameras)
    anchors = []
    cameras = {c["index"]: c for c in room["cameras"]}
    for camera, entries in zip(room["cameras"], segments, strict=True):
        if camera["index"] in excluded:
            continue
        for detection, mask in entries:
            if not any(word in detection["label"] for word in ("column", "pillar")):
                continue
            if detection["predicted_iou"] < 0.85 or detection["score"] < 0.28:
                continue
            # A cut-off mask cannot establish where a column contacts the floor.
            if mask[-1].any() or mask[0].any():
                continue
            candidates = floor_anchors(camera, mask, floor_y)
            if not candidates:
                continue
            anchor = candidates[len(candidates) // 2]
            anchors.append({"camera": camera["index"], "mask": detection["mask"],
                            "point": anchor["point"], "pixel": anchor["pixel"],
                            "detector_score": detection["score"],
                            "sam_score": detection["predicted_iou"]})
    pairs, rejected = [], []
    for i, a in enumerate(anchors):
        for j in range(i + 1, len(anchors)):
            b = anchors[j]
            if a["camera"] == b["camera"]:
                continue
            errors = []
            for source, target in ((a, b), (b, a)):
                camera = cameras[target["camera"]]
                uv, depth = project(np.asarray([source["point"]]), camera)
                if not np.isfinite(uv).all() or depth[0] <= 2 * camera["fov"]:
                    break
                errors.append(float(np.linalg.norm(uv[0] - target["pixel"])))
            if len(errors) == 2 and max(errors) <= max_pixel_error:
                pairs.append({"anchors": [i, j], "pixel_errors": errors})
            else:
                rejected.append({"anchors": [i, j], "pixel_errors": errors,
                                 "reason": "reprojection_error" if len(errors) == 2
                                 else "behind_camera_or_invalid_projection"})
    for pair in pairs:
        a, b = pair["anchors"]
        pair["ambiguous"] = any(
            other is not pair and (
                (a in other["anchors"] and any(
                    k != a and anchors[k]["camera"] == anchors[b]["camera"]
                    for k in other["anchors"]
                )) or (b in other["anchors"] and any(
                    k != b and anchors[k]["camera"] == anchors[a]["camera"]
                    for k in other["anchors"]
                ))
            ) for other in pairs
        )
    return {"anchors": anchors, "pairs": pairs, "rejected_pairs": rejected, "floor_y": floor_y,
            "excluded_cameras": sorted(excluded), "max_pixel_error": max_pixel_error,
            "qualification": "Provisional floor-plane correspondences; no recovered geometry, object identity or structural support approval"}
