"""Reversible, bounded geometry search scored against local depth proposals.

A lower score means closer agreement with a model, not proven original geometry.
Original room cameras, collision records and script anchors are never modified.
"""

import copy
import hashlib
import html
import json

import cv2
import numpy as np
from room3d_preview import render


def load_depth(root, survey, room, images):
    record = json.loads((root / "depth.json").read_text())
    if record["survey_sha256"] != hashlib.sha256(survey.read_bytes()).hexdigest():
        raise ValueError("Depth survey hash mismatch")
    views = []
    for camera, image, item in zip(
        room["cameras"], images, record["views"], strict=True
    ):
        if (
            item["camera"] != camera["index"]
            or item["rgb_sha256"] != hashlib.sha256(image.tobytes()).hexdigest()
        ):
            raise ValueError("Depth camera/image mismatch")
        depth = np.load(root / f"camera{camera['index']}-depth.npy", allow_pickle=False)
        confidence = np.load(
            root / f"camera{camera['index']}-confidence.npy", allow_pickle=False
        )
        if (
            depth.ndim != 2
            or depth.shape != confidence.shape
            or not np.isfinite(depth).all()
            or not (depth > 0).all()
        ):
            raise ValueError("Invalid depth/confidence array")
        if not np.isfinite(confidence).all():
            raise ValueError("Invalid depth confidence")
        depth = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_LINEAR)
        confidence = cv2.resize(confidence, (320, 240), interpolation=cv2.INTER_NEAREST)
        valid = confidence >= np.percentile(confidence, 40)
        views.append((depth, valid))
    return views


def depth_error(rendered, target, valid):
    """Fixed target support prevents disappearing surfaces from improving score."""
    if not valid.any():
        raise ValueError("No depth evidence")
    error = np.ones(target.shape, dtype=float)
    visible = np.isfinite(rendered) & (rendered > 0)
    error[visible] = np.minimum(np.abs(np.log(rendered[visible] / target[visible])), 1)
    return float(error[valid].mean())


def improvements(before, after):
    return (
        all(a <= b + 0.005 for a, b in zip(after, before, strict=True))
        and float(np.mean(after)) < float(np.mean(before)) - 0.0005
    )


def candidates(recipe):
    for index, item in enumerate(recipe.get("primitives", [])):
        if item["kind"] not in ("door_panel", "chest_body", "typewriter_table"):
            continue
        for factor in (0.8, 1.2):
            candidate = copy.deepcopy(recipe)
            obj = candidate["primitives"][index]
            if "box" in obj:
                obj["box"][4] = max(-7000, min(-300, obj["box"][4] * factor))
            else:
                obj["table_top"] = max(
                    -3000, min(-500, obj.get("table_top", -1550) * factor)
                )
            yield f"primitive_{index}_height_{factor}", candidate


def silhouette_support(mesh, room, segments, obj, group):
    evidence = obj.get("evidence", {})
    if segments is None or not evidence.get("mask"):
        return None
    for camera, entries in zip(room["cameras"], segments, strict=True):
        if camera["index"] != evidence["camera"]:
            continue
        for detection, target in entries:
            if detection.get("mask") != evidence["mask"]:
                continue
            _, _, scene = render(mesh, camera, return_depth=True)
            _, _, part = render(mesh, camera, return_depth=True, only_groups={group})
            finite = np.isfinite(part) & np.isfinite(scene)
            visible = np.zeros(scene.shape, dtype=bool)
            visible[finite] = np.abs(part[finite] - scene[finite]) < np.maximum(
                50, part[finite] * 0.01
            )
            union = int((target | visible).sum())
            return float((target & visible).sum() / max(1, union))
    return None


def refine(room, recipe, depth_views, out, generate_mesh, objects=(), segments=None):
    out.mkdir(parents=True, exist_ok=True)
    baseline = copy.deepcopy(recipe)
    accepted = copy.deepcopy(recipe)
    history = []

    def evaluate(candidate, name):
        mesh = out / (name + ".obj")
        generate_mesh(room, mesh, candidate)
        scores = []
        for camera, (target, valid) in zip(room["cameras"], depth_views, strict=True):
            image, _, depth = render(mesh, camera, return_depth=True)
            image.save(out / f"{name}-camera{camera['index']}.png")
            scores.append(depth_error(depth, target, valid))
        (out / (name + ".json")).write_text(json.dumps(candidate, indent=2) + "\n")
        return scores

    initial = evaluate(baseline, "before")
    current = initial
    # One bounded pass; each candidate is compared against the current accepted
    # state in every original view. Rejection leaves that state unchanged.
    for name, candidate in candidates(baseline):
        # Keep earlier accepted changes to other primitives.
        index = int(name.split("_")[1])
        merged = copy.deepcopy(accepted)
        merged["primitives"][index] = candidate["primitives"][index]
        scores = evaluate(merged, name)
        keep = improvements(current, scores)
        history.append(
            {"candidate": name, "before": current, "after": scores, "accepted": keep}
        )
        if keep:
            accepted, current = merged, scores
    added = set()
    for pass_index in range(2):
        changed = False
        for i, obj in enumerate(objects):
            if i in added:
                continue
            candidate = copy.deepcopy(accepted)
            group = f"primitive_{len(candidate['primitives'])}"
            candidate["primitives"].append(obj)
            candidate["skip_boundaries"] = sorted(
                set(candidate.get("skip_boundaries", []))
                | set(obj.get("replace_visual_boundaries", []))
            )
            name = f"object_{i}_{obj['kind']}_pass{pass_index}"
            scores = evaluate(candidate, name)
            overlap = silhouette_support(
                out / (name + ".obj"), room, segments, obj, group
            )
            bounded = all(a <= b + 0.005 for a, b in zip(scores, current, strict=True))
            keep = improvements(current, scores) or (
                bounded and overlap is not None and overlap >= 0.35
            )
            history.append(
                {
                    "candidate": name,
                    "before": current,
                    "after": scores,
                    "accepted": keep,
                    "visible_silhouette_iou": overlap,
                    "criterion": "Depth improvement or observed silhouette coverage, with bounded per-view depth regression",
                }
            )
            if keep:
                accepted, current = candidate, scores
                added.add(i)
                changed = True
        if not changed:
            break
    if not any(p["kind"] == "ceiling" for p in accepted.get("primitives", [])):
        # Complete the unobserved top of the room with a conservative envelope.
        # Outside original cameras get an explicit cutaway; gameplay cameras
        # inside the inferred volume retain the roof.
        candidate = copy.deepcopy(accepted)
        bounds = room["boundaries"]
        stair_tops = [
            p["box"][4] for p in candidate["primitives"] if p["kind"] == "staircase"
        ]
        roof_y = min(
            -candidate.get("wall_height", 6000), min(stair_tops, default=0) - 2500
        )
        index = len(candidate["primitives"])
        candidate["primitives"].append(
            {
                "kind": "ceiling",
                "box": [
                    min(b["xmin"] for b in bounds),
                    min(b["zmin"] for b in bounds),
                    max(b["xmax"] for b in bounds),
                    max(b["zmax"] for b in bounds),
                    roof_y - 50,
                    roof_y,
                ],
                "estimated": "Unobserved enclosing roof; height from walls and inferred upper stair level",
            }
        )
        candidate.setdefault("hide", []).extend(
            [
                [c["index"], f"primitive_{index}"]
                for c in room["cameras"]
                if c["origin"][1] < roof_y
            ]
        )
        scores = evaluate(candidate, "inferred_enclosure")
        # A hidden completion may add coverage without improving original views.
        # It must still satisfy the per-view regression bound.
        keep = all(a <= b + 0.005 for a, b in zip(scores, current, strict=True))
        history.append(
            {
                "candidate": "inferred_enclosure",
                "before": current,
                "after": scores,
                "accepted": keep,
                "criterion": "Adds hidden roof coverage with bounded original-view regression",
            }
        )
        if keep:
            accepted, current = candidate, scores
    final = evaluate(accepted, "after")
    report = {
        "metric": "mean capped absolute log depth error on fixed confident pixels",
        "qualification": "Agreement with DA3 proposals; not ground-truth fidelity",
        "before": initial,
        "after": final,
        "trials": history,
        "invariants": "Original camera/collision/script data unchanged; proposed heights and closed object additions searched",
    }
    (out / "corrections.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [
        "<h1>Reversible depth refinement</h1>",
        "<p>Scores measure agreement with estimated depth, not ground truth.</p>",
        '<a href="corrections.json">Full evidence</a>',
    ]
    for camera in room["cameras"]:
        i = camera["index"]
        rows.append(
            f'<h2>Camera {i}</h2><p>Before / After</p><img src="before-camera{i}.png"><img src="after-camera{i}.png">'
        )
    for trial in history:
        name = html.escape(trial["candidate"])
        rows.append(
            f'<p><a href="{name}.json">{name}</a>: {"accepted" if trial["accepted"] else "rejected"}</p>'
        )
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Depth refinement</title>'
        + "\n".join(rows)
    )
    return accepted, report


def calibrate_floor_depth(observation, floor_depth, scene_depth, floor_mask):
    """Calibrate appearance depth scale against a visible, semantically supported floor.

    Checkerboard tiles separate fitting and validation pixels. This is an image
    calibration diagnostic, not held-out-camera geometric validation. Rejection
    returns the original observation; geometry fitting depths are never mutated.
    """
    target, valid = observation
    finite = np.isfinite(floor_depth) & np.isfinite(scene_depth)
    visible = finite & (np.abs(np.where(finite, floor_depth, 0)
                               - np.where(finite, scene_depth, 0)) < 2)
    selected = visible & floor_mask & valid & np.isfinite(target) & (target > 0)
    y, x = np.nonzero(selected)
    report = {'accepted': False, 'samples': len(x), 'scale': 1.,
              'qualification': 'Floor-calibrated appearance depth; not independent geometry validation'}
    if len(x) < 300:
        return observation, report
    fit = (x//16+y//16) % 2 == 0
    if min(int(fit.sum()), int((~fit).sum())) < 100:
        return observation, report
    ratios = floor_depth[selected]/target[selected]
    scale = float(np.median(ratios[fit]))
    error = float(np.percentile(np.abs(np.log(ratios[~fit]/scale)), 90))
    report.update(proposed_scale=scale, validation_p90_log_error=error)
    if not .5 <= scale <= 2 or error > .08:
        return observation, report
    report.update(accepted=True, scale=scale)
    return (target*scale, valid.copy()), report
