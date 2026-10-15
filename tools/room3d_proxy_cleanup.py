"""Opt-in removal of provisional visual collision boxes beneath fitted stairs.

Original collision records and actual primitives are never edited. Floor masks
and model depth only nominate alternatives; final geometry stays estimated.
"""

import copy
import json
import math

import numpy as np


def flight_covers(boundary, stairs, recipe):
    """Require the entire proxy footprint inside a ground flight/attached slab."""
    if stairs.get("kind") != "staircase" or not stairs.get("shape"):
        return False
    shape = stairs["shape"]
    origin = np.asarray(shape["origin"], float)
    p = shape["parameters"]
    if abs(origin[1] - float(recipe.get("floor_y", 0))) > 1e-5:
        return False  # A raised support must remain present.
    yaw = float(shape["yaw"])
    across = np.array([math.cos(yaw), -math.sin(yaw)])
    along = np.array([math.sin(yaw), math.cos(yaw)])
    width, run = float(p["width"]), float(p["run"])
    if not np.isfinite([width, run, yaw, *origin]).all() or min(width, run) <= 0:
        return False
    extent = run
    for landing in recipe.get("primitives", []):
        if landing.get("kind") != "landing" or not landing.get("parts"):
            continue
        xyz = np.asarray(landing["parts"], float).reshape(-1, 3)
        delta = xyz[:, [0, 2]] - origin[[0, 2]]
        x, z = delta @ across, delta @ along
        if (np.isfinite(xyz).all()
                and abs(xyz[:, 1].min() - (origin[1] - p["rise"])) <= 1e-5
                and abs(z.min() - run) <= 1e-5
                and np.abs(x).max() <= width / 2 + 1e-5):
            extent = max(extent, float(z.max()))
    corners = np.array([[boundary[x], boundary[z]]
                        for x in ("xmin", "xmax") for z in ("zmin", "zmax")])
    delta = corners - origin[[0, 2]]
    x, z = delta @ across, delta @ along
    return bool((np.abs(x) <= width / 2 + 1e-5).all()
                and (z >= -1e-5).all() and (z <= extent + 1e-5).all())


def eligible_boundaries(room, recipe):
    attached = {p.get("attachment_prior", {}).get("group")
                for p in recipe.get("primitives", [])}
    return {
        f"boundary_{b['index']}": b for b in room["boundaries"]
        if b["type"] & 255 in (1, 5) and not b["flags"] & 256
        and b["index"] not in recipe.get("skip_boundaries", [])
        and b["index"] not in recipe.get("shared_wall_boundaries", [])
        and str(b["index"]) not in recipe.get("boundary_heights", {})
        and f"boundary_{b['index']}" not in attached
        and any(flight_covers(b, p, recipe) for p in recipe.get("primitives", []))
    }


def fit(room, recipe, segments, depths, out, generate_mesh, heldout_cameras=()):
    from room3d_preview import render
    from room3d_refine import depth_error

    out.mkdir(parents=True, exist_ok=True)
    heldout = set(heldout_cameras)
    camera_ids = {c["index"] for c in room["cameras"]}
    if not heldout <= camera_ids:
        raise ValueError("Expected known reserved cameras")
    report = {"accepted": False, "selected": None, "diagnostics": [], "trials": [],
              "training_cameras": sorted(camera_ids - heldout), "heldout_cameras": sorted(heldout),
              "qualification": "Estimated visual proxy replacement; original collision data and primitives unchanged. Not fidelity or navigation proof."}

    def finish(candidate):
        (out / "results.json").write_text(json.dumps(report, indent=2))
        return candidate, report

    if heldout == camera_ids:
        report["reason"] = "no_training_camera"
        return finish(recipe)
    eligible = eligible_boundaries(room, recipe)
    if not eligible:
        report["reason"] = "no_ground_flight_proxy"
        return finish(recipe)
    path = out / "baseline.obj"
    generate_mesh(room, path, recipe)
    scenes, before = {}, {}
    views = list(zip(room["cameras"], segments, depths, strict=True))
    for camera, _, (target, valid) in views:
        index = camera["index"]
        scenes[index] = render(path, camera, return_depth=True)[2]
        before[index] = depth_error(scenes[index], target, valid)
    guard = None
    if recipe.get("floor_depth_guard", False):
        from room3d_depth_guard import FloorDepthGuard

        guard = FloorDepthGuard(room, segments, depths, path, scenes)
        report["floor_depth_calibration"] = guard.calibration
    votes = {}
    for camera, entries, (target, valid) in views:
        index = camera["index"]
        if index in heldout:
            continue
        masks = [m for item, m in entries
                 if item["label"].strip().casefold() in ("floor", "wooden floor")
                 and item["score"] >= .3 and item["predicted_iou"] >= .9 and m.sum() >= 500]
        if not masks:
            continue
        mask = np.logical_or.reduce(masks)
        floor = render(path, camera, return_depth=True, only_groups={"floor"})[2]
        for name, boundary in eligible.items():
            proxy = render(path, camera, return_depth=True, only_groups={name})[2]
            blocked = mask & np.isfinite(floor) & np.isfinite(proxy) & (proxy < floor - 50)
            nearest = np.zeros(mask.shape, bool)
            nearest[blocked] = np.abs(proxy[blocked] - scenes[index][blocked]) < 1
            reliable = nearest & valid & np.isfinite(target) & (target > 0)
            if nearest.sum() < 500 or reliable.sum() < 50:
                continue
            observed = float(np.median(target[reliable]))
            surface = float(np.median(proxy[nearest]))
            if observed - surface <= max(180, observed * .12):
                continue
            votes[boundary["index"]] = votes.get(boundary["index"], 0) + int(nearest.sum())
            report["diagnostics"].append({"camera": index, "group": name,
                                          "floor_pixels": int(nearest.sum()),
                                          "reliable_pixels": int(reliable.sum()),
                                          "observed_depth": observed, "surface_depth": surface})
    indices = sorted(votes, key=lambda i: (-votes[i], i))[:3]
    if not indices:
        report["reason"] = "no_training_floor_contradiction"
        return finish(recipe)
    alternatives = [[i] for i in indices] + ([indices] if len(indices) > 1 else [])
    solutions = []
    for number, removed in enumerate(alternatives):
        candidate = copy.deepcopy(recipe)
        candidate["skip_boundaries"] = sorted(set(recipe.get("skip_boundaries", [])) | set(removed))
        mesh = out / f"candidate{number}.obj"
        generate_mesh(room, mesh, candidate)
        training, calibrated = [], []
        for camera, _, (target, valid) in views:
            index = camera["index"]
            if index in heldout:
                continue
            scene = render(mesh, camera, return_depth=True)[2]
            error = depth_error(scene, target, valid)
            training.append({"camera": index, "depth_delta": error - before[index]})
            if guard is not None:
                calibrated.append(guard.compare(index, scene))
        loss = float(np.mean([v["depth_delta"] for v in training]))
        safe = all(v["depth_delta"] <= .005 for v in training) and loss < -.0005
        safe = safe and all(v["safe"] for v in calibrated)
        report["trials"].append({"removed_visual_proxies": removed, "training": training,
                                 "loss": loss, "training_safe": bool(safe)})
        if guard is not None:
            report["trials"][-1]["floor_depth_guard"] = calibrated
        if safe:
            solutions.append((loss, number, candidate, mesh))
    if not solutions:
        report["reason"] = "no_training_safe_replacement"
        return finish(recipe)
    _, report["selected"], candidate, mesh = min(solutions, key=lambda row: row[:2])
    validation = []
    for camera, _, (target, valid) in views:
        index = camera["index"]
        if index in heldout:
            scene = render(mesh, camera, return_depth=True)[2]
            delta = depth_error(scene, target, valid) - before[index]
            check = {"camera": index, "depth_delta": delta, "valid": bool(delta <= .005)}
            if guard is not None:
                check["floor_depth_guard"] = guard.compare(index, scene)
                check["valid"] = check["valid"] and check["floor_depth_guard"]["safe"]
            validation.append(check)
    report["heldout"] = validation
    report["accepted"] = all(v["valid"] for v in validation)
    report["reason"] = "visual_proxy_replaced" if report["accepted"] else "reserved_view_rejected"
    return finish(candidate if report["accepted"] else recipe)
