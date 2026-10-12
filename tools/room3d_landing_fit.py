"""Bounded creative landing proposals; no recovery or gameplay collision claims."""

import copy

import numpy as np
from room3d_assemblies import landing_for_stairs
from room3d_connections import stair_connections, stair_headroom, support_mesh
from room3d_interpretation import door_clear
from room3d_preview import render
from room3d_refine import depth_error


def fit_landing(room, recipe, stair_index, depths, out, generate_mesh,
                heldout_cameras=(), candidate_depths=(400, 800, 1200, 2000),
                depth_reference=None):
    """Choose largest training-safe landing, then validate held-out views once.

    Preserve the fitted flight exactly. Require designated walkable contact at
    both ends, known clear sampled headroom, preserved visible stair pixels and
    <=0.005 scene-depth degradation against depth_reference (or the input scene).
    An estimated slab is not independent proof
    of an original upper floor, wall support, or runtime traversal.
    """
    out.mkdir(parents=True, exist_ok=True)
    stairs = recipe["primitives"][stair_index]
    if stairs.get("kind") != "staircase" or "shape" not in stairs:
        raise ValueError("Expected a parametric staircase")
    camera_ids = {c["index"] for c in room["cameras"]}
    heldout = set(heldout_cameras)
    if not heldout <= camera_ids or heldout == camera_ids:
        raise ValueError("Reserve existing cameras and leave at least one training view")
    if len(depths) != len(room["cameras"]):
        raise ValueError("Camera/depth count mismatch")
    lengths = sorted(set(map(float, candidate_depths)))
    if not lengths or not np.isfinite(lengths).all() or min(lengths) <= 200:
        raise ValueError("Landing depths must be finite and exceed the 200-unit exit probe")
    group = f"primitive_{stair_index}"
    gallery_group = f"primitive_{len(recipe['primitives'])}"
    baseline = out / "before.obj"
    generate_mesh(room, baseline, recipe)
    triangles, groups = support_mesh(baseline)
    walkable = {"floor", *recipe.get("walkable_surface_groups", [])}
    before_contact = stair_connections(stairs, triangles, groups=groups,
                                       excluded_groups={group}, walkable_groups=walkable)
    report = {"accepted": False, "trials": [], "selected": None,
              "training_cameras": sorted(camera_ids - heldout),
              "heldout_cameras": sorted(heldout), "before_contact": before_contact,
              "qualification": "Estimated visual landing; support and runtime traversal unverified"}
    if before_contact["walkable_valid"]:
        report["reason"] = "already_connected"
        return recipe, report
    if not before_contact["ends"]["bottom"]["walkable"]:
        report["reason"] = "base_not_designated_walkable"
        return recipe, report
    cached = {}
    for camera, (target, valid) in zip(room["cameras"], depths, strict=True):
        if camera["index"] in heldout:
            continue
        scene = render(baseline, camera, return_depth=True)[2]
        part = render(baseline, camera, return_depth=True, only_groups={group})[2]
        reference = render(depth_reference, camera, return_depth=True)[2] if depth_reference else scene
        cached[camera["index"]] = (scene, part, depth_error(reference, target, valid), target, valid)

    def evaluate(path, camera, data):
        old, part, reference_error, target, valid = data
        scene = render(path, camera, return_depth=True)[2]
        visible = np.isfinite(part) & (part <= old + 50)
        retained = visible & (part <= scene + 50)
        fraction = float(retained.sum() / visible.sum()) if visible.any() else None
        delta = depth_error(scene, target, valid) - reference_error
        return {"camera": camera["index"], "depth_delta": delta,
                "retained_visible_stair_fraction": fraction,
                "valid": bool(delta <= .005 and (fraction is None or fraction >= .9))}

    eligible = []
    for length in lengths:
        candidate = copy.deepcopy(recipe)
        landing = landing_for_stairs(stairs, length)
        candidate["primitives"].append(landing)
        path = out / f"landing{len(report['trials'])}.obj"
        generate_mesh(room, path, candidate)
        triangles, groups = support_mesh(path)
        contacts = stair_connections(stairs, triangles, groups=groups,
                                     excluded_groups={group},
                                     walkable_groups=walkable | {gallery_group})
        overhead = {f"primitive_{i}" for i, p in enumerate(candidate["primitives"])
                    if p["kind"] in ("floor", "landing", "ceiling")}
        headroom = stair_headroom(stairs, triangles, groups, overhead)
        training = [evaluate(path, c, cached[c["index"]]) for c in room["cameras"]
                    if c["index"] not in heldout]
        gates = {"walkable_contacts": contacts["walkable_valid"],
                 "known_headroom": headroom["clear"],
                 "door_clearance": door_clear(landing, recipe),
                 "training_views": all(s["valid"] for s in training)}
        trial = {"depth": length, "gates": gates, "training": training,
                 "contacts": contacts, "headroom": headroom}
        report["trials"].append(trial)
        if all(gates.values()):
            eligible.append((len(report["trials"]) - 1, candidate, path))
    if not eligible:
        report["reason"] = "no_training_safe_landing"
        return recipe, report
    index, candidate, path = eligible[-1]
    report["selected"] = index
    validation = []
    for camera, (target, valid) in zip(room["cameras"], depths, strict=True):
        if camera["index"] not in heldout:
            continue
        old = render(baseline, camera, return_depth=True)[2]
        part = render(baseline, camera, return_depth=True, only_groups={group})[2]
        reference = render(depth_reference, camera, return_depth=True)[2] if depth_reference else old
        validation.append(evaluate(path, camera, (old, part, depth_error(reference, target, valid), target, valid)))
    report["heldout"] = validation
    report["accepted"] = all(s["valid"] for s in validation)
    report["reason"] = "creative_landing_selected" if report["accepted"] else "heldout_rejected"
    return (candidate if report["accepted"] else recipe), report
