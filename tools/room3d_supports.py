"""Joint hypotheses for stairs and nearby provisional collision-box supports.

No room identifiers or manually chosen elevations. A box is only a candidate
support; image agreement and physical contact still require verification.
"""

import copy
import json

import numpy as np


def hypotheses(room, recipe, training):
    floor = float(recipe.get("floor_y", 0))
    yield recipe, {"kind": "floor", "y": floor}
    points = np.concatenate([obs["points"] for obs in training])
    for boundary in room["boundaries"]:
        index = boundary["index"]
        if (
            boundary["type"] & 255 not in (1, 5)
            or boundary["flags"] & 0x100
            or index in recipe.get("skip_boundaries", [])
            or index in recipe.get("shared_wall_boundaries", [])
        ):
            continue
        lo = np.array([boundary["xmin"], boundary["zmin"]])
        hi = np.array([boundary["xmax"], boundary["zmax"]])
        inside = ((points[:, [0, 2]] >= lo) & (points[:, [0, 2]] <= hi)).all(1)
        if inside.sum() < 30 or inside.mean() < 0.2:
            continue
        inferred_height = floor - float(np.percentile(points[inside, 1], 90))
        prior = recipe.get("boundary_heights", {}).get(
            str(index), recipe.get("obstacle_height", 1500)
        )
        heights = sorted(
            {float(prior), *(round(inferred_height * f) for f in (0.75, 1, 1.25))}
        )
        for height in heights:
            if not 300 <= height <= 6000:
                continue
            candidate = copy.deepcopy(recipe)
            candidate.setdefault("boundary_heights", {})[str(index)] = height
            yield (
                candidate,
                {
                    "kind": "collision_box",
                    "boundary": index,
                    "y": floor - height,
                    "xmin": float(lo[0]),
                    "zmin": float(lo[1]),
                    "xmax": float(hi[0]),
                    "zmax": float(hi[1]),
                    "qualification": "Estimated support, selected from training observations",
                },
            )


def contact(obj, support):
    if obj["kind"] != "staircase":
        return True
    vertices = np.asarray(obj["parts"], float).reshape(-1, 3)
    if abs(vertices[:, 1].max() - support["y"]) > 1e-5:
        return False
    if support["kind"] == "floor":
        return True
    # All solid-step bases must fit the support footprint. Prevent search from
    # moving a supported initialization off its platform or through its edge.
    base = vertices[np.isclose(vertices[:, 1], support["y"], atol=1e-5, rtol=0)]
    return bool(
        (
            (base[:, 0] >= support["xmin"] - 1e-5)
            & (base[:, 0] <= support["xmax"] + 1e-5)
            & (base[:, 2] >= support["zmin"] - 1e-5)
            & (base[:, 2] <= support["zmax"] + 1e-5)
        ).all()
    )


def fit(room, recipe, group, depths, out, generate_mesh):
    from room3d_preview import render
    from room3d_refine import depth_error
    from room3d_shape_fit import fit_group

    training = group[:-1] if len(group) >= 3 else group
    cameras = {c["index"]: (c, d) for c, d in zip(room["cameras"], depths, strict=True)}
    trials, solutions = [], []
    candidates = list(hypotheses(room, recipe, training))
    for index, (candidate, support) in enumerate(candidates):
        dest = out / f"support{index}"
        obj, report = fit_group(
            room, candidate, group, depths, dest, generate_mesh, support=support
        )
        # Only branch from the unchanged floor candidate. This bounds the
        # search and prevents recursively shortening an already altered room.
        if index == 0 and recipe.get("creative_fallback", False):
            for altered, evidence in occlusion_hypotheses(recipe, report, training):
                candidates.append((altered, {**support, "visual_correction": evidence}))
        eligible = obj is not None and contact(obj, support)
        selection = None
        if eligible:
            combined = copy.deepcopy(candidate)
            combined.setdefault("primitives", []).append(obj)
            path = dest / "combined.obj"
            generate_mesh(room, path, combined)
            errors = []
            for obs in training:
                camera, (target, valid) = cameras[obs["camera"]]
                errors.append(
                    depth_error(
                        render(path, camera, return_depth=True)[2], target, valid
                    )
                )
            selection = float(
                np.mean([s["loss"] for s in report["after"]]) + 0.2 * np.mean(errors)
            )
            solutions.append((selection, obj, report, candidate, support))
        trial = {
            "support": support,
            "contact_valid": bool(eligible),
            "selection_loss": selection,
            "report": report,
        }
        (dest / "result.json").write_text(
            json.dumps({"object": obj, **trial}, indent=2)
        )
        trials.append({k: v for k, v in trial.items() if k != "report"})
    if not solutions:
        return (
            None,
            {"status": "no_supported_candidate", "support_trials": trials},
            recipe,
        )
    raised_searches = []
    if (recipe.get("raised_support_occlusion", False)
            and recipe.get("creative_fallback", False)):
        from room3d_supported_occlusion import refine_support_candidates

        solutions, raised_searches = refine_support_candidates(
            room, solutions, group, depths,
            out / "raised-occlusion", generate_mesh,
        )
    _, obj, report, candidate, support = min(solutions, key=lambda s: s[0])
    report.update(support=support, support_trials=trials)
    if raised_searches:
        report["raised_support_search"] = raised_searches
    if recipe.get("creative_fallback", False):
        candidate, headroom_report = fit_headroom(
            room, candidate, obj, training, depths, out / "headroom", generate_mesh
        )
        report["headroom_search"] = headroom_report
    return obj, report, candidate


def fit_headroom(room, recipe, obj, training, depths, out, generate_mesh):
    """Choose a local overhead proposal on training views; final gates run later."""
    from room3d_assemblies import stair_ceiling_patches, stair_proportions
    from room3d_connections import stair_headroom, support_mesh
    from room3d_preview import render
    from room3d_refine import depth_error
    from room3d_shape_fit import compare, tread_edge_error

    report = {"trials": [], "selected": None,
              "training_cameras": [obs["camera"] for obs in training],
              "qualification": "Training-selected enclosure; final all-camera checks required"}
    if not stair_proportions(obj)["plausible"]:
        return recipe, report
    out.mkdir(exist_ok=True)
    cameras = {c["index"]: (c, d) for c, d in zip(room["cameras"], depths, strict=True)}
    overhead = {f"primitive_{i}" for i, p in enumerate(recipe["primitives"])
                if p.get("kind") == "ceiling"}

    def measure(candidate, name):
        combined = copy.deepcopy(candidate)
        combined["primitives"].append(obj)
        path = out / f"{name}.obj"
        generate_mesh(room, path, combined)
        triangles, groups = support_mesh(path)
        headroom = stair_headroom(obj, triangles, groups, overhead)
        scores, errors = [], []
        for obs in training:
            camera, (target, valid) = cameras[obs["camera"]]
            scene = render(path, camera, return_depth=True)[2]
            part = render(path, camera, return_depth=True,
                          only_groups={f"primitive_{len(recipe['primitives'])}"})[2]
            score = compare(part, scene, obs["mask"], target, valid)
            detail = tread_edge_error(obj, camera, obs, part, scene)
            if detail is not None:
                score["tread_edge_error"] = detail
                score["loss"] = 0.8 * score["loss"] + 0.2 * detail
            scores.append(score)
            errors.append(depth_error(scene, target, valid))
        return {"headroom": headroom, "scores": scores, "depth_errors": errors}

    baseline = measure(recipe, "baseline")
    report["baseline"] = baseline
    if not baseline["headroom"]["obstructed_samples"]:
        return recipe, report
    eligible = []
    for index, (candidate, proposal) in enumerate(stair_ceiling_patches(recipe, obj)):
        result = measure(candidate, f"candidate{index}")
        keep = result["headroom"]["clear"] and all(
            a["loss"] <= b["loss"] + 0.03
            for a, b in zip(result["scores"], baseline["scores"], strict=True)
        ) and all(a <= b + 0.005 for a, b in zip(
            result["depth_errors"], baseline["depth_errors"], strict=True))
        report["trials"].append({"proposal": proposal, **result, "eligible": bool(keep)})
        if keep:
            loss = float(np.mean([s["loss"] for s in result["scores"]])
                         + 0.2 * np.mean(result["depth_errors"]))
            eligible.append((loss, index, candidate))
    if eligible:
        _, report["selected"], recipe = min(eligible, key=lambda item: item[0])
    (out / "results.json").write_text(json.dumps(report, indent=2))
    return recipe, report


def occlusion_hypotheses(recipe, report, training=()):
    """Propose alternatives from depth or repeated treads, never acceptance."""
    known = {s["group"] for s in recipe.get("structural_surfaces", [])}
    creative_cameras = creative_occlusion_views(recipe, report, training)
    structural_cameras = set()
    for obs in training:
        evidence = (obs.get("structural_refinement") or {}).get(
            "structural_evidence", {}
        )
        if evidence.get("tread_edges", 0) >= 5 and evidence.get("correlation", 0) >= 0.9:
            structural_cameras.add(obs["camera"])
    votes = {}
    reasons = {}
    for view in report.get("visibility_diagnostics", []):
        if view["camera"] not in report.get("training_cameras", []):
            continue
        for item in view["occluders"]:
            name = item["group"]
            observed = item.get("observed_depth")
            depth_evidence = (
                item["reliable_pixels"] >= 8
                and observed is not None
                and np.isfinite(observed)
                and observed > 0
                and observed - item["surface_depth"] > max(180, observed * 0.12)
            )
            tread_evidence = (
                view["camera"] in structural_cameras
                and item.get("nearest_pixels", 0) >= 8
            )
            creative_evidence = (
                view["camera"] in creative_cameras
                and item.get("nearest_pixels", 0) >= 50
            )
            if (
                name not in known
                or not name.startswith("boundary_")
                or not name[9:].isdigit()
                or not (depth_evidence or tread_evidence or creative_evidence)
            ):
                continue
            votes[name] = votes.get(name, 0) + item.get(
                "nearest_pixels", item["reliable_pixels"]
            )
            reasons.setdefault(name, set()).update(
                reason for reason, present in (
                    ("depth_discrepancy", depth_evidence),
                    ("repeated_tread_structure", tread_evidence),
                    ("estimated_multiview_mask_layout", creative_evidence),
                ) if present
            )
    groups = sorted(votes, key=lambda name: (-votes[name], name))[:3]
    if not groups:
        return
    heights = (750, 1500, 3000) if creative_cameras else (1500, 3000)
    for height in heights:
        candidate = lower_visual_occluders(recipe, groups, height)
        if candidate != recipe:
            yield candidate, {
                "groups": groups,
                "height": height,
                "evidence_pixels": {name: votes[name] for name in groups},
                "evidence": {name: sorted(reasons[name]) for name in groups},
                "qualification": "Training-view proposal; requires all-camera acceptance",
            }


def creative_occlusion_views(recipe, report, training):
    """Bound visual alternatives for an explicit approximate multi-view design.

    Mask agreement is a proposal cue, not proof of original architecture. Only
    training views count; model-depth and all-camera acceptance gates stay in
    the caller. Single-view or structurally observed stairs keep the old path.
    """
    if not recipe.get("creative_stair_replacement", False) or not report.get(
        "creative_layout_prior"
    ):
        return set()
    cameras = {
        obs["camera"] for obs in training
        if obs.get("approximate_stair_mask")
        and obs.get("kind") == "staircase"
        and obs["camera"] in report.get("training_cameras", [])
    }
    if len(cameras) < 2:
        return set()
    return {
        view["camera"] for view in report.get("visibility_diagnostics", [])
        if view["camera"] in cameras
        and view.get("projected_pixels", 0) >= 150
        and view.get("target_overlap", 0) >= view["projected_pixels"] * 0.6
        and view.get("scene_visible_pixels", view["projected_pixels"])
        <= view["projected_pixels"] * 0.8
    }


def lower_visual_occluders(recipe, groups, height):
    """Propose shorter visual extrusions; never modify source collision records."""
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Visual obstacle height must be finite and positive")
    candidate = copy.deepcopy(recipe)
    floor = float(recipe.get("floor_y", 0))
    groups = set(groups)
    known = {s["group"] for s in recipe.get("structural_surfaces", [])}
    for group in groups:
        if (
            group not in known
            or not group.startswith("boundary_")
            or not group[9:].isdigit()
        ):
            raise ValueError("Expected an existing generated boundary group")
        key = group[9:]
        prior = recipe.get("boundary_heights", {}).get(
            key, recipe.get("wall_height", 6000)
        )
        candidate.setdefault("boundary_heights", {})[key] = min(float(prior), height)
    for surface in candidate.get("structural_surfaces", []):
        if surface["group"] in groups:
            for point in surface["points"]:
                point[1] = max(point[1], floor - height)
    return candidate
