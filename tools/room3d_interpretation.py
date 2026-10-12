"""Deterministic creative detail alternatives, distinct from recovered geometry."""

import copy

import numpy as np
from room3d_shapes import generate


def proposals(reference):
    shape = reference["shape"]
    p = dict(shape["parameters"])
    kind = reference["kind"]
    if kind == "lamp":
        if reference.get("attachment_prior"):
            from room3d_fixtures import anchor_fixture

            for scale in (0.85, 1.15):
                try:
                    candidate = generate(kind, {**p, "radius": p["radius"] * scale},
                                         shape["origin"], shape["yaw"])
                except ValueError:
                    continue
                yield anchor_fixture(candidate, reference["attachment_prior"])
            return
        for arms in (4, 6):
            yield generate(
                kind,
                {
                    **p,
                    "profile": "candelabra",
                    "arms": arms,
                    "shade": min(p["shade"], p["radius"] * 0.65),
                },
                shape["origin"],
                shape["yaw"],
            )
    elif kind == "staircase":
        # Preserve flight endpoints and support; choose regular new treads.
        p["steps"] = int(np.clip(round(p["rise"] / 180), 2, 48))
        yield generate(kind, p, shape["origin"], shape["yaw"])
    elif kind == "railing":
        if p.get('profile') in ('handrail', 'handrail_segments'):
            # Keep a continuous-rail interpretation; unseen posts are not
            # implied by a visible wall-mounted rail.
            return
        p["posts"] = int(np.clip(round(p["run"] / 400) + 1, 2, 48))
        yield generate(kind, p, shape["origin"], shape["yaw"])


def placement(part, scene, target):
    visible = np.isfinite(part) & (part <= scene + 50)
    if visible.sum() < 8 or not target.any():
        return {"valid": False, "reason": "not_visible"}
    a, b = np.argwhere(visible), np.argwhere(target)
    displacement = float(np.linalg.norm(a.mean(0) - b.mean(0)))
    ratio = float(len(a) / len(b))
    overlap = float((visible & target).sum() / len(a))
    tolerance = max(8.0, float(np.linalg.norm(np.ptp(b, axis=0))) * 0.3)
    return {
        "valid": bool(
            0.15 <= ratio <= 3 and displacement <= tolerance and overlap >= 0.3
        ),
        "visible_pixels": len(a),
        "area_ratio": ratio,
        "centroid_error": displacement,
        "overlap_fraction": overlap,
    }


def door_clear(obj, recipe):
    box = np.asarray(obj["box"])
    for item in recipe.get("primitives", []):
        if item.get("kind") != "door_panel" or "box" not in item:
            continue
        other = np.asarray(item["box"])
        if np.all(
            np.minimum(box[[2, 3, 5]], other[[2, 3, 5]])
            > np.maximum(box[[0, 1, 4]], other[[0, 1, 4]])
        ):
            return False
    return True


def interpret(
    room, baseline, stripped, group, depths, reference, support, out, generate_mesh
):
    from room3d_preview import render
    from room3d_refine import depth_error
    from room3d_shape_fit import isolated_recipe
    from room3d_supports import contact

    out.mkdir(exist_ok=True)
    report = {
        "path": "procedural_interpretation",
        "style": "classic_mansion",
        "seed": 0,
        "trials": [],
        "accepted": False,
        "qualification": "Creative visual replacement; gameplay collision data is unchanged. Door-panel clearance is not a full traversal test. Lamp ceiling attachment and railing support are inherited, not verified.",
    }
    if reference is None or "shape" not in reference:
        report["reason"] = "no_placement_anchor"
        return None, None, report
    original_path = out / "original.obj"
    generate_mesh(room, original_path, baseline)
    original_depths = [
        render(original_path, c, return_depth=True)[2] for c in room["cameras"]
    ]
    before = [
        depth_error(d, target, valid)
        for d, (target, valid) in zip(original_depths, depths, strict=True)
    ]
    observed = {o["camera"]: o for o in group}
    connection_triangles = None
    if reference["kind"] == "staircase":
        from room3d_connections import stair_connections, support_mesh

        support_path = out / "support-scene.obj"
        generate_mesh(room, support_path, stripped)
        connection_triangles, connection_groups = support_mesh(support_path)
    attachment_geometry = None
    if reference.get("attachment_prior"):
        from room3d_connections import support_mesh

        support_path = out / "attachment-scene.obj"
        generate_mesh(room, support_path, stripped)
        attachment_geometry = support_mesh(support_path)
    for index, obj in enumerate(proposals(reference)):
        obj["interpretation"] = {
            "style": report["style"],
            "seed": 0,
            "reason": "insufficient_evidence_fit",
        }
        obj["evidence"] = copy.deepcopy(reference.get("evidence", {}))
        candidate = copy.deepcopy(stripped)
        candidate.setdefault("primitives", []).append(obj)
        path, isolated = out / f"candidate{index}.obj", out / f"isolated{index}.obj"
        generate_mesh(room, path, candidate)
        generate_mesh(room, isolated, isolated_recipe(room, candidate, obj))
        scores, after = [], []
        for camera, (target, valid) in zip(room["cameras"], depths, strict=True):
            scene = render(path, camera, return_depth=True)[2]
            after.append(depth_error(scene, target, valid))
            if camera["index"] in observed:
                part = render(
                    isolated, camera, return_depth=True, only_groups={"primitive_0"}
                )[2]
                scores.append(placement(part, scene, observed[camera["index"]]["mask"]))
        gates = {
            "placement": bool(scores) and all(s["valid"] for s in scores),
            "depth_consistency": all(
                a <= b + 0.01 for a, b in zip(after, before, strict=True)
            ),
            "door_clearance": door_clear(obj, baseline),
            "stair_support": contact(
                obj, support or {"kind": "floor", "y": baseline.get("floor_y", 0)}
            ),
        }
        connections = None
        headroom = None
        landing_assembly = None
        if attachment_geometry is not None:
            from room3d_fixtures import fixture_contact

            gates["fixture_attachment"] = fixture_contact(obj, *attachment_geometry)
        if connection_triangles is not None:
            connections = stair_connections(
                obj,
                connection_triangles,
                groups=connection_groups,
                excluded_groups={
                    f"primitive_{i}"
                    for i, item in enumerate(stripped.get("primitives", []))
                    if item.get("kind") == "ceiling"
                },
                walkable_groups={"floor", *stripped.get("walkable_surface_groups", [])},
            )
            gates["stair_end_connections"] = connections["valid"]
            if stripped.get("scene_depth_polish", False) or stripped.get("constrain_stair_headroom", False):
                from room3d_connections import stair_headroom

                overhead = {f"primitive_{i}" for i, item in enumerate(stripped.get("primitives", []))
                            if item.get("kind") == "ceiling"}
                headroom = stair_headroom(obj, connection_triangles, connection_groups, overhead)
                gates["headroom_safe"] = headroom["obstructed_samples"] == 0
            if (
                stripped.get("joint_creative_stair_landings", False)
                and stripped.get("creative_stair_landings", False)
                and not gates["stair_end_connections"]
                and all(gates.get(name, True) for name in
                        ("door_clearance", "stair_support", "headroom_safe"))
            ):
                # The fallback cannot wait for acceptance before proposing the
                # very landing required for its missing top-end connection.
                from room3d_landing_fit import fit_landing

                stair_index = len(candidate["primitives"]) - 1
                heldout = [group[-1]["camera"]] if len(group) >= 3 else []
                assembly_out = out / f"candidate{index}-landing"
                assembly_out.mkdir()
                assembled, landing_assembly = fit_landing(
                    room, candidate, stair_index, depths, assembly_out,
                    generate_mesh, heldout_cameras=heldout,
                    depth_reference=original_path,
                )
                if landing_assembly["accepted"]:
                    candidate = assembled
                    obj = candidate["primitives"][stair_index]
                    generate_mesh(room, path, candidate)
                    triangles, groups = support_mesh(path)
                    excluded = {f"primitive_{stair_index}", *(
                        f"primitive_{i}" for i, item in enumerate(candidate["primitives"])
                        if item.get("kind") == "ceiling"
                    )}
                    connections = stair_connections(
                        obj, triangles, groups=groups, excluded_groups=excluded,
                        walkable_groups={"floor", f"primitive_{stair_index + 1}",
                                         *candidate.get("walkable_surface_groups", [])},
                    )
                    gates["stair_end_connections"] = connections["walkable_valid"]
                    # Recheck the whole assembly against the original baseline,
                    # including reserved views, rather than accumulating slack.
                    scores, after = [], []
                    for camera, (target, valid) in zip(room["cameras"], depths, strict=True):
                        scene = render(path, camera, return_depth=True)[2]
                        after.append(depth_error(scene, target, valid))
                        if camera["index"] in observed:
                            part = render(path, camera, return_depth=True,
                                          only_groups={f"primitive_{stair_index}"})[2]
                            scores.append(placement(part, scene, observed[camera["index"]]["mask"]))
                    gates["placement"] = bool(scores) and all(s["valid"] for s in scores)
                    gates["depth_consistency"] = all(
                        a <= b + .005 for a, b in zip(after, before, strict=True)
                    )
        trial = {
            "connections": connections,
            **({"headroom": headroom} if headroom is not None else {}),
            "shape": obj["shape"],
            "placement": scores,
            "gates": gates,
            "depth_before": before,
            "depth_after": after,
            **({"landing_assembly": landing_assembly} if landing_assembly is not None else {}),
        }
        report["trials"].append(trial)
        if all(gates.values()):
            report.update(accepted=True, selected=index)
            if landing_assembly is not None and landing_assembly["accepted"]:
                report["landing_assembly"] = landing_assembly
            return obj, candidate, report
    return None, None, report
