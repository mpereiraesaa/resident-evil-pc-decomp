"""Shared object observations and bounded render-based fitting of shape priors.

Scores quantify agreement with masks/estimated depth, not recovered fidelity.
No room identifiers, textures, or hand-authored per-room dimensions are used.
"""

import copy
import hashlib
import math

import cv2
import numpy as np
from room3d_preview import camera_basis, render
from room3d_shapes import generate
from room3d_stair_edges import STAIR_LABELS


def view_loss(scores, balanced=False):
    """Score training views; optionally give the worst view equal aggregate weight."""
    losses = [score["loss"] for score in scores]
    if not losses:
        raise ValueError("At least one training view is required")
    mean = float(np.mean(losses))
    return .5 * (mean + max(losses)) if balanced else mean


def observations(room, segments, depths, creative_stair_replacement=False):
    result = []
    for camera, entries, (depth, valid) in zip(
        room["cameras"], segments, depths, strict=True
    ):
        shadowed = {name for item, mask in entries
                    if item.get('fixture_recovery') and item['label'].strip().casefold() == 'ceiling fan'
                    and np.isfinite([item['score'], item['predicted_iou']]).all()
                    and item['score'] >= .3 and item['predicted_iou'] >= .8 and mask.sum() >= 24
                    for name in item['fixture_recovery'].get('superseded_masks', [])}
        for detection, mask in entries:
            if detection['mask'] in shadowed:
                continue
            confidence = [detection['score'], detection['predicted_iou']]
            if not np.isfinite(confidence).all() or not all(0 <= v <= 1 for v in confidence):
                continue
            refinement = detection.get("structural_refinement", {})
            structural = (
                refinement.get("accepted_proposal", False)
                and refinement.get("structural_evidence", {}).get("tread_edges", 0) >= 5
                and detection["predicted_iou"] >= 0.2
            )
            approximate_stair = (creative_stair_replacement
                                 and not structural
                                 and detection['label'].strip().casefold() in STAIR_LABELS
                                 and .65 <= detection['predicted_iou'] < .8
                                 and mask.sum() >= 150)
            if detection["score"] < 0.3 or (
                detection["predicted_iou"] < 0.8 and not structural and not approximate_stair
            ):
                continue
            label = detection["label"]
            kinds = [
                k
                for k, words in [
                    ("staircase", ("staircase",)),
                    ("railing", ("railing",)),
                    ("lamp", ("lamp", "chandelier", "ceiling fan")),
                ]
                if any(w in label for w in words)
                and not (k == "staircase" and "railing" in label)
            ]
            yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
            if len(xx) < 24:
                continue
            z = depth[yy, xx]
            points = np.column_stack(
                ((xx - 160) * z / camera["fov"], (yy - 120) * z / camera["fov"], z)
            )
            points = points @ camera_basis(camera) + camera["origin"]
            tread_points = None
            polygon = refinement.get("structural_evidence", {}).get("polygon")
            if polygon is not None:
                region = np.zeros(mask.shape, "uint8")
                cv2.fillConvexPoly(region, np.rint(polygon).astype("int32"), 1)
                selected = points[region[yy, xx].astype(bool)]
                if len(selected) >= 24:
                    tread_points = selected
            lo, hi = np.percentile(points, [5, 95], axis=0)
            for kind in kinds:
                result.append(
                    {
                        "kind": kind,
                        "camera": camera["index"],
                        "camera_model": camera,
                        "mask": mask,
                        "mask_name": detection["mask"],
                        "mask_names": [detection["mask"], *detection.get('fixture_recovery', {}).get('superseded_masks', [])],
                        "points": points,
                        "tread_points": tread_points,
                        "lo": lo,
                        "hi": hi,
                        "label": label,
                        "structural_refinement": refinement or None,
                        "predicted_iou": detection["predicted_iou"],
                        'confidence_evidence': [{'mask': detection['mask'],
                            'mask_pixels_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                            'sam_prediction_source': 'refined_mask' if refinement.get('accepted_proposal') else 'source_mask',
                            'detector_score': float(detection['score']),
                            'predicted_iou': float(detection['predicted_iou']),
                            **({'original_predicted_iou': float(detection['original_predicted_iou'])}
                               if 'original_predicted_iou' in detection
                               and np.isfinite(detection['original_predicted_iou'])
                               and 0 <= detection['original_predicted_iou'] <= 1 else {})}],
                        **({'approximate_stair_mask': True} if approximate_stair else {}),
                    }
                )
    return merge_duplicate_masks(result)


def merge_duplicate_masks(items):
    """Merge overlapping detections of one object without joining nearby lamps."""
    result = []
    for item in items:
        matches = []
        for other in result:
            if item["camera"] != other["camera"] or item["kind"] != other["kind"]:
                continue
            intersection = int((item["mask"] & other["mask"]).sum())
            areas = [int(item["mask"].sum()), int(other["mask"].sum())]
            if (
                min(areas)
                and max(areas) / min(areas) <= 2.5
                and intersection / min(areas) >= 0.85
            ):
                matches.append(other)
        if len(matches) != 1:
            result.append(copy.deepcopy(item))
            continue
        other = matches[0]
        other["mask"] |= item["mask"]
        other["mask_names"] = sorted(set(other["mask_names"] + item["mask_names"]))
        if item.get('confidence_evidence'):
            other.setdefault('confidence_evidence', []).extend(copy.deepcopy(item['confidence_evidence']))
        other["points"] = np.unique(
            np.concatenate([other["points"], item["points"]]), axis=0
        )
        other["lo"], other["hi"] = np.percentile(other["points"], [5, 95], axis=0)
    return result


def compatible(a, b):
    if a["kind"] != b["kind"] or a["camera"] == b["camera"]:
        return False
    # A modest uncertainty envelope accounts for predicted depth bias, but
    # complete-link clustering prevents a chain from merging separate lamps.
    alo, ahi = a["lo"] - 150, a["hi"] + 150
    blo, bhi = b["lo"] - 150, b["hi"] + 150
    intersection = np.maximum(0, np.minimum(ahi, bhi) - np.maximum(alo, blo))
    overlap = np.prod(intersection) / max(
        1, min(np.prod(ahi - alo), np.prod(bhi - blo))
    )
    if overlap < 0.2:
        return False
    # Depth-space overlap alone can join different fixtures after scale shifts.
    # Lamp mask centers must also admit a finite, consistent camera-ray solution.
    if a["kind"] == "lamp" and all("camera_model" in o for o in (a, b)):
        return triangulate_centers([a, b]) is not None
    return True


def associate(items):
    groups = []
    for item in sorted(items, key=lambda a: (a["kind"], a["camera"], a["mask_name"])):
        candidates = [g for g in groups if all(compatible(item, other) for other in g)]
        if len(candidates) == 1:
            candidates[0].append(item)
        else:
            # Ambiguous matches stay independent, rather than merging furniture.
            groups.append([item])
    return groups


def tread_run_direction(group):
    """Infer horizontal run perpendicular to tread lines, using known cameras.

    A line's backprojection plane intersects a horizontal plane in the tread
    width direction. This orientation does not depend on monocular depth scale.
    """
    widths = []
    for obs in group:
        evidence = (obs.get("structural_refinement") or {}).get(
            "structural_evidence", {}
        )
        camera = obs.get("camera_model")
        if camera is None:
            continue
        for a, b in evidence.get("line_segments", []):
            line = np.cross([*a, 1], [*b, 1])
            normal = np.array(
                [line[0] * camera["fov"], line[1] * camera["fov"], line @ [160, 120, 1]]
            ) @ camera_basis(camera)
            width = np.cross(normal, [0, 1, 0])[[0, 2]]
            length = np.linalg.norm(width)
            if length > 1e-8:
                widths.append(width / length)
    if len(widths) < 5:
        return None
    values, vectors = np.linalg.eigh(np.asarray(widths).T @ np.asarray(widths))
    if values[0] > values[1] * 0.1:
        return None
    width = vectors[:, -1]
    return np.array([-width[1], width[0]])


def tread_direction_error(obj, direction):
    """Bounded axial orientation disagreement; flight ascent remains ambiguous."""
    if obj['kind'] != 'staircase' or direction is None:
        return None
    yaw = obj['shape']['yaw']
    run = np.array([math.sin(yaw), math.cos(yaw)])
    # Direction comes from horizontal image lines, independently of depth.
    # The sign of a horizontal line does not identify the ascending end.
    return float(np.clip(1 - np.dot(run, direction) ** 2, 0, 1))


def seed(group, use_treads=False):
    points = np.concatenate(
        [
            p[:: max(1, len(p) // 1000)]
            for o in group
            for p in [
                o["tread_points"]
                if use_treads and o.get("tread_points") is not None
                else o["points"]
            ]
        ]
    )
    if use_treads and group[0]["kind"] == "staircase":
        run_direction = tread_run_direction(group)
        if run_direction is not None:
            across = np.array([run_direction[1], -run_direction[0]])
            along = points[:, [0, 2]] @ run_direction
            if np.ptp(along) < 500:
                return None
            slope = np.polyfit(along, points[:, 1], 1)[0]
            if slope > 0:
                run_direction = -run_direction
                along = -along
            lo, hi = np.percentile(along, [5, 95])
            cross_lo, cross_hi = np.percentile(points[:, [0, 2]] @ across, [5, 95])
            base = run_direction * lo + across * ((cross_lo + cross_hi) / 2)
            rise = float(np.clip(abs(slope) * (hi - lo), 100, 7000))
            return generate(
                "staircase",
                {
                    "width": float(np.clip(cross_hi - cross_lo, 400, 4000)),
                    "run": float(hi - lo),
                    "rise": rise,
                    "steps": int(np.clip(round(rise / 220), 2, 48)),
                    "landing": 0,
                },
                [base[0], float(np.percentile(points[:, 1], 95)), base[1]],
                math.atan2(run_direction[0], run_direction[1]),
            )
    lo, hi = np.percentile(points, [5, 95], axis=0)
    span = hi - lo
    axis = 0 if span[0] > span[2] else 2
    cross = 2 if axis == 0 else 0
    center = (lo + hi) / 2
    yaw = math.pi / 2 if axis == 0 else 0
    kind = group[0]["kind"]
    if kind in ("staircase", "railing"):
        if span[axis] < 500:
            return None
        slope, _ = np.polyfit(points[:, axis], points[:, 1], 1)
        rise = float(np.clip(abs(slope) * span[axis], 100, 7000))
        origin = center.copy()
        origin[axis] = lo[axis] if slope < 0 else hi[axis]
        origin[1] = hi[1]
        if slope >= 0:
            yaw += math.pi
        if kind == "staircase":
            params = {
                "width": float(np.clip(span[cross], 400, 4000)),
                "run": float(span[axis]),
                "rise": rise,
                "steps": int(np.clip(round(rise / 220), 2, 48)),
                "landing": 0,
            }
        else:
            params = {
                "run": float(span[axis]),
                "rise": rise,
                "height": 900.0,
                "thickness": 45.0,
                "posts": int(np.clip(round(span[axis] / 450) + 1, 2, 32)),
            }
    else:
        if max(span) > 4000:
            return None
        origin = [center[0], hi[1], center[2]]
        params = {
            "height": float(max(150, span[1])),
            "radius": float(max(80, max(span[0], span[2]) / 2)),
            "thickness": 35.0,
            "arms": 3,
            "shade": float(max(80, min(350, max(span[0], span[2]) / 3))),
        }
        yaw = 0
    return generate(kind, params, origin, yaw)


def triangulate_centers(group):
    """Intersect mask-center rays; reject weak baselines and inconsistent views."""
    if len(group) < 2:
        return None
    origins, directions = [], []
    for obs in group:
        camera = obs["camera_model"]
        y, x = np.nonzero(obs["mask"])
        ray = np.array(
            [
                (np.median(x) - 160) / camera["fov"],
                (np.median(y) - 120) / camera["fov"],
                1.0,
            ]
        ) @ camera_basis(camera)
        origins.append(np.asarray(camera["origin"], float))
        directions.append(ray / np.linalg.norm(ray))
    projectors = [np.eye(3) - np.outer(d, d) for d in directions]
    matrix = sum(projectors)
    if np.linalg.cond(matrix) > 1e4:
        return None
    center = np.linalg.solve(
        matrix, sum(p @ o for p, o in zip(projectors, origins, strict=True))
    )
    for obs, origin, direction, projection in zip(
        group, origins, directions, projectors, strict=True
    ):
        distance = np.dot(center - origin, direction)
        if distance <= 0:
            return None
        y, x = np.nonzero(obs["mask"])
        # Mask centers are only approximate correspondences, particularly under
        # foreshortening. Bound ray residual in image units relative to extent.
        tolerance_px = max(3.0, 0.15 * math.hypot(np.ptp(x), np.ptp(y)))
        error_px = (
            np.linalg.norm(projection @ (center - origin))
            * obs["camera_model"]["fov"]
            / distance
        )
        if error_px > tolerance_px:
            return None
        view = (center - origin) @ camera_basis(obs["camera_model"]).T
        if view[2] <= obs["camera_model"]["fov"] * 2:
            return None
    return center


def initial_shapes(room, group, floor_y=0, ceiling_fans=False):
    """Independent hypotheses prevent a single noisy depth cloud fixing topology."""
    rough = seed(group)
    kind = group[0]["kind"]
    if rough is not None and kind != "staircase":
        yield rough
    if kind == "staircase":
        # Monocular depth may place a convincing flight in mid-air. Intersect
        # its lowest observed band with the room floor and scale the hypothesis
        # along the source rays, preserving its initial projected extent.
        for use_treads in (False, True):
            for obs in group:
                if use_treads and (
                    obs.get("tread_points") is None
                    or tread_run_direction([obs]) is None
                ):
                    continue
                points = (
                    obs["tread_points"]
                    if use_treads and obs.get("tread_points") is not None
                    else obs["points"]
                )
                bottom = np.median(
                    points[points[:, 1] >= np.percentile(points[:, 1], 90)], axis=0
                )
                camera_origin = np.asarray(obs["camera_model"]["origin"], float)
                denominator = bottom[1] - camera_origin[1]
                if abs(denominator) < 1:
                    continue
                scale = (floor_y - camera_origin[1]) / denominator
                if not 0.25 <= scale <= 4:
                    continue
                anchored = dict(obs)
                anchored.pop("tread_points", None)
                anchored["points"] = camera_origin + (points - camera_origin) * scale
                proposal = seed([anchored], use_treads=use_treads)
                if proposal is not None:
                    shape = proposal["shape"]
                    origin = list(shape["origin"])
                    origin[1] = floor_y
                    obj = generate(kind, shape["parameters"], origin, shape["yaw"])
                    obj["floor_contact_prior"] = (
                        "Lowest observed band assumed to contact the shared floor; upper landings remain inferred"
                    )
                    obj["initialization"] = (
                        "tread_region" if use_treads else "mask_depth"
                    )
                    yield obj
        from room3d_objects import stair_proposals

        for obs in group:
            for prior in stair_proposals(
                room, obs["camera_model"], {"label": obs["label"]}, obs["points"]
            ):
                x0, z0, x1, z1, top, base = prior["box"]
                along_x = prior["axis"] == "x"
                run, width = (x1 - x0, z1 - z0) if along_x else (z1 - z0, x1 - x0)
                positive = prior["rise_positive"]
                origin = [
                    (x0 if positive else x1) if along_x else (x0 + x1) / 2,
                    base,
                    (z0 + z1) / 2 if along_x else (z0 if positive else z1),
                ]
                yaw = (math.pi / 2 if along_x else 0) + (0 if positive else math.pi)
                params = {
                    "width": width,
                    "run": run,
                    "rise": base - top,
                    "steps": int(np.clip(round((base - top) / 220), 2, 64)),
                    "landing": 0,
                }
                yield generate(kind, params, origin, yaw)
    elif kind == "lamp":
        triangulated = triangulate_centers(group)
        for obs in group:
            camera = obs["camera_model"]
            rotation = camera_basis(camera)
            view_points = (obs["points"] - camera["origin"]) @ rotation.T
            z = float(np.median(view_points[:, 2]))
            if triangulated is not None:
                z = float(((triangulated - camera["origin"]) @ rotation.T)[2])
            y, x = np.nonzero(obs["mask"])
            # Thin fixture depth often contains background values. Initialize
            # its extent on a plane; do not treat that plane as final geometry.
            flat = np.column_stack(
                (
                    (x - 160) * z / camera["fov"],
                    (y - 120) * z / camera["fov"],
                    np.full(len(x), z),
                )
            )
            points = flat @ rotation + camera["origin"]
            lo, hi = np.percentile(points, [2, 98], axis=0)
            center = triangulated if triangulated is not None else (lo + hi) / 2
            height = float(np.clip((hi[1] - lo[1]) / 0.65, 150, 3000))
            radius = float(np.clip(max(hi[0] - lo[0], hi[2] - lo[2]) / 2, 80, 1200))
            if ceiling_fans:
                for blades in (3, 4, 5):
                    fan_height = max(height, radius * .4)
                    yield generate(kind, {
                        "profile": "ceiling_fan", "arms": blades,
                        "height": fan_height, "radius": radius,
                        "thickness": radius * .02, "blade_width": radius * .18,
                        "shade": min(fan_height * .2, radius * .2),
                    }, [center[0], center[1] + fan_height * .35, center[2]])
            for arms in (1, 3, 4, 5):
                params = {
                    "height": height,
                    "radius": radius,
                    "thickness": 35.0,
                    "arms": arms,
                    "shade": float(np.clip(radius * 0.6, 80, 450)),
                }
                for profile in ("shade", "candle"):
                    yield generate(
                        kind,
                        {**params, "profile": profile},
                        [center[0], center[1] + height * 0.675, center[2]],
                        0,
                    )


def variants(obj, scale=1.0):
    shape = obj["shape"]
    p, origin, yaw = shape["parameters"], shape["origin"], shape["yaw"]
    for axis in range(3):
        if obj["kind"] == "staircase" and axis == 1:
            continue
        for shift in (-150 * scale, 150 * scale):
            moved = list(origin)
            moved[axis] += shift
            yield generate(obj["kind"], p, moved, yaw)
    for angle in (-0.15 * scale, 0.15 * scale):
        yield generate(obj["kind"], p, origin, yaw + angle)
    for key in p:
        if key == "arms" and p.get("profile") in ("wall_linear", "ceiling_linear", "ceiling_panel"):
            # Tube counts are separate initialized families; do not erase
            # repeated training-strip evidence during continuous fitting.
            continue
        if key in ("steps", "posts", "arms"):
            choices = [p[key] - 1, p[key] + 1]
        elif key in ("profile", "structure", "segments"):
            # Topology families have independent starts; do not swap profile
            # mid-pass and lose the shape each hypothesis is optimizing.
            continue
        elif key == "landing":
            choices = [0, p["run"] * 0.2]
        else:
            choices = [p[key] * (1 - 0.15 * scale), p[key] * (1 + 0.15 * scale)]
        for value in choices:
            changed = dict(p)
            changed[key] = value
            try:
                yield generate(obj["kind"], changed, origin, yaw)
            except ValueError:
                continue


def rebase_variant(candidate, reference, current):
    """Keep accepted changes to other coordinates within the same search pass."""
    proposed, original, latest = (
        obj["shape"] for obj in (candidate, reference, current)
    )
    params = dict(latest["parameters"])
    params.update(
        {
            k: v
            for k, v in proposed["parameters"].items()
            if v != original["parameters"][k]
        }
    )
    origin = [
        new if new != old else live
        for new, old, live in zip(
            proposed["origin"], original["origin"], latest["origin"], strict=True
        )
    ]
    yaw = proposed["yaw"] if proposed["yaw"] != original["yaw"] else latest["yaw"]
    return generate(current["kind"], params, origin, yaw)


def compare(part, scene, target, depth, confident, model_occlusion=True):
    finite = np.isfinite(part)
    visible = finite & (part <= scene + 50)
    # Only confident closer observations explain missing object pixels.
    if model_occlusion:
        visible &= ~(confident & (depth < part - np.maximum(180, depth * 0.12)))
    union = (visible | target).sum()
    iou = float((visible & target).sum() / max(1, union))
    shared = visible & target & confident
    depth_error = (
        float(np.minimum(np.abs(np.log(part[shared] / depth[shared])), 1).mean())
        if shared.any()
        else 1.0
    )

    def edge(mask):
        return mask & ~cv2.erode(mask.astype("uint8"), np.ones((3, 3), "uint8")).astype(
            bool
        )

    a, b = edge(visible), edge(target)
    if a.any() and b.any():
        da = cv2.distanceTransform((~a).astype("uint8"), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((~b).astype("uint8"), cv2.DIST_L2, 3)
        edge_error = min(1.0, float((da[b].mean() + db[a].mean()) / 40))
    else:
        edge_error = 1.0
    return {
        "loss": 0.65 * (1 - iou) + 0.2 * edge_error + 0.15 * depth_error,
        "silhouette_iou": iou,
        "edge_error": edge_error,
        "depth_error": depth_error,
        "visible_pixels": int(visible.sum()),
    }


def tread_edge_error(obj, camera, obs, part, scene, constrain_extent=False):
    """Compare visible tread noses with detected internal lines, not the outline.

    Missing cached line evidence leaves the original objective unchanged. These
    lines are image hypotheses, not verified tread counts or ground truth.
    """
    evidence = (obs.get("structural_refinement") or {}).get("structural_evidence", {})
    lines = np.asarray(evidence.get("line_segments", []), float)
    if obj["kind"] != "staircase" or lines.size == 0:
        return None
    if lines.ndim != 3 or lines.shape[1:] != (2, 2) or not np.isfinite(lines).all():
        raise ValueError("Invalid tread line evidence")
    if len(lines) < 5:
        return None
    target = np.zeros(part.shape, "uint8")
    for a, b in np.rint(lines).astype(int):
        cv2.line(target, tuple(a), tuple(b), 1)
    target = target.astype(bool) & obs["mask"]
    predicted = np.zeros(part.shape, bool)
    steps = obj["shape"]["parameters"]["steps"]
    # Prism vertices 0 and 1 are the upper front edge of each step.
    noses = np.asarray(obj["parts"], float)[:steps, :2]
    for a, b in noses:
        points = a + np.linspace(0, 1, 256)[:, None] * (b - a)
        view = (points - camera["origin"]) @ camera_basis(camera).T
        view = view[view[:, 2] >= 2 * camera["fov"]]
        xy = np.rint(view[:, :2] * camera["fov"] / view[:, 2:] + [160, 120]).astype(int)
        inside = ((xy >= 0) & (xy < [part.shape[1], part.shape[0]])).all(1)
        xy, z = xy[inside], view[inside, 2]
        x, y = xy.T
        visible = (z <= part[y, x] + 50) & (z <= scene[y, x] + 50)
        predicted[y[visible], x[visible]] = True
    # Cropping predicted edges to the target hides excessive tread widths.
    # Keep the legacy objective unless explicitly evaluating this new prior.
    if not constrain_extent:
        predicted &= obs["mask"]
    if not target.any() or not predicted.any():
        return 1.0
    distances = [
        cv2.distanceTransform((~mask).astype("uint8"), cv2.DIST_L2, 3)
        for mask in (target, predicted)
    ]
    mean_distance = float(
        (distances[0][predicted].mean() + distances[1][target].mean()) / 2
    )
    # A hard cutoff creates a flat objective once lines are several pixels
    # apart: a better initialization then cannot earn any detail improvement.
    # Retain ordering at every finite distance; absent edges still score 1.
    return mean_distance / (mean_distance + 6.0)


def visibility_diagnostic(path, camera, part, scene, target, depth, confident):
    """Attribute a failed fit to projection, room occlusion, or depth evidence.

    These measurements guide later corrections; they never authorize removing a
    surface. Collision extrusion and predicted depth can both be wrong.
    """
    projected = np.isfinite(part)
    overlap = projected & target
    scene_visible = projected & (part <= scene + 50)
    depth_visible = projected & ~(
        confident & (depth < part - np.maximum(180, depth * 0.12))
    )
    report = {
        "projected_pixels": int(projected.sum()),
        "target_overlap": int(overlap.sum()),
        "scene_visible_pixels": int(scene_visible.sum()),
        "depth_visible_pixels": int(depth_visible.sum()),
        "combined_visible_pixels": int((scene_visible & depth_visible).sum()),
        "occluders": [],
    }
    if not overlap.any():
        return report
    groups = sorted(
        {
            name
            for line in path.read_text().splitlines()
            if line.startswith("g ")
            for name in line.split()[1:]
        }
    )
    for name in groups:
        surface = render(path, camera, return_depth=True, only_groups={name})[2]
        blocked = overlap & np.isfinite(surface) & (surface < part - 50)
        # Avoid inf-inf arithmetic on background pixels.
        nearest = np.zeros(part.shape, bool)
        nearest[blocked] = np.abs(surface[blocked] - scene[blocked]) < 1
        if not nearest.any():
            continue
        reliable = nearest & confident & np.isfinite(depth) & (depth > 0)
        report["occluders"].append(
            {
                "group": name,
                "nearest_pixels": int(nearest.sum()),
                "surface_depth": float(np.median(surface[nearest])),
                "observed_depth": float(np.median(depth[reliable]))
                if reliable.any()
                else None,
                "reliable_pixels": int(reliable.sum()),
            }
        )
    report["occluders"].sort(key=lambda item: item["nearest_pixels"], reverse=True)
    return report


def isolated_recipe(room, recipe, obj):
    """Exclude architecture from the isolated-object export, not scene scoring."""
    return {
        "floor_y": recipe.get("floor_y", 0),
        "primitives": [obj],
        "skip_boundaries": [b["index"] for b in room["boundaries"]],
        "hide": [],
    }


def fit_group(
    room,
    recipe,
    group,
    depths,
    out,
    generate_mesh,
    rounds=5,
    support=None,
    initial_candidates=None,
    fixed_parameters=(),
    images=None,
):
    """Optimize on training views only; reserve one view when >=3 observe it."""
    out.mkdir(parents=True, exist_ok=True)
    training = group[:-1] if len(group) >= 3 else group
    heldout = group[-1:] if len(group) >= 3 else []
    tread_direction = (tread_run_direction(training)
                       if recipe.get('constrain_tread_direction', False) else None)
    base_y = support["y"] if support is not None else recipe.get("floor_y", 0)
    starts = (
        copy.deepcopy(list(initial_candidates))
        if initial_candidates is not None
        else list(initial_shapes(room, training, base_y,
                                 ceiling_fans=recipe.get('ceiling_fan_fixtures', False)))
    )
    if (initial_candidates is None and group[0]['kind'] == 'railing'
            and recipe.get('handrail_candidates', False)):
        from room3d_railings import handrail_seed, surface_handrail_seed

        handrail = handrail_seed(training)
        if handrail is not None:
            starts.append(handrail)
        wall_scene = out/'handrail-support.obj'
        generate_mesh(room, wall_scene, recipe)
        mounted = surface_handrail_seed(room, training, wall_scene)
        if mounted is not None:
            starts.append(mounted)
    if (
        initial_candidates is None
        and group[0]["kind"] == "staircase"
        and recipe.get("creative_fallback", False)
    ):
        from room3d_levels import stair_level_starts

        starts.extend(
            stair_level_starts(
                training,
                recipe.get("level_hypotheses", {}),
                base_y,
                excluded_cameras=[o["camera"] for o in heldout],
            )
        )
    if not starts and not (recipe.get('creative_stair_replacement', False)
                          and group[0]['kind'] == 'staircase'
                          and any(o.get('approximate_stair_mask', False) for o in training)):
        return None, {"status": "insufficient_extent"}
    if (initial_candidates is None and group[0]["kind"] == "staircase"
            and recipe.get("creative_fallback", False)
            and recipe.get("constrain_stair_proportions", False)
            and recipe.get("feasible_stair_starts", False)):
        from room3d_assemblies import feasible_stair_seed

        # At most one alternative per observation-derived family. Retain all
        # original starts; independently constrained level priors stay intact.
        repaired_families = set()
        for start in list(starts):
            family = start.get("initialization")
            if family not in {"mask_depth", "tread_region"} or family in repaired_families:
                continue
            proposal = feasible_stair_seed(start)
            if proposal is not None:
                starts.append(proposal)
                repaired_families.add(family)
    if (initial_candidates is None and group[0]["kind"] == "staircase"
            and recipe.get("creative_fallback", False)
            and recipe.get("suspended_stair_starts", False)):
        # Bound added structure hypotheses to the observation-derived families;
        # keep every solid seed and leave independent level priors unchanged.
        families = {"mask_depth", "tread_region", "feasible_mask_depth", "feasible_tread_region"}
        added = set()
        for start in list(starts):
            name = start.get("initialization")
            if name not in families or name in added:
                continue
            shape = start["shape"]
            proposal = generate("staircase", {**shape["parameters"],
                                "structure": "suspended", "slab_depth": 100},
                                shape["origin"], shape["yaw"])
            proposal["initialization"] = name + "_suspended"
            proposal["interpretation"] = {
                "style": "classic_mansion", "source": "regular_step_prior",
                "qualification": "Estimated suspended flight; structural support and traversal unverified",
            }
            starts.append(proposal)
            added.add(name)
    creative_axes = None
    if (recipe.get('creative_stair_replacement', False)
            and any(o.get('approximate_stair_mask', False) for o in training)
            and group[0]['kind'] == 'staircase'):
        from room3d_creative_stairs import mask_volume_starts, replacement_starts

        creative_axes = recipe.get('tread_horizontal_axes', [[1, 0], [0, 1]])
        if initial_candidates is None:
            starts = list(replacement_starts(starts, creative_axes))
            starts.extend(mask_volume_starts(room, training, creative_axes, base_y))
        if not starts:
            return None, {'status': 'insufficient_creative_layout_extent'}
    by_camera = {
        c["index"]: (c, d) for c, d in zip(room["cameras"], depths, strict=True)
    }
    scene_path = out / "scene.obj"
    generate_mesh(room, scene_path, recipe)
    if (initial_candidates is None and group[0]['kind'] == 'lamp'
            and recipe.get('ceiling_light_fixtures', False)):
        from room3d_ceiling_linear import mask_starts
        from room3d_connections import support_mesh

        ceilings = {f'primitive_{i}' for i, p in enumerate(recipe.get('primitives', []))
                    if p.get('kind') == 'ceiling'}
        ceilings |= {s['group'] for s in recipe.get('structural_surfaces', [])
                     if s['material'] == 'ceiling'}
        starts.extend(mask_starts(training, *support_mesh(scene_path), ceilings))
    if (
        initial_candidates is None and group[0]["kind"] == "lamp"
        and recipe.get("creative_fallback", False)
    ):
        from room3d_connections import support_mesh
        from room3d_fixtures import ray_wall_lamp_proposals, wall_lamp_proposals

        triangles, surface_groups = support_mesh(scene_path)
        wall_groups = {name for name in surface_groups if name.startswith("boundary_")}
        for index, proposal in enumerate(wall_lamp_proposals(
            starts[0], triangles, surface_groups, wall_groups,
            camera_origins=[obs["camera_model"]["origin"] for obs in training],
        )):
            proposal["initialization"] = f"wall_attachment_{index}"
            starts.append(proposal)
        for index, proposal in enumerate(ray_wall_lamp_proposals(
            starts[0], training, triangles, surface_groups, wall_groups
        )):
            proposal["initialization"] = f"wall_ray_{index}"
            starts.append(proposal)
        if recipe.get("inset_fixture_anchors", False):
            from room3d_fixtures import inset_wall_lamp

            for start in list(starts):
                proposal = inset_wall_lamp(start, triangles, surface_groups)
                if proposal is not None:
                    proposal["initialization"] = start.get("initialization", "wall") + "_inset"
                    starts.append(proposal)
        if recipe.get("multilight_wall_fixtures", False):
            from room3d_fixtures import multilight_wall_proposals, wall_plate_contact

            attached_starts = [s for s in starts if wall_plate_contact(s, triangles, surface_groups)]
            # Prefer mask-ray anchors before nearest-point alternatives. Keep
            # four support seeds and eight topology/size proposals per seed.
            attached_starts.sort(key=lambda s: not s.get("initialization", "").startswith("wall_ray"))
            for seed, start in enumerate(attached_starts[:4]):
                for index, proposal in enumerate(multilight_wall_proposals(start, triangles, surface_groups)):
                    proposal["initialization"] = f"multilight_{seed}_{index}"
                    starts.append(proposal)
        if recipe.get("linear_wall_fixtures", False):
            from room3d_fixtures import linear_wall_proposals, wall_plate_contact

            seeds = [s for s in starts if wall_plate_contact(s, triangles, surface_groups)]
            seeds.sort(key=lambda s: not s.get("initialization", "").startswith("wall_ray"))
            training_cameras = {obs['camera'] for obs in training}
            source_images = ({c['index']: im for c, im in zip(room['cameras'], images, strict=True)
                              if c['index'] in training_cameras} if images is not None else None)
            for seed, start in enumerate(seeds[:4]):
                for index, proposal in enumerate(linear_wall_proposals(start, training, triangles, surface_groups, source_images)):
                    proposal["initialization"] = f"linear_wall_{seed}_{index}"
                    starts.append(proposal)
        from room3d_fixtures import ray_table_lamp_proposals

        furniture_groups = {
            f"primitive_{i}" for i, primitive in enumerate(recipe.get("primitives", []))
            if primitive["kind"] in {"chest_body", "table", "cabinet", "desk"}
        }
        # Non-wall collision blocks supply approximate furniture/platform tops.
        furniture_groups.update(
            f"boundary_{b['index']}" for b in room["boundaries"]
            if not b["flags"] & 0x100 and b["type"] & 255 in (1, 5)
        )
        for index, proposal in enumerate(ray_table_lamp_proposals(
            starts[0], training, triangles, surface_groups, furniture_groups
        )):
            proposal["initialization"] = f"table_ray_{index}"
            starts.append(proposal)
    if recipe.get('ceiling_fan_anchors', False):
        from room3d_connections import support_mesh
        from room3d_fixtures import ceiling_fan_proposals

        triangles, surface_groups = support_mesh(scene_path)
        ceilings = {f'primitive_{i}' for i, p in enumerate(recipe.get('primitives', []))
                    if p.get('kind') == 'ceiling'}
        attached_starts = []
        for start in starts:
            if start['shape']['parameters'].get('profile') == 'ceiling_fan':
                attached_starts.extend(ceiling_fan_proposals(start, triangles, surface_groups, ceilings))
            else:
                attached_starts.append(start)
        starts = attached_starts
        if not starts:
            return None, {'status': 'insufficient_ceiling_attachment'}
    if recipe.get("orientable_wall_shades", False):
        from room3d_fixtures import anchor_fixture

        for start in list(starts):
            params = start["shape"]["parameters"]
            if params.get("profile") != "wall_sconce":
                continue
            for tilt in (-math.pi / 2, math.pi / 2):
                oriented = generate("lamp", {**params, "shade_tilt": tilt,
                                     "radius": max(params["radius"], params["shade"] + params["thickness"] / 2)})
                oriented = anchor_fixture(oriented, start["attachment_prior"])
                oriented["initialization"] = f"{start.get('initialization')}_tilt{tilt}"
                starts.append(oriented)
    scenes = {
        o["camera"]: render(scene_path, by_camera[o["camera"]][0], return_depth=True)[2]
        for o in group
    }
    scene_fit_cameras = []
    scene_baselines = {}
    # A staged polish first resolves mask geometry/support/enclosure, then
    # invokes this search on that fitted shape with the depth objective enabled.
    depth_fit = recipe.get("scene_depth_fit", False) and not recipe.get("scene_depth_polish", False)
    if depth_fit:
        from room3d_refine import depth_error

        excluded = {o["camera"] for o in heldout}
        scene_fit_cameras = [c["index"] for c in room["cameras"] if c["index"] not in excluded]
        for index in scene_fit_cameras:
            camera, (target, valid) = by_camera[index]
            if index not in scenes:
                scenes[index] = render(scene_path, camera, return_depth=True)[2]
            scene_baselines[index] = depth_error(scenes[index], target, valid)
    # Render an isolated object with the same original-camera projection.
    path = out / "candidate.obj"
    attachment_mesh = None
    attachment_checks = {}
    headroom_mesh = None
    overhead_groups = set()
    if recipe.get("constrain_stair_headroom", False) and any(o["kind"] == "staircase" for o in starts):
        from room3d_connections import stair_headroom, support_mesh

        headroom_mesh = support_mesh(scene_path)
        overhead_groups = {f"primitive_{i}" for i, p in enumerate(recipe.get("primitives", []))
                           if p.get("kind") == "ceiling"}
    if any(obj.get("attachment_prior") for obj in starts):
        from room3d_connections import support_mesh

        attachment_mesh = support_mesh(scene_path)

    def evaluate(obj, views, training_loss=True):
        proportions = None
        if obj["kind"] == "staircase" and recipe.get("constrain_stair_proportions", False):
            from room3d_assemblies import stair_proportions

            proportions = stair_proportions(obj)["plausible"]
        minimal = isolated_recipe(room, recipe, obj)
        generate_mesh(room, path, minimal)
        clearance = None
        if headroom_mesh is not None and obj["kind"] == "staircase":
            clearance = stair_headroom(obj, *headroom_mesh, overhead_groups)
        scores = []
        parts = {}
        scene_excess = 0.0
        if training_loss and scene_fit_cameras:
            for index in scene_fit_cameras:
                camera, (target, valid) = by_camera[index]
                parts[index] = render(path, camera, return_depth=True,
                                      only_groups={"primitive_0"})[2]
                combined = np.minimum(parts[index], scenes[index])
                delta = depth_error(combined, target, valid) - scene_baselines[index]
                scene_excess = max(scene_excess, delta - .005)
        attached = None
        if obj.get("attachment_prior") and attachment_mesh is not None:
            from room3d_fixtures import fixture_contact

            # Radius/shade changes leave the same plate on the same immutable
            # scene. Reuse its geometric check across views and shape trials.
            profile = obj["shape"]["parameters"].get("profile")
            key = (obj["attachment_prior"]["group"], profile,
                   np.asarray(obj["parts"][0 if profile in ('ceiling_linear', 'ceiling_panel') else 1], float).tobytes())
            if key not in attachment_checks:
                attachment_checks[key] = fixture_contact(obj, *attachment_mesh)
            attached = attachment_checks[key]
        for obs in views:
            camera, (depth, valid) = by_camera[obs["camera"]]
            part = parts.get(obs["camera"])
            if part is None:
                part = render(path, camera, return_depth=True,
                              only_groups={"primitive_0"})[2]
            score = compare(part, scenes[obs["camera"]], obs["mask"], depth, valid,
                            model_occlusion=creative_axes is None)
            detail = tread_edge_error(obj, camera, obs, part, scenes[obs["camera"]],
                                      constrain_extent=recipe.get("constrain_tread_extent", False))
            if detail is not None:
                score["tread_edge_error"] = detail
                score["loss"] = 0.8 * score["loss"] + 0.2 * detail
            orientation = tread_direction_error(obj, tread_direction)
            if orientation is not None:
                score['tread_direction_error'] = orientation
                score['loss'] += .25 * orientation
            if training_loss and scene_fit_cameras:
                score["scene_depth_excess"] = float(scene_excess)
                score["loss"] += 20 * scene_excess
            if support is not None:
                from room3d_supports import contact

                score["contact_valid"] = contact(obj, support)
                if not score["contact_valid"]:
                    score["loss"] = 2.0
            if attached is not None:
                score["attachment_valid"] = attached
                if not score["attachment_valid"]:
                    score["loss"] = 2.0
            if proportions is not None:
                score["proportions_valid"] = proportions
                if not proportions:
                    score["loss"] = 2.0
            if clearance is not None:
                score["headroom_obstructed_samples"] = clearance["obstructed_samples"]
                score["headroom_unknown_samples"] = clearance["unknown_samples"]
                if clearance["obstructed_samples"]:
                    score["loss"] = 2.0
            if creative_axes is not None:
                from room3d_creative_stairs import replacement_valid

                score['creative_layout_valid'] = replacement_valid(obj, creative_axes)
                if not score['creative_layout_valid']:
                    score['loss'] = 2.0
            scores.append(score)
        return scores

    def objective(scores):
        return view_loss(scores, recipe.get("balanced_view_fit", False))

    starting_scores = [evaluate(obj, training) for obj in starts]
    chosen = int(
        np.argmin([objective(scores) for scores in starting_scores])
    )
    initial = starting_scores[chosen]
    trials = []
    # Discrete topology can trap a single-start search: a five-arm lamp may
    # initially cover a rotated four-arm target better. Optimize each family
    # hypothesis independently, then compare on training views only.
    ranked = sorted(
        range(len(starts)),
        key=lambda i: objective(starting_scores[i]),
    )
    selected, families = [], set()
    profile_counts = {}
    for i in ranked:
        params = starts[i]["shape"]["parameters"]
        key = (
            starts[i].get("initialization"),
            params.get("profile", "shade"),
            params.get("structure", "solid"),
            params.get("arms"),
            params.get("steps"),
            params.get("posts"),
        )
        profile = (params.get("profile", "shade"), params.get("structure", "solid"))
        if key not in families and profile_counts.get(profile, 0) < 4:
            selected.append(i)
            families.add(key)
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
    solutions = []
    for start_index in selected:
        current, best = starts[start_index], starting_scores[start_index]
        frozen = set(fixed_parameters)
        prior = starts[start_index].get("level_prior")
        attachment = starts[start_index].get("attachment_prior")
        if prior:
            frozen.update(("rise", "steps", "run", "landing"))
        for iteration in range(rounds):
            scale = 2.0 ** (1 - iteration)
            reference = current
            for candidate in list(variants(reference, scale)):
                try:
                    candidate = rebase_variant(candidate, reference, current)
                    if attachment:
                        from room3d_fixtures import anchor_fixture

                        if recipe.get("refine_fixture_anchor", False):
                            from room3d_fixtures import slide_fixture_anchor

                            candidate = slide_fixture_anchor(candidate, current)
                        else:
                            candidate = anchor_fixture(candidate, attachment)
                        if candidate["shape"] == current["shape"]:
                            continue
                except ValueError:
                    # A valid independent landing/run move can become invalid
                    # after another coordinate was accepted in this pass.
                    continue
                if any(
                    candidate["shape"]["parameters"].get(key)
                    != reference["shape"]["parameters"].get(key)
                    for key in frozen
                ):
                    continue
                scores = evaluate(candidate, training)
                keep = objective(scores) < objective(best) - 0.0001 and all(
                    a["loss"] <= b["loss"] + 0.03
                    for a, b in zip(scores, best, strict=True)
                )
                trials.append(
                    {
                        "initialization": start_index,
                        "iteration": iteration,
                        "scale": scale,
                        "shape": candidate["shape"],
                        "scores": scores,
                        "accepted": bool(keep),
                    }
                )
                if keep:
                    current, best = candidate, scores
        if (starts[start_index].get("interpretation")
                and (starts[start_index].get("initialization", "").startswith("feasible_")
                     or starts[start_index]["shape"]["parameters"].get("structure") == "suspended")):
            current["interpretation"] = copy.deepcopy(starts[start_index]["interpretation"])
            current["initialization"] = starts[start_index]["initialization"]
        if prior:
            current["level_prior"] = copy.deepcopy(prior)
            current["initialization"] = starts[start_index]["initialization"]
        if attachment:
            current["attachment_prior"] = copy.deepcopy(current.get("attachment_prior", attachment))
            current["interpretation"] = {
                "style": "classic_mansion", "source": "fixture_attachment_prior",
                "qualification": "Supported procedural fixture; original design not recovered",
            }
        solutions.append((current, best, start_index))
    selection_pool = solutions
    if (recipe.get("creative_fallback", False) and recipe.get("single_view_wall_fixtures", False)
            and recipe.get("inset_fixture_anchors", False)
            and len(group) == 1 and group[0]["kind"] == "lamp"):
        attached_solutions = [solution for solution in solutions
                              if solution[0].get("attachment_prior", {}).get("group", "").startswith("boundary_")
                              and all(s.get("attachment_valid") is True
                                      and s.get("visible_pixels", 0) >= 16 for s in solution[1])]
        # A better floating silhouette cannot displace a supported creative
        # proposal that is eligible for the requested single-view path. Final
        # mask/depth/attachment gates still decide whether to adopt anything.
        if attached_solutions:
            selection_pool = attached_solutions
    current, best, selected_start = min(selection_pool, key=lambda pair: objective(pair[1]))
    regularization = None
    if (
        current["kind"] == "staircase"
        and recipe.get("creative_fallback", False)
        and not set(fixed_parameters).intersection(("steps", "landing"))
        and not current.get("level_prior")
    ):
        from room3d_assemblies import regularize_stairs

        current, best, regularization = regularize_stairs(
            current, best, lambda obj: evaluate(obj, training)
        )
    if (current['kind'] == 'railing'
            and current['shape']['parameters'].get('profile') in ('handrail', 'handrail_segments')):
        current['estimated'] = (
            'Procedural visible rail hypothesis; hidden brackets, corner joins, '
            'wall attachment, original fidelity and navigation unverified'
        )
    diagnostics = []
    # Nearly hidden candidates need attribution too: a few stray visible pixels
    # must not suppress the report that explains a failed fit.
    if any(score["silhouette_iou"] < 0.35 for score in best):
        minimal = isolated_recipe(room, recipe, current)
        generate_mesh(room, path, minimal)
        for obs, score in zip(training, best, strict=True):
            if score["silhouette_iou"] >= 0.35:
                continue
            camera, (depth, valid) = by_camera[obs["camera"]]
            part = render(path, camera, return_depth=True, only_groups={"primitive_0"})[
                2
            ]
            diagnostic = visibility_diagnostic(
                scene_path,
                camera,
                part,
                scenes[obs["camera"]],
                obs["mask"],
                depth,
                valid,
            )
            diagnostics.append({"camera": obs["camera"], **diagnostic})
    return current, {
        "status": "fitted",
        "training_cameras": [o["camera"] for o in training],
        "heldout_cameras": [o["camera"] for o in heldout],
        "initializations": [
            {"shape": obj["shape"], "scores": scores,
             "initialization": obj.get("initialization"),
             "attachment_prior": obj.get("attachment_prior")}
            for obj, scores in zip(starts, starting_scores, strict=True)
        ],
        "chosen_initialization": chosen,
        "training_objective": "mean_and_worst_view" if recipe.get("balanced_view_fit", False) else "mean",
        **({'tread_direction_prior': {
            'run_direction_xz': None if tread_direction is None else tread_direction.tolist(),
            'source_cameras': [o['camera'] for o in training],
            'excluded_cameras': [o['camera'] for o in heldout],
            'weight': .25,
            'qualification': 'Coherent horizontal image-line hypothesis; ascent sign and line identity unverified',
        }} if recipe.get('constrain_tread_direction', False) else {}),
        "optimized_initialization": selected_start,
        **({'creative_layout_prior': {'axes_xz': creative_axes, 'maximum_width': 4000,
                                     'qualification': 'Explicit creative design, not original tread evidence'}}
           if creative_axes is not None else {}),
        "floor_contact_prior": base_y if current["kind"] == "staircase" else None,
        "before": initial,
        "after": best,
        "heldout": evaluate(current, heldout, training_loss=False),
        "trials": trials,
        "visibility_diagnostics": diagnostics,
        "regularization": regularization,
        "attachment_prior": current.get("attachment_prior"),
        **({"scene_depth_fit": {
            "training_scene_cameras": scene_fit_cameras,
            "excluded_heldout_cameras": [o["camera"] for o in heldout],
            "tolerance": .005, "weight": 20,
            "qualification": "Depth-model penalty; final all-camera acceptance remains independent",
        }} if depth_fit else {}),
        **({"structure_search": {
            "optimized_initializations": selected,
            "optimized_structures": [starts[i]["shape"]["parameters"].get("structure", "solid")
                                     for i in selected],
            "selected_structure": current["shape"]["parameters"].get("structure", "solid"),
            "limit_per_structure": 4,
        }} if recipe.get("suspended_stair_starts", False) and current["kind"] == "staircase" else {}),
        "qualification": "Mask/depth agreement; not ground-truth geometry or 80% fidelity",
    }


def single_view_fixture_evidence(obj, group, scores, attached):
    """Conservative creative placement evidence; never proves recovered shape."""
    return bool(
        obj.get("kind") == "lamp"
        and len(group) == 1 and len(scores) == 1
        and obj.get("attachment_prior", {}).get("group", "").startswith("boundary_")
        and obj.get("interpretation", {}).get("source") == "fixture_attachment_prior"
        and attached
        and scores[0].get("silhouette_iou", 0) >= 0.6
        and scores[0].get("visible_pixels", 0) >= 16
    )


def fit_objects(room, recipe, segments, depths, out, generate_mesh, images=None):
    """Evaluate reversible replacements and retain every rejected trial report."""
    import html
    import json

    from room3d_refine import depth_error

    if (recipe.get("fixture_part_fit", False) or recipe.get('creative_fixture_part_fit', False)) and images is None:
        raise ValueError("fixture_part_fit requires original camera images")
    if recipe.get("stair_part_reviews") and images is None:
        raise ValueError("stair_part_reviews requires original camera images")
    out.mkdir(parents=True, exist_ok=True)
    current = copy.deepcopy(recipe)
    part_reviews = []
    if recipe.get("stair_part_reviews"):
        from pathlib import Path

        from room3d_part_review import annotate_segments

        segments, part_reviews = annotate_segments(
            room, images, segments, Path(recipe["stair_part_reviews"]),
        )
    semantic_rejections = []
    if recipe.get("semantic_stair_guard", False):
        from room3d_semantics import filter_stair_conflicts

        segments, semantic_rejections = filter_stair_conflicts(room, segments)
    groups = associate(observations(room, segments, depths,
                                   creative_stair_replacement=recipe.get('creative_stair_replacement', False)))
    reports = []
    rows = [
        '<meta charset="utf-8"><title>Parametric details</title>',
        "<style>body{background:#171a20;color:#eee;font:16px sans-serif}img{width:320px}a{color:#8bd}</style>",
        "<h1>Parametric detail trials</h1><p>Original / previous geometry / candidate geometry. Untextured views isolate shape. Rejected candidates are retained for inspection. Scores are mask/depth agreement, not fidelity.</p>",
    ]
    cameras = {c["index"]: (c, d) for c, d in zip(room["cameras"], depths, strict=True)}
    for number, group in enumerate(groups):
        previous_recipe = current
        dest = out / f"object{number}"
        dest.mkdir(exist_ok=True)
        kind = group[0]["kind"]
        # Only inferred objects may be replaced. Script-owned props are excluded.
        matches = []
        for i, obj in enumerate(current["primitives"]):
            e = obj.get("evidence", {})
            if obj["kind"] != kind or "camera" not in e:
                continue
            if any(
                e["camera"] == obs["camera"]
                and (
                    e.get("mask") in obs.get("mask_names", [obs["mask_name"]])
                    or (kind == "staircase" and e.get("label") == obs["label"])
                )
                for obs in group
            ):
                matches.append(i)
        if not matches and kind == "staircase":
            spatial = []
            for i, old in enumerate(current["primitives"]):
                if (
                    old["kind"] != kind
                    or "camera" not in old.get("evidence", {})
                    or "box" not in old
                ):
                    continue
                b = old["box"]
                lo = np.array([b[0], b[4], b[1]])
                hi = np.array([b[2], b[5], b[3]])
                for obs in group:
                    intersection = np.maximum(
                        0, np.minimum(hi, obs["hi"]) - np.maximum(lo, obs["lo"])
                    )
                    fraction = np.prod(intersection) / max(
                        1, min(np.prod(hi - lo), np.prod(obs["hi"] - obs["lo"]))
                    )
                    if fraction >= 0.25:
                        spatial.append(i)
                        break
            if len(spatial) == 1:
                matches = spatial
        base_path = dest / "before.obj"
        generate_mesh(room, base_path, current)
        stripped = copy.deepcopy(current)
        kept = [i for i in range(len(current["primitives"])) if i not in matches]
        remap = {f"primitive_{old}": f"primitive_{new}" for new, old in enumerate(kept)}
        stripped["primitives"] = [stripped["primitives"][i] for i in kept]
        stripped["hide"] = [
            [c, remap.get(g, g)]
            for c, g in stripped.get("hide", [])
            if not g.startswith("primitive_") or g in remap
        ]
        depth_reference_path, depth_reference_recipe = base_path, previous_recipe
        proxy_reference = (recipe.get("fixture_proxy_depth_reference", False)
                           and kind == "lamp" and bool(matches) and len(group) >= 2)
        if proxy_reference:
            depth_reference_path = dest / "proxy-free-depth-reference.obj"
            depth_reference_recipe = copy.deepcopy(stripped)
            generate_mesh(room, depth_reference_path, depth_reference_recipe)
        if kind == "staircase" and recipe.get("joint_stair_supports", False):
            from room3d_supports import fit as fit_supports

            obj, report, stripped = fit_supports(
                room, stripped, group, depths, dest, generate_mesh
            )
        elif kind == "lamp" and recipe.get("creative_fallback", False):
            from room3d_fixtures import fit_fixture_supports

            obj, report, stripped = fit_fixture_supports(
                room, stripped, group, depths, dest, generate_mesh, images=images
            )
        else:
            obj, report = fit_group(room, stripped, group, depths, dest, generate_mesh, images=images)
        report.update(
            kind=kind,
            observations=[
                {"camera": o["camera"], "mask": o["mask_name"], "label": o["label"],
                 **({'confidence_evidence': copy.deepcopy(o['confidence_evidence'])}
                    if 'confidence_evidence' in o else {}),
                 **({'approximate_stair_mask': True, 'predicted_iou': o['predicted_iou'],
                     'qualification': 'Approximate design cue; confidence unchanged, tread identity unverified'}
                    if o.get('approximate_stair_mask', False) else {})}
                for o in group
            ],
            replaced_indices=matches,
            accepted=False,
        )
        if proxy_reference:
            report["depth_reference"] = {
                "kind": "background_without_matched_inferred_fixture_proxies",
                "removed_proxy_indices": matches,
                "qualification": "Native, script-owned and other geometry retained; no model pixels excluded or tolerance changed. Original proxy-scene errors retained separately.",
            }
        if obj is not None:
            if kind == "staircase" and recipe.get("scene_depth_polish", False):
                polish_recipe = {**stripped, "scene_depth_fit": True,
                                 "scene_depth_polish": False, "balanced_view_fit": True,
                                 "constrain_stair_headroom": True}
                polished, polish_report = fit_group(
                    room, polish_recipe, group, depths, dest / "depth-polish",
                    generate_mesh, support=report.get("support"),
                    initial_candidates=[obj],
                )
                report = {**report, **polish_report, "depth_polish": {
                    "initial_shape": obj["shape"], "initial_report": report,
                    "qualification": "Mask/support/enclosure fit followed by whole-scene depth refinement; final gates unchanged",
                }}
                obj = polished
            if kind == "lamp" and recipe.get("fixture_part_fit", False):
                from room3d_fixture_parts import refine_wall_parts

                obj, part_report = refine_wall_parts(
                    room, stripped, obj, group, depths, images,
                    dest / "parts", generate_mesh,
                )
                report["part_refinement"] = part_report
                if part_report.get("status") == "selected":
                    report["after"] = part_report["after"]
            candidate = copy.deepcopy(stripped)
            obj["evidence"] = {
                "camera": group[0]["camera"],
                "mask": group[0]["mask_name"],
                "shared_observations": report["observations"],
            }
            candidate["primitives"].append(obj)
            after_path = dest / "after.obj"
            generate_mesh(room, after_path, candidate)
            if kind == "staircase":
                from room3d_connections import stair_headroom, support_mesh

                triangles, surface_groups = support_mesh(after_path)
                overhead = {
                    f"primitive_{i}"
                    for i, primitive in enumerate(candidate["primitives"])
                    if primitive.get("kind") == "ceiling"
                }
                report["headroom"] = stair_headroom(
                    obj, triangles, surface_groups, overhead
                )
            before_scores, after_scores = [], []
            for obs in group:
                camera, (target, valid) = cameras[obs["camera"]]
                old_scene = render(base_path, camera, return_depth=True)[2]
                old_part = render(
                    base_path,
                    camera,
                    return_depth=True,
                    only_groups={f"primitive_{i}" for i in matches},
                )[2]
                new_scene = render(after_path, camera, return_depth=True)[2]
                new_part = render(
                    after_path,
                    camera,
                    return_depth=True,
                    only_groups={f"primitive_{len(candidate['primitives']) - 1}"},
                )[2]
                before_scores.append(
                    compare(old_part, old_scene, obs["mask"], target, valid,
                            model_occlusion=not bool(report.get('creative_layout_prior')))
                )
                after_scores.append(
                    compare(new_part, new_scene, obs["mask"], target, valid,
                            model_occlusion=not bool(report.get('creative_layout_prior')))
                )
            depth_before, depth_after, original_depth_before = [], [], []
            for camera, (target, valid) in zip(room["cameras"], depths, strict=True):
                before_image, _, old_depth = render(
                    base_path, camera, return_depth=True
                )
                after_image, _, new_depth = render(
                    after_path, camera, return_depth=True
                )
                before_image.save(dest / f"camera{camera['index']}-before.png")
                after_image.save(dest / f"camera{camera['index']}-after.png")
                original_depth_before.append(depth_error(old_depth, target, valid))
                if proxy_reference:
                    old_depth = render(depth_reference_path, camera, return_depth=True)[2]
                depth_before.append(depth_error(old_depth, target, valid))
                depth_after.append(depth_error(new_depth, target, valid))
            if proxy_reference:
                report["original_proxy_scene_depth_before"] = original_depth_before
            train_count = len(report["training_cameras"])
            improving = (
                np.mean([s["loss"] for s in after_scores[:train_count]])
                < np.mean([s["loss"] for s in before_scores[:train_count]]) - 0.02
            )
            consistent = all(
                a["loss"] <= b["loss"] + 0.03
                for a, b in zip(after_scores, before_scores, strict=True)
            )
            depth_safe = all(
                a <= b + 0.005 for a, b in zip(depth_after, depth_before, strict=True)
            )
            supported = min(s["silhouette_iou"] for s in after_scores) >= (
                0.2 if matches and not any(o['label'].strip().casefold() == 'ceiling fan' for o in group) else 0.35
            )
            # A single-view fixture cannot establish shared placement. Stairs
            # may additionally use a repeated-tread structural prior, explicitly
            # reported as estimated and still subject to all-camera depth gates.
            structural_prior = kind == "staircase" and all(
                (o.get("structural_refinement") or {})
                .get("structural_evidence", {})
                .get("tread_edges", 0)
                >= 5
                and (o.get("structural_refinement") or {})
                .get("structural_evidence", {})
                .get("correlation", 0)
                >= 0.9
                for o in group
            )
            attached = all(s.get("attachment_valid", True) for s in report["after"])
            if obj['shape']['parameters'].get('profile') == 'ceiling_fan':
                from room3d_connections import lamp_attachment, support_mesh
                from room3d_fixtures import fixture_contact

                prior = obj.get('attachment_prior', {})
                triangles, surface_groups = support_mesh(after_path)
                ceilings = {f'primitive_{i}' for i, p in enumerate(candidate['primitives'])
                            if p.get('kind') == 'ceiling'}
                contact = (prior.get('support') == 'ceiling' and prior.get('group') in ceilings
                           and fixture_contact(obj, triangles, surface_groups))
                report['ceiling_attachment'] = {
                    **lamp_attachment(obj, triangles, surface_groups, ceilings),
                    'stem_cap_contact': bool(contact),
                    'qualification': 'Finite stem-cap contact with inferred ceiling; mechanical support unverified',
                }
                attached &= bool(contact)
            if obj['shape']['parameters'].get('profile') in ('ceiling_linear', 'ceiling_panel'):
                from room3d_connections import support_mesh
                from room3d_fixtures import fixture_contact

                ceilings = {f'primitive_{i}' for i, p in enumerate(candidate['primitives'])
                            if p.get('kind') == 'ceiling'}
                ceilings |= {s['group'] for s in candidate.get('structural_surfaces', [])
                             if s['material'] == 'ceiling'}
                prior = obj.get('attachment_prior', {})
                contact = (prior.get('group') in ceilings
                           and fixture_contact(obj, *support_mesh(after_path)))
                report['ceiling_attachment'] = {'mount_cap_contact': bool(contact),
                    'qualification': 'Finite estimated ceiling/mount contact; hidden suspension and mechanical support unverified'}
                attached &= bool(contact)
            fixture_prior = (
                recipe.get("creative_fallback", False)
                and recipe.get("single_view_wall_fixtures", False)
                and single_view_fixture_evidence(
                    obj, group, after_scores,
                    bool(report["after"]) and all(
                        s.get("attachment_valid") is True for s in report["after"]
                    ),
                )
            )
            supported &= bool(matches) or len(group) >= 2 or structural_prior or fixture_prior
            plausible = True
            if kind == "staircase" and recipe.get("creative_fallback", False):
                from room3d_assemblies import stair_proportions

                report["proportions"] = stair_proportions(obj)
                plausible = report["proportions"]["plausible"]
            check_headroom = kind == "staircase" and (
                recipe.get("scene_depth_polish", False) or recipe.get("constrain_stair_headroom", False)
            )
            headroom_safe = not check_headroom or report["headroom"]["obstructed_samples"] == 0
            layout_prior = report.get('creative_layout_prior')
            layout_safe = True
            if layout_prior:
                from room3d_creative_stairs import replacement_valid

                layout_safe = replacement_valid(obj, layout_prior['axes_xz'])
            keep = bool(improving and consistent and depth_safe and supported and plausible
                        and attached and headroom_safe and layout_safe)
            report.update(
                accepted=keep,
                baseline=before_scores,
                candidate=after_scores,
                depth_before=depth_before,
                depth_after=depth_after,
                gates={
                    "improving": bool(improving),
                    "consistent": bool(consistent),
                    "depth_safe": bool(depth_safe),
                    "supported": bool(supported),
                    "plausible_proportions": bool(plausible),
                    "fixture_attachment": bool(attached),
                    **({"headroom_safe": bool(headroom_safe)} if check_headroom else {}),
                    **({'creative_layout_safe': layout_safe} if layout_prior else {}),
                },
                confidence={
                    "observing_cameras": len(group),
                    "single_view_structural_prior": bool(
                        structural_prior and len(group) == 1
                    ),
                    "single_view_fixture_prior": bool(fixture_prior),
                    "heldout_cameras": len(report["heldout_cameras"]),
                    "minimum_mask_iou": min(s["silhouette_iou"] for s in after_scores),
                    "meaning": "Evidence strength, not calibrated fidelity probability",
                },
            )
            report["path"] = "evidence_fit"
            if keep:
                current = candidate
                if obj.get("interpretation", {}).get("source") in (
                    "regular_step_prior", "wall_attachment_prior", "fixture_attachment_prior",
                    "approximate_mask_layout_replacement",
                ):
                    report.update(path="procedural_interpretation", evidence_fit_accepted=False)
            elif recipe.get("creative_fallback", False) and (
                bool(matches) or len(group) >= 2 or structural_prior
            ):
                from room3d_interpretation import interpret

                creative_obj, creative_recipe, interpretation = interpret(
                    room,
                    current,
                    stripped,
                    group,
                    depths,
                    obj,
                    report.get("support"),
                    dest / "interpretation",
                    generate_mesh,
                )
                report["interpretation"] = interpretation
                if (interpretation['accepted'] and layout_prior
                        and not replacement_valid(creative_obj, layout_prior['axes_xz'])):
                    interpretation.update(accepted=False, reason='creative_layout_prior_failed')
                if interpretation["accepted"]:
                    if interpretation.get("landing_assembly", {}).get("accepted"):
                        report["landing_assembly"] = interpretation["landing_assembly"]
                    # Retain the rejected reconstruction and its diagnostics.
                    (dest / "evidence-fit.json").write_text(
                        json.dumps({"object": obj, "report": report}, indent=2) + "\n"
                    )
                    after_path.rename(dest / "evidence-fit.obj")
                    for camera in room["cameras"]:
                        i = camera["index"]
                        (dest / f"camera{i}-after.png").rename(
                            dest / f"camera{i}-evidence-fit.png"
                        )
                    obj, current = creative_obj, creative_recipe
                    generate_mesh(room, after_path, current)
                    for camera in room["cameras"]:
                        render(after_path, camera)[0].save(
                            dest / f"camera{camera['index']}-after.png"
                        )
                    report.update(accepted=True, path="procedural_interpretation")
                    report["evidence_fit_accepted"] = False
        if (kind == "staircase" and report["accepted"]
                and recipe.get("creative_fallback", False)
                and recipe.get("creative_stair_landings", False)
                and not report.get("landing_assembly", {}).get("accepted", False)):
            from room3d_landing_fit import fit_landing

            stair_index = len(current["primitives"]) - 1
            assembly, assembly_report = fit_landing(
                room, current, stair_index, depths, dest / "landing-assembly",
                generate_mesh, heldout_cameras=report["heldout_cameras"],
                depth_reference=base_path,
            )
            if assembly_report["accepted"]:
                assembled_path = dest / "landing-assembly" / "selected.obj"
                generate_mesh(room, assembled_path, assembly)
                assembly_scores, assembly_before_scores = [], []
                for obs in group:
                    camera, (target, valid) = cameras[obs["camera"]]
                    old_scene = render(after_path, camera, return_depth=True)[2]
                    old_part = render(after_path, camera, return_depth=True,
                                      only_groups={f"primitive_{stair_index}"})[2]
                    assembly_before_scores.append(compare(old_part, old_scene, obs["mask"], target, valid))
                    scene = render(assembled_path, camera, return_depth=True)[2]
                    part = render(assembled_path, camera, return_depth=True,
                                  only_groups={f"primitive_{stair_index}"})[2]
                    assembly_scores.append(compare(part, scene, obs["mask"], target, valid))
                unchanged = all(a["loss"] <= b["loss"] + .03 for a, b in
                                zip(assembly_scores, assembly_before_scores, strict=True))
                if report["path"] == "evidence_fit":
                    # Do not accumulate a second tolerance over the first fit.
                    # Keep the original before-object acceptance conditions.
                    unchanged &= all(a["loss"] <= b["loss"] + .03 for a, b in
                                     zip(assembly_scores, before_scores, strict=True))
                    unchanged &= np.mean([s["loss"] for s in assembly_scores[:train_count]]) < (
                        np.mean([s["loss"] for s in before_scores[:train_count]]) - .02)
                    unchanged &= min(s["silhouette_iou"] for s in assembly_scores) >= (.2 if matches else .35)
                assembly_report["object_scores_before"] = assembly_before_scores
                assembly_report["object_scores"] = assembly_scores
                assembly_report["object_fit_preserved"] = bool(unchanged)
                assembly_report["accepted"] = bool(unchanged)
                if unchanged:
                    current = assembly
                    report["candidate"] = assembly_scores
                    report["headroom"] = assembly_report["trials"][assembly_report["selected"]]["headroom"]
                    report["confidence"]["minimum_mask_iou"] = min(s["silhouette_iou"] for s in assembly_scores)
                    generate_mesh(room, after_path, current)
                    report["depth_after"] = []
                    for camera, (target, valid) in zip(room["cameras"], depths, strict=True):
                        image, _, depth = render(after_path, camera, return_depth=True)
                        image.save(dest / f"camera{camera['index']}-after.png")
                        report["depth_after"].append(depth_error(depth, target, valid))
                else:
                    assembly_report["reason"] = "object_fit_regressed"
            report["landing_assembly"] = assembly_report
        if (kind == 'lamp' and report['accepted'] and report.get('path') == 'procedural_interpretation'
                and recipe.get('creative_fixture_part_fit', False)):
            from room3d_creative_fixture_parts import refine as refine_creative_parts

            refined, refined_obj, part_report = refine_creative_parts(
                room, current, len(current['primitives'])-1, group, depths, images, segments,
                dest/'creative-parts', generate_mesh,
            )
            report['creative_part_refinement'] = part_report
            if part_report['accepted']:
                current, obj = refined, refined_obj
                generate_mesh(room, after_path, current)
                for camera in room['cameras']:
                    render(after_path, camera)[0].save(dest/f"camera{camera['index']}-after.png")
        if report["accepted"] and recipe.get("floor_depth_guard", False):
            from room3d_depth_guard import check_selected_scene

            guard_report = check_selected_scene(
                room, depth_reference_recipe, current, segments, depths,
                dest / "floor-depth-guard", generate_mesh,
                heldout_cameras=report["heldout_cameras"],
            )
            report["floor_depth_guard"] = guard_report
            report.setdefault("gates", {})["floor_depth_safe"] = guard_report["accepted"]
            if not guard_report["accepted"]:
                current = previous_recipe
                report.update(accepted=False, rejection_reason="floor_depth_guard_failed")
        (dest / "result.json").write_text(
            json.dumps({"object": obj, "report": report}, indent=2) + "\n"
        )
        reports.append(report)
        rows.append(
            f'<h2>Object {number}: {html.escape(kind)} ({html.escape(report.get("path", "no_fit"))}) — {"accepted" if report["accepted"] else "rejected"}</h2><p><a href="object{number}/result.json">Parameters, observations and gates</a></p>'
        )
        if obj is not None:
            for camera in room["cameras"]:
                i = camera["index"]
                rows.append(
                    f'<p>Camera {i}</p><img src="../camera{i}-original.png"><img src="object{number}/camera{i}-before.png"><img src="object{number}/camera{i}-after.png">'
                )
    result = {
        "objects": reports,
        "accepted": sum(r["accepted"] for r in reports),
        "interpreted": sum(
            r.get("path") == "procedural_interpretation" and r["accepted"]
            for r in reports
        ),
        "reconstructed": sum(
            r.get("path") == "evidence_fit" and r["accepted"] for r in reports
        ),
        "qualification": "Approximate priors; hidden parts inferred; acceptance is not ground-truth fidelity",
    }
    if recipe.get("semantic_stair_guard", False):
        result["semantic_rejections"] = semantic_rejections
        for rejection in semantic_rejections:
            rows.append('<p>Category abstention: ' + html.escape(
                f"camera {rejection['camera']}, {rejection['mask']}: "
                f"{rejection['reason']} ({rejection['contained_fraction']:.3f})"
            ) + '</p>')
    if recipe.get("stair_part_reviews"):
        result["part_reviews"] = part_reviews
        for review in part_reviews:
            rows.append('<p>Unverified part proposal: ' + html.escape(
                f"camera {review['camera']}, {review['mask']}: {review['category']}"
            ) + '</p>')
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "index.html").write_text("\n".join(rows))
    return current, result
