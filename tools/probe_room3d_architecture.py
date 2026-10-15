"""Frozen local architecture pilot on completed rooms; never edits the batch."""

import argparse
import html
import json
import logging
import os
import re
import shutil
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from batch_room3d import generate_mesh
from PIL import Image
from room3d_architecture import (
    column_axis_hypotheses,
    fit_column_axis,
    projected_object_evidence,
)
from room3d_checkpoint import digest, restore
from room3d_preview import render
from room3d_refine import depth_error, load_depth
from room3d_segment import load_masks
from room3d_segment import run as segment_images
from room3d_vision import run as detect_images

LABELS = ["stone column", "pillar", "archway", "balcony", "wall", "staircase", "railing"]
logger = logging.getLogger(__name__)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def probe_room(batch, name, out):
    source = batch / name
    checkpoint_path = batch / "checkpoints" / f"{name}.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    result = restore(checkpoint_path, checkpoint["key"], source)
    if result is None or result.get("status") != "image_assisted":
        raise ValueError("Expected an intact, completed assisted export")
    room = json.loads((source / "survey.json").read_text())
    recipe = json.loads((source / "before-details-recipe.json").read_text())
    floor = recipe.get("floor_y", 0)
    out.mkdir()
    paths = []
    for camera in room["cameras"]:
        path = out / f"camera{camera['index']}-original.png"
        shutil.copy2(source / path.name, path)
        paths.append(path)
    detect_images(paths, out / "detections", LABELS)
    segment_images(paths, out / "detections", out / "segmentation")
    images = [Image.open(path).convert("RGB") for path in paths]
    segments = load_masks(out / "segmentation", room, images)
    axes = column_axis_hypotheses(room, segments, floor)
    write(out / "axes.json", axes)
    eligible = [t for t in axes["trials"] if t["eligible"] and not t["ambiguous"]]
    candidates = []
    if eligible:
        depths = load_depth(batch / "depth" / name, source / "survey.json", room, images)
        generate_mesh(room, out / "scene.obj", recipe)
        scenes = [render(out / "scene.obj", c, return_depth=True)[2] for c in room["cameras"]]
        baseline = [depth_error(s, d, v) for s, (d, v) in zip(scenes, depths, strict=True)]
        for i, hypothesis in enumerate(eligible):
            dest = out / f"candidate{i}"
            obj, fit = fit_column_axis(room, hypothesis, segments, floor, dest, generate_mesh)
            write(dest / "object.json", obj)
            observed = {o["camera"]: o["mask"] for o in hypothesis["observations"]}
            checks = []
            for c, view, (depth, valid), scene, before, image in zip(
                room["cameras"], segments, depths, scenes, baseline, images, strict=True
            ):
                part = render(dest / "selected.obj", c, return_depth=True,
                              only_groups={"primitive_0"})[2]
                mask = next((m for d, m in view if d["mask"] == observed.get(c["index"])), None)
                evidence = projected_object_evidence(part, scene, depth, valid, mask)
                after = depth_error(np.minimum(scene, part), depth, valid)
                checks.append({"camera": c["index"], "used_for_fitting": c["index"] in observed,
                               "depth_delta": after - before, **evidence})
                # Diagnostic overlay respects reliable nearer observed depth.
                visible = np.isfinite(part) & (part <= scene + 50)
                visible &= ~(valid & (depth < part - np.maximum(180, depth * 0.12)))
                rgb = np.asarray(image).copy()
                rgb[visible] = (rgb[visible] * 0.5 + np.array([40, 240, 100]) * 0.5).astype(np.uint8)
                Image.fromarray(rgb).save(dest / f"camera{c['index']}-overlay.png")
            fit["view_evidence"] = checks
            write(dest / "fit.json", fit)
            selected = fit["trials"][fit["selected"]]
            candidates.append({"folder": dest.name, "training_scores": selected["scores"],
                               "max_depth_delta": max(c["depth_delta"] for c in checks),
                               "accepted_room_edit": False})
    columns = sum(any(w in d["label"] for w in ("column", "pillar"))
                  for view in segments for d, _ in view)
    summary = {"room": name, "cameras": len(room["cameras"]), "column_detections": columns,
               "axis_trials": len(axes["trials"]), "eligible_axis_pairs": len(eligible),
               "candidates": candidates, "source_checkpoint_key": checkpoint["key"],
               "accepted_room_edit": False, "status": "pilot_complete",
               "qualification": "Frozen diagnostic pilot; generation and positive training scores do not establish fidelity"}
    write(out / "summary.json", summary)
    rows = [f"<h1>{html.escape(name)} architecture pilot</h1>",
            "<p>Unverified local proposals. No room edits accepted. Candidate overlays suppress reliable nearer foreground; unknown depth remains unverified.</p>",
            '<a href="summary.json">Summary</a> · <a href="axes.json">Axis trials</a>']
    for camera in room["cameras"]:
        i = camera["index"]
        sheet = out / "segmentation" / f"camera{i}-original-segments.png"
        if sheet.exists():
            rows.append(f'<h2>Camera {i}</h2><img src="segmentation/{sheet.name}">')
        for candidate in candidates:
            folder = candidate["folder"]
            rows.append(f'<p>{folder}</p><img src="{folder}/camera{i}-overlay.png">')
    (out / "index.html").write_text(page("".join(rows)))
    if restore(checkpoint_path, checkpoint["key"], source) is None:
        raise ValueError("Source export changed during pilot")
    return summary


def page(body):
    return '<!doctype html><meta charset="utf-8"><title>Architecture pilot</title><style>body{background:#171a20;color:#eee;font:16px sans-serif}img{max-width:100%}a{color:#8bd}</style>' + body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.batch.resolve()) or args.out.exists():
        raise ValueError("Choose a new output directory outside the source batch")
    if any(re.fullmatch(r"room[0-9a-f]{4}", name) is None for name in args.rooms):
        raise ValueError("Expected lowercase room identifiers")
    if len(set(args.rooms)) != len(args.rooms):
        raise ValueError("Room identifiers must be unique")
    code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    args.out.mkdir(parents=True)
    manifest = {"labels": LABELS, "source_code": code, "results": [], "accepted_room_edits": False}
    write(args.out / "manifest.json", manifest)
    for name in args.rooms:
        try:
            result = probe_room(args.batch, name, args.out / name)
        except Exception as error:
            logger.exception("Architecture pilot failed for %s", name)
            result = {"room": name, "status": "error", "error": str(error)}
        manifest["results"].append(result)
        write(args.out / "manifest.json", manifest)
        print(name, result["status"], flush=True)
    manifest["source_code_unchanged"] = code == {
        p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    write(args.out / "manifest.json", manifest)
    links = "".join(f'<li><a href="{r["room"]}/index.html">{r["room"]}</a>: {r["status"]}</li>'
                    for r in manifest["results"])
    (args.out / "index.html").write_text(page('<h1>Frozen architecture pilot</h1><p>No room edits accepted.</p><ul>' + links + '</ul>'))
    return int(not manifest["source_code_unchanged"] or any(r["status"] == "error" for r in manifest["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
