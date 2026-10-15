"""Audit a completed local pipeline against current assets, renders and runtime.

This verifies evidence consistency and smoke-test scope, not visual fidelity.
Run with the tools venv from the repository root.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from batch_room3d import background
from room3d_depth import MODEL as DEPTH_MODEL, REVISION as DEPTH_REVISION
from room3d_preview import render
from room3d_refine import load_depth
from room3d_review import MODELS, PROMPT
from room3d_segment import MODEL as SEGMENT_MODEL, REVISION as SEGMENT_REVISION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_runtime_materials(mesh, runtime_folder, room):
    """The OBJ hash alone cannot prove that Wine used the reviewed appearance."""
    mods = runtime_folder / "mods" / "Stage1"
    sidecar = mesh.with_suffix(".cameras")
    if sidecar.exists():
        require(
            digest(sidecar) == digest(mods / f"{room.lower()}.cameras"),
            f"{room}: runtime cutaways differ",
        )
    for line in mesh.read_text().splitlines():
        if not line.startswith("mtllib "):
            continue
        name = line.split(maxsplit=1)[1]
        material = mesh.parent / name
        require(
            digest(material) == digest(mods / name), f"{room}: runtime material differs"
        )
        for entry in material.read_text().splitlines():
            if entry.startswith("map_Kd "):
                texture = entry.split(maxsplit=1)[1]
                require(
                    digest(mesh.parent / texture) == digest(mods / texture),
                    f"{room}: runtime texture differs: {texture}",
                )


def audit(root, executable, assets, rooms):
    manifest = read(root / "manifest.json")
    asset_files = {
        str(p.relative_to(assets)).lower(): p for p in assets.rglob("*") if p.is_file()
    }
    results = {item["room"].lower(): item for item in manifest["results"]}
    evidence = []
    for name in rooms:
        key = name.lower()
        folder = root / key
        result = read(folder / "result.json")
        require(result == results.get(key), f"{name}: manifest/result mismatch")
        require(result["status"] == "image_assisted", f"{name}: incomplete room")
        survey_path = folder / "survey.json"
        survey = read(survey_path)
        require(
            digest(asset_files[f"stage{int(name[4], 16)}/{survey['source']}".lower()])
            == survey["sha256"]
            == result["source_sha256"],
            f"{name}: original RDT changed",
        )
        mesh = folder / "assisted.obj"
        images = []
        backgrounds = {c["index"]: c for c in result["cameras"]}
        for camera in survey["cameras"]:
            stem = f"camera{camera['index']}"
            reference = folder / f"{stem}-original.png"
            image = Image.open(reference).convert("RGB")
            source_record = backgrounds[camera["index"]]
            source = asset_files[source_record["background"].lower()]
            require(
                digest(source) == source_record["background_sha256"],
                f"{name}/{stem}: original background changed",
            )
            require(
                background(source).tobytes() == image.tobytes(),
                f"{name}/{stem}: reference differs from original background",
            )
            images.append(image)
            segmentation = read(root / "segmentation" / key / f"{stem}-original.json")
            require(
                (segmentation["model"], segmentation["revision"])
                == (SEGMENT_MODEL, SEGMENT_REVISION),
                f"{name}/{stem}: SAM version mismatch",
            )
            require(
                segmentation["rgb_sha256"]
                == hashlib.sha256(image.tobytes()).hexdigest(),
                f"{name}/{stem}: segmentation image mismatch",
            )
            require(
                segmentation["detections_sha256"]
                == digest(root / "detections" / key / f"{stem}-original.json"),
                f"{name}/{stem}: detector input mismatch",
            )
            reviewed = folder / f"{stem}-assisted.png"
            current, _ = render(mesh, camera)
            require(
                current.tobytes() == Image.open(reviewed).convert("RGB").tobytes(),
                f"{name}/{stem}: reviewed render differs from current geometry/materials",
            )
            review = read(folder / f"{stem}-review.json")
            require(
                (review["model"], review["revision"]) == MODELS["4b"]
                and review["prompt"] == PROMPT,
                f"{name}/{stem}: review version mismatch",
            )
            require(
                [item["sha256"] for item in review["inputs"]]
                == [digest(reference), digest(reviewed)],
                f"{name}/{stem}: review input mismatch",
            )
            require(bool(review["response"].strip()), f"{name}/{stem}: empty review")
        depth_root = root / "depth" / key
        depth = read(depth_root / "depth.json")
        require(
            (depth["model"], depth["revision"]) == (DEPTH_MODEL, DEPTH_REVISION),
            f"{name}: depth version mismatch",
        )
        load_depth(depth_root, survey_path, survey, images)
        runtime = read(folder / result["runtime_report"])
        check_runtime_materials(mesh, (folder / result["runtime_report"]).parent, name)
        require(runtime == result["runtime"], f"{name}: runtime/result mismatch")
        require(
            runtime["passed"]
            and runtime["mesh_loaded"]
            and runtime["camera_transition"],
            f"{name}: runtime scenario failed",
        )
        if key == "room1180":
            require(
                runtime["menu_opened"] and runtime["menu_closed"],
                f"{name}: chest scenario failed",
            )
        require(
            runtime["mesh_sha256"] == digest(mesh),
            f"{name}: runtime used different mesh",
        )
        require(
            runtime["executable_sha256"] == digest(executable),
            f"{name}: executable changed",
        )
        correction = read(folder / "corrections" / "corrections.json")
        require(
            read(folder / "corrections" / "after.json")
            == read(folder / "assisted.json")["fitted_recipe"],
            f"{name}: accepted recipe mismatch",
        )
        views = read(folder / "new-views" / "views.json")
        require(
            views["status"] == "rendered" and views["views"],
            f"{name}: no new-view inspection",
        )
        for relative in (
            "index.html",
            "corrections/index.html",
            "new-views/index.html",
        ):
            require((folder / relative).is_file(), f"{name}: missing {relative}")
        evidence.append(
            {
                "room": name,
                "cameras_reviewed": len(images),
                "depth_before": correction["before"],
                "depth_after": correction["after"],
                "accepted_trials": sum(t["accepted"] for t in correction["trials"]),
                "new_view_coverage": [v["geometry_coverage"] for v in views["views"]],
                "runtime_scope": runtime["scope"],
            }
        )
    return {
        "passed": True,
        "scope": "Evidence consistency; not ground-truth fidelity or exhaustive gameplay",
        "rooms": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, required=True, help="Original USA asset directory"
    )
    parser.add_argument("--rooms", nargs="+", default=["ROOM1180", "ROOM10B0"])
    args = parser.parse_args()
    report = audit(args.output, args.executable, args.assets, args.rooms)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
