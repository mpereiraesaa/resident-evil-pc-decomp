#!/usr/bin/env python3
"""Build every room reading, install the meshes into a mod tree, verify them.

For each room3d/readings/ROOMxxxx.toml: extract the corrected originals,
build the mesh through room3d_reading.py, score it against the originals,
copy it into <out>/mods/StageN/ with room-prefixed texture names, and
optionally load it in the Wine build through tools/check_room3d_runtime.py.
Writes <out>/report.md and <out>/report.json. Readings whose room has an
identical layout sibling (room3d_rooms.py catalog) are copied to the
siblings as well.

    room3d_readings_batch.py --assets assets --out build/room3d-readings-batch
        [--rooms ROOM1050 ...] [--runtime-exe build/windows-cross/residentevil.exe]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room3d_handmade import edge_agreement  # noqa: E402
from room3d_reading import build_reading, survey_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def install(built_dir, name, mods_root, stage):
    """Copy a built room into mods/StageN with textures prefixed by the room."""
    dst = mods_root / f"Stage{stage}"
    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob(f"{name}*"):
        stale.unlink()
    obj = (built_dir / f"{name}.obj").read_text()
    mtl_lines = []
    for line in (built_dir / f"{name}.mtl").read_text().splitlines():
        if line.startswith("map_Kd "):
            texture = line.split(maxsplit=1)[1]
            shutil.copy2(built_dir / texture, dst / f"{name}-{texture}")
            line = f"map_Kd {name}-{texture}"
        mtl_lines.append(line)
    (dst / f"{name}.mtl").write_text("\n".join(mtl_lines) + "\n")
    (dst / f"{name}.obj").write_text(obj)
    shutil.copy2(built_dir / f"{name}.cameras", dst / f"{name}.cameras")


def runtime_check(exe, assets, mesh, out, stage, room, checker):
    if out.exists():
        shutil.rmtree(out)
    command = [sys.executable, str(checker), "--exe", str(exe),
               "--assets", str(assets.parent if assets.name == "USA" else assets), "--mesh", str(mesh),
               "--out", str(out), "--stage", str(stage), "--room", f"0x{room:02X}"]
    proc = subprocess.run(command, capture_output=True, text=True)
    result_path = out / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    return {"passed": False, "error": (proc.stderr or proc.stdout)[-400:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="assets directory (parent of USA)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readings", type=Path, default=ROOT / "room3d" / "readings")
    parser.add_argument("--rooms", nargs="*", help="RDT stems to limit to, e.g. ROOM1050")
    parser.add_argument("--runtime-exe", type=Path, help="Wine build to load every room in")
    parser.add_argument("--checker", type=Path, default=ROOT / "tools" / "check_room3d_runtime.py",
                        help="check_room3d_runtime.py to use (lives in the root checkout)")
    parser.add_argument("--siblings", type=Path, help="room3d_rooms.py catalog JSON to copy identical layouts")
    parser.add_argument("--merge", action="store_true",
                        help="keep entries already in report.json for rooms not rebuilt this run")
    args = parser.parse_args()
    usa = args.assets / "USA" if (args.assets / "USA").is_dir() else args.assets
    args.out.mkdir(parents=True, exist_ok=True)
    readings = sorted(args.readings.glob("ROOM*.toml"))
    if args.rooms:
        wanted = {r.upper() for r in args.rooms}
        readings = [r for r in readings if r.stem.upper() in wanted]
    siblings = {}
    if args.siblings and args.siblings.exists():
        catalog = json.loads(args.siblings.read_text())
        for layout in catalog.get("layouts", []):
            rooms = layout.get("rooms", [])
            for room in rooms:
                siblings[room] = [r for r in rooms if r != room]
    report = []
    for reading in readings:
        stem = reading.stem.upper()
        stage, room = int(stem[4]), int(stem[5:7], 16)
        rdt = usa / f"Stage{stage}" / f"{stem}.RDT"
        started = time.monotonic()
        entry = {"room": stem, "stage": stage}
        try:
            evidence = args.out / "evidence" / stem.lower()
            if not (evidence / "camera0-original.png").exists():
                if stage in (6, 7):
                    # The return stages ship no backgrounds of their own; the
                    # game reuses stage 1/2's, and their cameras are identical.
                    twin = args.out / "evidence" / f"room{stage - 5}{stem[5:].lower()}"
                    if not (twin / "camera0-original.png").exists():
                        survey_report(usa / f"Stage{stage - 5}" / f"ROOM{stage - 5}{stem[5:]}.RDT", twin)
                    shutil.copytree(twin, evidence, dirs_exist_ok=True)
                else:
                    survey_report(rdt, evidence)
            built = args.out / "built" / stem.lower()
            mesh = build_reading(reading, rdt, built, evidence)
            cameras = len(list(evidence.glob("camera*-original.png")))
            scores = edge_agreement(built, stem.lower(), evidence, cameras)
            entry["cameras"] = cameras
            entry["edge"] = [(c, round(p, 2), round(r, 2)) for c, p, r in scores]
            weak = []
            backdrop = set(tomllib.loads(reading.read_text()).get("backdrop", []))
            entry["backdrop"] = sorted(backdrop)
            for index in range(cameras):
                if index in backdrop:  # keeps the original background in game
                    continue
                pixels = np.asarray(Image.open(built / f"preview-camera{index}.png").convert("L"), dtype=float)
                coverage, mean = float((pixels > 3).mean()), float(pixels.mean())
                if coverage < 0.6 or mean < 12:
                    weak.append((index, round(coverage, 2), round(mean, 1)))
            entry["weak_cameras"] = weak
            entry["bytes"] = mesh.stat().st_size
            install(built, stem.lower(), args.out / "mods", stage)
            for sibling in siblings.get(stem, []):
                install(built, sibling.lower(), args.out / "mods", int(sibling[4]))
                shutil.copy2(built / f"{stem.lower()}.cameras", args.out / "mods" / f"Stage{sibling[4]}" / f"{sibling.lower()}.cameras")
            entry["siblings"] = siblings.get(stem, [])
            if args.runtime_exe:
                runtime = runtime_check(args.runtime_exe, usa, mesh, args.out / "runtime" / stem.lower(), stage, room, args.checker)
                entry["runtime"] = {k: runtime.get(k) for k in ("passed", "mesh_loaded", "cameras_seen", "camera_transition", "error") if k in runtime}
                if runtime.get("process_exit_before_cleanup") == 3:
                    entry["runtime"]["error"] = "skipped: another game instance is running (single-instance mutex)"
            entry["status"] = "ok"
        except Exception as error:  # keep going; the report shows the failure
            entry["status"] = f"error: {error}"
        entry["seconds"] = round(time.monotonic() - started, 1)
        report.append(entry)
        print(f"{stem}: {entry['status']} ({entry['seconds']}s)", flush=True)
    if args.merge and (args.out / "report.json").exists():
        rebuilt = {e["room"] for e in report}
        kept = [e for e in json.loads((args.out / "report.json").read_text()) if e["room"] not in rebuilt]
        report = sorted(kept + report, key=lambda e: e["room"])
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    loaded = sum(1 for e in report if e.get("runtime", {}).get("mesh_loaded"))
    weak = sum(len(e.get("weak_cameras", [])) for e in report)
    lines = ["# Room readings batch", "",
             f"{len(report)} rooms; {sum(1 for e in report if e['status'] == 'ok')} built; "
             f"{loaded} loaded in the engine; {weak} cameras flagged as mostly empty or very dark.", "",
             "| Room | Cameras | Edge precision / recall per camera | Weak cameras | Runtime | Status |", "|---|---|---|---|---|---|"]
    for e in report:
        edge = " ".join(f"{c}:{p:.2f}/{r:.2f}" for c, p, r in e.get("edge", []))
        runtime = e.get("runtime", {})
        rt = "-" if not runtime else ("loaded" if runtime.get("mesh_loaded") else f"not loaded: {runtime.get('error', 'see runtime dir')[:70]}")
        wk = " ".join(f"{c}" for c, _, _ in e.get("weak_cameras", [])) or "-"
        lines.append(f"| {e['room']} | {e.get('cameras', '-')} | {edge} | {wk} | {rt} | {e['status']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
