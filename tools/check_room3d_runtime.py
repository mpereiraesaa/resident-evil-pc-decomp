"""Run an isolated Wine room smoke test using the game's scripted-input hook.

No desktop focus or global key events are used. A live process alone is not a
pass: mesh load, camera transition and menu open/close must appear in its log.
The reference scenario is ROOM118, Chris, starting a new game on USA assets.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

PATTERN = re.compile(
    r"\[room3d-test\] stage=(\d+) room=([0-9A-F]+) camera=(\d+) menu=(\d+) door=(\d+) standalone=(\d+)"
)


def observations(log, room, stage=1):
    events = [
        tuple(int(v, 16 if i == 1 else 10) for i, v in enumerate(m.groups()))
        for m in PATTERN.finditer(log)
    ]
    events = [e for e in events if e[0] == stage and e[1] == room]
    cameras = {e[2] for e in events if not any(e[3:])}
    menu_index = next((i for i, e in enumerate(events) if e[3]), None)
    closed = menu_index is not None and any(
        not any(e[3:]) for e in events[menu_index + 1 :]
    )
    return {
        "room_events": events,
        "camera_transition": len(cameras) >= 2,
        "menu_opened": menu_index is not None,
        "menu_closed": closed,
        "mesh_loaded": bool(
            re.search(rf"\[room3d\] loaded .*room{stage}{room:02x}0\.obj", log, re.IGNORECASE)
        ),
        "cameras_seen": len(cameras),
    }


def run(executable, assets, mesh, output, timeout=150, target_room=0x18, stage=1):
    """Load `mesh` for (stage, target_room) through the debug menu.

    The three original scenarios (stage 1 rooms 0x18, 0x0B, 0x06) also require
    a camera transition and, for the storeroom, the item-box menu; any other
    room passes when its mesh loads and the player is placed.
    """
    if not (assets / "USA" / f"Stage{stage}").is_dir():
        raise ValueError("Assets must name the parent containing USA/StageN")
    mesh_name = f"room{stage}{target_room:02x}0"
    scripted = stage == 1 and target_room in (0x18, 0x0B, 0x06)
    required = ("mesh_loaded",) + (("camera_transition",) if scripted else ()) + (
        ("menu_opened", "menu_closed") if (stage == 1 and target_room == 0x18) else ()
    )
    if output.exists():
        raise ValueError("Use a fresh output directory to prevent stale test evidence")
    output.mkdir(parents=True)
    shutil.copy2(executable, output / "residentevil.exe")
    mod = output / "mods" / f"Stage{stage}"
    mod.mkdir(parents=True)
    shutil.copy2(mesh, mod / (mesh_name + ".obj"))
    sidecar = mesh.with_suffix(".cameras")
    if sidecar.exists():
        shutil.copy2(sidecar, mod / (mesh_name + ".cameras"))
    # Copy only the declared local material/texture dependencies. Never copy
    # unrelated game data or follow absolute / parent-traversal references.
    for line in mesh.read_text().splitlines():
        if not line.startswith("mtllib "):
            continue
        name = line.split(maxsplit=1)[1]
        if Path(name).name != name or "\\" in name:
            raise ValueError("Unsafe material path")
        material = mesh.parent / name
        shutil.copy2(material, mod / name)
        for entry in material.read_text().splitlines():
            if not entry.startswith("map_Kd "):
                continue
            texture = entry.split(maxsplit=1)[1]
            if Path(texture).name != texture or "\\" in texture:
                raise ValueError("Unsafe texture path")
            shutil.copy2(mesh.parent / texture, mod / texture)
    asset_win = "Z:" + str(assets.resolve()).replace("/", "\\")
    (output / "config.ini").write_text(
        f"[Display]\nFullScreen=0\nWidth=640\nHeight=480\nBitDepth=32\nVSync=1\n[Assets]\nPath={asset_win}\nVersion=USA\nMods=mods\n[Save]\nPath=saves\n[Game]\nMode=OG\nSkipUnskippableFmv=1\nSkipDoorAnimations=1\n[Debug]\nEnableDebug=1\n"
    )
    keys = [
        (13, 1000, 150),
        (13, 5000, 150),
        (67, 8000, 150),
        (67, 11000, 150),
        (13, 14000, 150),
        (13, 18000, 150),
        (13, 22000, 150),
        (13, 26000, 150),
    ]
    command_id = 0

    def send_keys(sequence):
        nonlocal command_id
        command_id += 1
        if len(sequence) > 128:
            raise ValueError("Input sequence exceeds engine limit")
        text = (
            ",".join(f"{vk}:{start}-{duration}" for vk, start, duration in sequence)
            + f"\n#{command_id}\n"
        )
        temp = output / "test-keys.tmp"
        temp.write_text(text)
        temp.replace(output / "test-keys.txt")

    send_keys(keys)
    env = dict(
        os.environ,
        RE1_DEBUGLOG="1",
        RE1_TEST_BACKGROUND="1",
        RE1_ROOM3D_TRACE="1",
        WINEDEBUG="-all",
        RE1_TEST_KEYS_FILE="test-keys.txt",
    )
    env.pop("RE1_TEST_KEYS", None)
    valid_rooms = sorted(
        int(p.stem[5:7], 16)
        for p in (assets / "USA" / f"Stage{stage}").glob("ROOM*0.RDT")
        if p.stat().st_size > 4
    )
    phase = "boot"

    result = {}
    with (output / "wine.log").open("w") as stream:
        process = subprocess.Popen(
            ["wine", "./residentevil.exe"],
            cwd=output,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        start = time.monotonic()
        try:
            while time.monotonic() - start < timeout and process.poll() is None:
                logpath = output / "re1_debug.log"
                log = logpath.read_text(errors="replace") if logpath.exists() else ""
                if phase == "boot" and "[room3d-test] stage=1" in log:
                    # During the opening event, C advances dialogue. F1 is
                    # retried until the debug menu actually reports opening.
                    send_keys(
                        [
                            (key, 100 + i * 1000 + offset, 150)
                            for i in range(60)
                            for key, offset in [(112, 0), (67, 500)]
                        ]
                    )
                    phase = "await_debug_menu"
                opened = re.search(r"\[debugmenu-test\] opened room=([0-9A-F]+)", log)
                if phase == "await_debug_menu" and opened:
                    current = int(opened.group(1), 16)
                    # RIGHT moves to the next stage; the room cursor keeps its
                    # index, clamped and stepped up past unavailable rooms.
                    stage_presses = stage - 1
                    cursor = current
                    if stage_presses:
                        cursor = min(cursor, valid_rooms[-1])
                        cursor = next(r for r in valid_rooms if r >= cursor)
                    steps = valid_rooms.index(target_room) - valid_rooms.index(cursor)
                    key = 38 if steps >= 0 else 40
                    t = 100
                    sequence = [(13, t, 150)]
                    t += 600
                    for _ in range(stage_presses):
                        sequence.append((39, t, 90))
                        t += 250
                    for _ in range(abs(steps)):
                        sequence.append((key, t, 90))
                        t += 250
                    sequence.append((13, t + 300, 150))
                    send_keys(sequence)
                    phase = "await_mesh"
                observed = observations(log, target_room, stage)
                ready = (
                    observed["mesh_loaded"]
                    and observed["room_events"]
                    and not any(observed["room_events"][-1][3:])
                    and "[debugmenu] room change: placed at" in log
                )
                if phase == "await_mesh" and ready:
                    send_keys(
                        [(38, 1500, 3000), (67, 5500, 150), (86, 8000, 150)]
                        if (stage == 1 and target_room == 0x18)
                        else [(38, 1500, 18000)]
                    )
                    phase = "interaction"
                result = observations(log, target_room, stage)
                if all(result[k] for k in required) and (scripted or phase == "interaction" and time.monotonic() - start > 40):
                    break
                time.sleep(0.5)
            result["mesh_sha256"] = hashlib.sha256(
                (mod / (mesh_name + ".obj")).read_bytes()
            ).hexdigest()
            result["executable_sha256"] = hashlib.sha256(
                (output / "residentevil.exe").read_bytes()
            ).hexdigest()
            result["driver_phase"] = phase
            result["elapsed_seconds"] = round(time.monotonic() - start, 2)
            result["process_exit_before_cleanup"] = process.poll()
            result["passed"] = all(result.get(k, False) for k in required)
            result["scope"] = (
                f"{mesh_name.upper()} smoke test: {', '.join(required)}; no visual fidelity, item transfer or save assertions"
            )
            result["stage"] = stage
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--room", type=lambda v: int(v, 16), default=0x18)
    parser.add_argument("--stage", type=int, default=1, help="RDT stage digit (ROOM<stage>xx0)")
    args = parser.parse_args()
    report = run(args.exe, args.assets, args.mesh, args.out, target_room=args.room, stage=args.stage)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
