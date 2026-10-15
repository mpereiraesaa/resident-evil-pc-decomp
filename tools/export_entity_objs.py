"""Batch-export USA EMD viewport meshes and rig manifests through the game.

The viewer loads each EMD with the normal engine path, so this exports the
same vertex buffers and UVs that gameplay uses. A separate Wine prefix and
runtime executable keep this audit independent of other build work.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh", choices=("source", "refined"), default="source")
    parser.add_argument("--refinement", choices=("1", "2", "3"), default="3")
    options = parser.parse_args()

    audit = json.loads(options.audit.read_text())
    exe = options.exe.resolve()
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in audit:
        selector = item["index"]
        name = "".join(c if c.isalnum() else "_" for c in selector)
        env = os.environ.copy()
        env.update({
            "WINEPREFIX": str(options.wine_prefix.resolve()),
            "RE1_MODEL_VIEW": selector,
            "RE1_MODEL_VIEW_EXIT_FRAMES": "2",
            "RE1_ENTITY_OBJ_EXPORT_DIR": str(output),
            "RE1_ENTITY_OBJ_EXPORT_STAGE": options.mesh,
        })
        env.pop("RE1_ENTITY_MESH", None)
        env.pop("RE1_ENTITY_MESH_REPLACE_DIR", None)
        env.pop("RE1_ENTITY_MESH_PATCH_DIR", None)
        env.pop("RE1_HD_MODELS", None)
        env.pop("RE1_HD_MODELS_DIR", None)
        env.pop("RE1_HD_MODELS_PATCH_DIR", None)
        env.pop("RE1_CHRIS_MESH", None)
        if options.mesh == "refined":
            env["RE1_ENTITY_MESH"] = options.refinement
        started = time.monotonic()
        result = subprocess.run(
            ["wine", str(exe)], cwd=exe.parent, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", timeout=90,
        )
        log = output / f"{name}_{options.mesh}.log"
        log.write_text(result.stdout)
        manifest = output / f"{name}_rig.json"
        refs = []
        source_path = None
        if manifest.is_file():
            rig = json.loads(manifest.read_text())
            source_path = rig.get("sourcePath")
            refs = [joint["obj"] for joint in rig["joints"] if joint["obj"]]
        missing = [ref for ref in refs if not (output / ref).is_file()]
        faces = sum(sum(line.startswith("f ") for line in (output / ref).open())
                    for ref in refs if (output / ref).is_file())
        expected_faces = item.get("source_polygons") if options.mesh == "source" else None
        entry = {
            "path": item["path"], "actual_path": source_path,
            "selector": selector,
            "exit_code": result.returncode, "manifest": manifest.name if manifest.is_file() else None,
            "parts": len(refs), "missing": missing,
            "faces": faces, "expected_faces": expected_faces,
            "seconds": round(time.monotonic() - started, 2),
        }
        entries.append(entry)
        print(f"{selector:>5} {item['path']}: {entry['parts']} parts, "
              f"exit {result.returncode}, {entry['seconds']}s", flush=True)
        (output / "report.json").write_text(json.dumps(entries, indent=2) + "\n")
    successful = sum(entry["exit_code"] == 0 and not entry["missing"] and
                     entry["manifest"] is not None and
                     entry["actual_path"] == entry["path"] and
                     (entry["expected_faces"] is None or
                      entry["faces"] == entry["expected_faces"])
                     for entry in entries)
    print(f"Validated {successful}/{len(entries)} rig manifests", flush=True)
    if successful != len(entries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
