"""Report authored replacement coverage by distinct EMD and drawable part.

The 16x render-only subdivision covers every drawable model when enabled;
this report tracks separate topology/shape replacements so that coverage is
not confused with the generic subdivision pass.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--replacements", type=Path,
                        default=Path("tools/data/entity_mesh_replacements"))
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()

    source = json.loads(options.verification.read_text())
    known = {Path(entry["path"]).stem: entry for entry in source["entries"]}
    files = {}
    for path in sorted(options.replacements.glob("*.r1mesh")):
        stem = path.stem.rsplit("_", 1)[0]
        if stem not in known:
            raise ValueError(f"{path}: no verified EMD source")
        files.setdefault(stem, []).append(path.name)

    entries = []
    for entry in source["entries"]:
        stem = Path(entry["path"]).stem
        entries.append({"path": entry["path"], "parts": entry["parts"],
                        "replacement_files": files.get(stem, [])})
    drawable = sum(entry["parts"] > 0 for entry in entries)
    covered = sum(entry["parts"] > 0 and bool(entry["replacement_files"])
                  for entry in entries)
    drawable_parts = sum(entry["parts"] for entry in entries)
    replaced_parts = sum(len(entry["replacement_files"]) for entry in entries)
    report = {"models": len(entries), "drawable_models": drawable,
              "models_with_authored_replacements": covered,
              "drawable_models_remaining": drawable - covered,
              "drawable_parts": drawable_parts,
              "parts_with_authored_replacements": replaced_parts,
              "drawable_parts_remaining": drawable_parts - replaced_parts,
              "replacement_files": replaced_parts,
              "entries": entries}
    if options.out:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{covered}/{drawable} drawable EMDs and "
          f"{replaced_parts}/{drawable_parts} parts have authored replacements "
          f"({drawable - covered} models and {drawable_parts - replaced_parts} "
          "parts remain)")


if __name__ == "__main__":
    main()
