"""Build inspectable shoulder-sculpt candidates for the seven human EMDs.

Requires the verified 16x refined OBJ batch and Blender. Candidates go into
the output directory; this script does not install them into the game mod.
"""

import argparse
import json
from pathlib import Path
import subprocess


MODELS = {
    "0": "enemy/char10.emd", "1": "enemy/char11.emd",
    "2": "enemy/char12.emd", "3": "enemy/char13.emd",
    "38": "enemy/em1022.emd", "39": "enemy/em1023.emd",
    "40": "enemy/em1024.emd",
}


def run_blender(blender, script, blend, arguments, log):
    command = [str(blender), "--background"]
    if blend is not None:
        command.append(str(blend))
    command.extend(["--python", str(script), "--", *map(str, arguments)])
    with log.open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Blender failed ({result.returncode}): {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selectors", nargs="*", choices=tuple(MODELS),
                        default=list(MODELS))
    options = parser.parse_args()
    blender = options.blender.resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    scripts = Path(__file__).resolve().parent
    source = options.refined_dir.resolve()
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for selector in options.selectors:
        manifest = source / f"{selector}_rig.json"
        rig = json.loads(manifest.read_text())
        if rig["sourcePath"] != MODELS[selector]:
            raise ValueError(f"{selector}: expected {MODELS[selector]}, "
                             f"got {rig['sourcePath']}")
        model = Path(rig["sourcePath"]).stem
        folder = output / model
        folder.mkdir(parents=True, exist_ok=True)
        assembled = folder / "assembled.blend"
        sculpted = folder / "shoulders.blend"
        subdivided = folder / "shoulders-subdivided.blend"
        replacement = folder / "replacement"
        replacement.mkdir(exist_ok=True)
        run_blender(blender, scripts / "assemble_entity_obj.py", None,
                    [manifest, "--blend", assembled, "--render",
                     folder / "torso-before.png", "--focus-joint", "0"],
                    folder / "assemble.log")
        run_blender(blender, scripts / "sculpt_human_shoulders.py", assembled,
                    ["--blend", sculpted, "--render", folder / "torso-after.png"],
                    folder / "sculpt.log")
        run_blender(blender, scripts / "subdivide_entity_joint.py", sculpted,
                    ["--joint", "0", "--cuts", "1", "--blend", subdivided],
                    folder / "subdivide.log")
        run_blender(blender, scripts / "export_entity_mesh.py", subdivided,
                    ["--out", replacement, "--only-joint", "0",
                     "--preserve-source-normals"], folder / "export.log")
        files = list(replacement.glob("*.r1mesh"))
        if len(files) != 1:
            raise RuntimeError(f"{selector}: expected one candidate mesh")
        print(f"{selector} {rig['sourcePath']} -> {files[0]}", flush=True)


if __name__ == "__main__":
    main()
