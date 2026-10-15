#!/usr/bin/env python3
"""Run an isolated, local TRELLIS.cpp object probe from an existing SAM mask.

Requires Pillow and a separately installed trellis-cli + GGUF models. No network
calls, background removal service, game mutation or room alignment are performed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(image_path, mask_path, out, exposure=1.0):
    image = Image.open(image_path).convert("RGB")
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure multiplier must be positive and finite")
    lut = [min(255, round(value * exposure)) for value in range(256)]
    image = image.point(lut * 3).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size or not mask.getbbox():
        raise ValueError("Mask must be nonempty and match source dimensions")
    image.putalpha(mask)
    box = mask.getbbox()
    crop = image.crop(box)
    side = max(crop.size)
    padding = round(side * 0.15)
    canvas = Image.new("RGBA", (side + 2 * padding,) * 2)
    canvas.alpha_composite(
        crop, ((canvas.width - crop.width) // 2, (canvas.height - crop.height) // 2)
    )
    canvas.save(out / "input.png")
    return {
        "source": str(image_path),
        "mask": str(mask_path),
        "source_sha256": sha256(image_path),
        "mask_sha256": sha256(mask_path),
        "crop_bbox": box,
        "crop_size": crop.size,
        "exposure_multiplier": exposure,
        "preprocessing": "SAM alpha, bounding crop, 15% transparent padding",
        "input_sha256": sha256(out / "input.png"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "mask", "out", "runtime", "models"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exposure", type=float, default=1.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = prepare(args.image, args.mask, args.out, args.exposure)
    binary = args.runtime.resolve() / "trellis-cli"
    command = [
        str(binary),
        str((args.out / "input.png").resolve()),
        str((args.out / "object.glb").resolve()),
        "--models",
        str(args.models.resolve()),
        "--res",
        "512",
        "--steps",
        "12",
        "--seed",
        "42",
        "--atlas",
        "1024",
        "--webp",
        "off",
        "--threads",
        "4",
        "--require-gpu",
        "--verbose",
    ]
    evidence.update(
        command=command,
        runtime_sha256=sha256(binary),
        model_files={p.name: sha256(p) for p in sorted(args.models.glob("*.gguf"))},
        limitation="Single-image generative approximation; hidden surfaces inferred; no room placement or gameplay integration",
    )
    (args.out / "manifest.json").write_text(json.dumps(evidence, indent=2))
    env = os.environ.copy()
    # Native runtime supports host-backed chunk streaming in the large decoders.
    env["TRELLIS_CONVNEXT_STREAM_ALL"] = "1"
    evidence["decoder_streaming"] = True
    env["LD_LIBRARY_PATH"] = (
        str(args.runtime.resolve()) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    start = time.monotonic()
    samples = []
    with (args.out / "inference.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            try:
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                for line in sample.stdout.splitlines():
                    pid, memory = line.split(",")
                    if int(pid) == process.pid:
                        samples.append(
                            [round(time.monotonic() - start, 2), int(memory)]
                        )
            except (subprocess.SubprocessError, ValueError):
                pass
            time.sleep(1)
    evidence.update(
        exit_code=process.returncode,
        seconds=time.monotonic() - start,
        sampled_process_vram_peak_mib=max((s[1] for s in samples), default=None),
        vram_samples=samples,
    )
    artifact = args.out / "object.glb"
    if artifact.exists():
        evidence["output_sha256"] = sha256(artifact)
    (args.out / "manifest.json").write_text(json.dumps(evidence, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
