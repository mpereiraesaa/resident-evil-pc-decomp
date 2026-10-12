"""Local Qwen visual discrepancy proposals; never authoritative approval."""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from PIL import Image, ImageDraw

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT = (
    "This single image contains two panels: A on the LEFT and B on the RIGHT. "
    "Compare the two panels, which may be identical. First determine whether "
    "there is any clearly visible difference. If they are identical or you cannot "
    "reliably find a difference, respond exactly NO_CLEAR_DIFFERENCE. "
    "Otherwise report at most three directly observable differences, with their "
    "locations in each image. Do not assume either image has missing objects or "
    "defects. Do not infer unseen surfaces or gameplay. Never invent a difference "
    "to fill the requested count."
)


MODELS = {
    "2b": (MODEL, REVISION),
    "4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
}


def reuse_review(source, reference, reconstruction, out, size="4b"):
    """Reuse only the exact model/prompt and two exact image files reviewed."""
    if source is None or not source.is_file():
        return False
    record = json.loads(source.read_text())
    model, revision = MODELS[size]
    expected = [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (reference, reconstruction)
    ]
    if (
        record.get("model") != model
        or record.get("revision") != revision
        or record.get("prompt") != PROMPT
        or [item.get("sha256") for item in record.get("inputs", [])] != expected
    ):
        return False
    record["cache_source"] = str(source.resolve())
    record["inputs"] = [
        {"path": str(p), "sha256": sha}
        for p, sha in zip((reference, reconstruction), expected, strict=True)
    ]
    out.write_text(json.dumps(record, indent=2) + "\n")
    return True


def run(reference, reconstruction, out, size="2b"):
    model_id, revision = MODELS[size]
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, trust_remote_code=False
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .to("cpu")
        .eval()
    )
    images = [Image.open(p).convert("RGB") for p in (reference, reconstruction)]
    sheet = Image.new(
        "RGB",
        (
            sum(image.width for image in images),
            max(image.height for image in images) + 24,
        ),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)
    x = 0
    for label, image in zip(("A reference", "B comparison"), images, strict=True):
        sheet.paste(image, (x, 24))
        draw.text((x + 8, 5), label, fill="white")
        x += image.width
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out.with_suffix(".png"))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": sheet}]
            + [{"type": "text", "text": PROMPT}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    started = time.monotonic()
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    reply = processor.batch_decode(
        result[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    record = {
        "model": model_id,
        "revision": revision,
        "device": "cpu",
        "dtype": "bfloat16",
        "seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "prompt": PROMPT,
        "status": "unverified_review_proposal",
        "response": reply,
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (reference, reconstruction)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", choices=MODELS, default="2b")
    args = parser.parse_args()
    run(args.reference, args.reconstruction, args.out, args.size)
