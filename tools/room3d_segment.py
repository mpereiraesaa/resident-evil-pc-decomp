"""Local SAM 2.1 masks prompted by hash-verified Grounding DINO proposals.

Masks are hypotheses, not ground truth. Downloads pinned weights only; image
inference is CPU-local. Output keeps each detection separate, including overlaps.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from PIL import Image, ImageDraw

MODEL = "facebook/sam2.1-hiera-tiny"
REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"


def verified_boxes(image, record):
    if record.get("rgb_sha256") != hashlib.sha256(image.tobytes()).hexdigest():
        raise ValueError("Detection image hash mismatch")
    boxes = []
    width, height = image.size
    for detection in record["detections"]:
        box = np.asarray(detection["box"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Invalid detection box")
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty detection box")
        boxes.append(box.tolist())
    return boxes


def load_masks(root, room, images):
    """Read verified, image-aligned proposals without loading a neural model."""
    result = []
    for camera, image in zip(room["cameras"], images, strict=True):
        record = json.loads(
            (root / f"camera{camera['index']}-original.json").read_text()
        )
        if record.get("rgb_sha256") != hashlib.sha256(image.tobytes()).hexdigest():
            raise ValueError("Segmentation image hash mismatch")
        entries = []
        for item in record["segments"]:
            name = item["mask"]
            if Path(name).name != name or "\\" in name:
                raise ValueError("Mask path must be a filename")
            mask = np.asarray(Image.open(root / name).convert("L")) > 0
            if mask.shape != (image.height, image.width):
                raise ValueError("Mask dimensions differ from source image")
            entries.append((item, mask))
        result.append(entries)
    return result


def matches_material(label, keywords):
    # "wall lamp" is a lamp, not a wall material. Substring matching here
    # previously painted white light-bulb pixels onto entire unseen walls.
    return any(
        (
            label.strip() == word
            if word == "wall"
            else word in label and not (word == "staircase" and "railing" in label)
        )
        for word in keywords
    )


def texture_support(entries, kind, xy, foreground_fixture_guard=False):
    """Reject confident incompatible object pixels on architectural surfaces.

    This is a semantic guard, not proof of geometry. Unknown pixels remain usable.
    Ambiguous labels and weak SAM masks cannot veto texture samples.
    """
    fixture_safe = np.ones(len(xy), dtype=bool)
    if foreground_fixture_guard:
        # Explicit synonyms only: this veto does not promote detections into
        # geometry or make a mixed furniture/fixture label a material source.
        fixture_labels = {
            "lamp": "lamp", "wall lamp": "lamp", "chandelier": "lamp",
            "ceiling lamp": "lamp", "table lamp": "lamp", "wall lamp chandelier": "lamp",
            "ceiling fan": "lamp",
            "railing": "railing", "staircase railing": "railing",
            "wooden staircase railing": "railing",
        }
        for item, mask in entries:
            category = fixture_labels.get(item["label"].strip().casefold())
            if (category is not None and category != kind
                    and np.isfinite(item["score"]) and np.isfinite(item["predicted_iou"])
                    and item["score"] >= .3 and item["predicted_iou"] >= .8):
                fixture_safe &= ~mask[xy[:, 1], xy[:, 0]]
    allowed = {
        "staircase": ("staircase",),
        "railing": ("railing",),
        "barrel": ("barrel",),
        "lamp": ("lamp", "chandelier", "ceiling fan"),
        "floor": ("floor",),
        "wall": ("wall", "door", "window", "painting"),
        "ceiling": ("ceiling",),
        "door_panel": ("door",),
        "chest_body": ("chest", "crate"),
        "typewriter_table": ("table", "typewriter"),
    }.get(kind)
    if allowed is None:
        return fixture_safe
    compatible = np.zeros(len(xy), dtype=bool)
    conflicting = np.zeros(len(xy), dtype=bool)
    hard_conflict = np.zeros(len(xy), dtype=bool)
    for item, mask in entries:
        if not np.isfinite(item["score"]) or not np.isfinite(item["predicted_iou"]):
            continue
        # Door geometry provides a category prior independent of the detector.
        # A well-localized door mask may resolve a competing crate label even
        # when its detector score is lower. It cannot override foreground fixtures.
        door_support = (kind == "door_panel" and item["label"].strip() == "door"
                        and item["score"] >= 0.3 and item["predicted_iou"] >= 0.9)
        if not door_support and (item["score"] < 0.4 or item["predicted_iou"] < 0.7):
            continue
        hit = mask[xy[:, 1], xy[:, 0]]
        # An overlapping wall/stair mask must not overrule a thin foreground
        # fixture: that would paint lamps and rails onto architecture again.
        fixture = any(
            word in item["label"] for word in ("lamp", "chandelier", "railing", "ceiling fan")
        )
        if fixture and kind in ("wall", "floor", "ceiling", "staircase", "door_panel"):
            hard_conflict |= hit
        elif matches_material(item["label"], allowed):
            compatible |= hit
        else:
            conflicting |= hit
    return (~conflicting | compatible) & ~hard_conflict & fixture_safe


class Segmenter:
    def __init__(self):
        import torch
        from transformers import Sam2Model, Sam2Processor

        torch.set_num_threads(4)
        self.torch = torch
        self.processor = Sam2Processor.from_pretrained(MODEL, revision=REVISION)
        self.model = (
            Sam2Model.from_pretrained(
                MODEL, revision=REVISION, use_safetensors=True, trust_remote_code=False
            )
            .to("cpu")
            .eval()
        )

    def predict(self, image, boxes):
        if not boxes:
            return np.empty((0, image.height, image.width), bool), []
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        masks = (
            self.processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0][:, 0]
            .numpy()
            .astype(bool)
        )
        scores = outputs.iou_scores[0, :, 0].tolist()
        return masks, scores


def run(images, detections, out):
    out.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter()
    for path in images:
        image = Image.open(path).convert("RGB")
        record_path = detections / (path.stem + ".json")
        record = json.loads(record_path.read_text())
        boxes = verified_boxes(image, record)
        started = time.monotonic()
        masks, scores = segmenter.predict(image, boxes)
        entries = []
        tiles = []
        for i, (mask, score, detection) in enumerate(
            zip(masks, scores, record["detections"], strict=True)
        ):
            name = f"{path.stem}-mask{i:02d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / name)
            entries.append(
                dict(
                    detection,
                    mask=name,
                    mask_pixels=int(mask.sum()),
                    predicted_iou=float(score),
                    status="unverified_mask_proposal",
                )
            )
            rgb = np.asarray(image).copy()
            rgb[~mask] //= 5
            tile = Image.fromarray(rgb)
            ImageDraw.Draw(tile).text(
                (3, 3),
                f"{i}: {detection['label']}",
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            tiles.append(tile)
        if tiles:
            sheet = Image.new(
                "RGB", (image.width * 3, image.height * ((len(tiles) + 2) // 3))
            )
            for i, tile in enumerate(tiles):
                sheet.paste(tile, (i % 3 * image.width, i // 3 * image.height))
            sheet.save(out / (path.stem + "-segments.png"))
        output = {
            "model": MODEL,
            "revision": REVISION,
            "device": "cpu",
            "image": path.name,
            "rgb_sha256": record["rgb_sha256"],
            "detections_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "seconds": time.monotonic() - started,
            "segments": entries,
        }
        (out / (path.stem + ".json")).write_text(json.dumps(output, indent=2) + "\n")
        print(
            f"{path.name}: {len(entries)} masks, {output['seconds']:.1f}s", flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.images, args.detections, args.out)
