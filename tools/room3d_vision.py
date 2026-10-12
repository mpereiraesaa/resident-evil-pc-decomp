"""Optional local object detector. Downloads model weights, never uploads images.

Requires torch (CPU is sufficient), torchvision and transformers. Model proposals
remain unverified; bounding boxes are not recovered object geometry.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

MODEL = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
LABELS = [
    "door",
    "wall",
    "wooden floor",
    "ceiling",
    "table",
    "typewriter",
    "storage chest",
    "wooden barrel",
    "wooden crate",
    "wall lamp",
    "chandelier",
    "staircase",
    "railing",
    "painting",
    "window",
]


class Detector:
    def __init__(self, labels=None):
        self.labels = list(LABELS if labels is None else labels)
        if not self.labels or any(
            not isinstance(label, str) or not label.strip() or "." in label
            for label in self.labels
        ):
            raise ValueError("Expected nonempty category names without prompt separators")
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        torch.set_num_threads(4)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=False
        )
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                MODEL, revision=REVISION, trust_remote_code=False, use_safetensors=True
            )
            .to("cpu")
            .eval()
        )

    def detect(self, image):
        inputs = self.processor(
            images=image, text=". ".join(self.labels) + ".", return_tensors="pt"
        )
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.28,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result.get("text_labels", result.get("labels"))
        proposals = []
        for label, box, score in zip(
            labels, result["boxes"], result["scores"], strict=True
        ):
            if not isinstance(label, str):
                raise TypeError("Detector did not return semantic text labels")
            proposals.append(
                {
                    "label": label,
                    "box": [round(float(v), 2) for v in box],
                    "score": round(float(score), 4),
                    "status": "proposal",
                }
            )
        return proposals


def run(images, out, labels=None):
    out.mkdir(parents=True, exist_ok=True)
    detector = Detector(labels)
    records = []
    for path in images:
        image = Image.open(path).convert("RGB")
        proposals = detector.detect(image)
        record = {
            "image": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "model": MODEL,
            "revision": REVISION,
            "device": "cpu",
            "prompt_labels": detector.labels,
            "detections": proposals,
        }
        (out / (path.stem + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        # Scientific annotation of detector output; original input stays intact.
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for item in proposals:
            x0, y0, x1, y1 = item["box"]
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1, y1), outline="#ffd166", width=1)
                draw.text(
                    (max(0, x0), max(0, y0 - 10)),
                    item["label"],
                    fill="#ffd166",
                    stroke_width=1,
                    stroke_fill="black",
                )
        annotated.save(out / (path.stem + "-detections.png"))
        records.append(record)
        print(f"{path.name}: {len(proposals)} proposals", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", help="Override detection categories; recorded in every result")
    args = parser.parse_args()
    run(args.images, args.out, args.labels)
