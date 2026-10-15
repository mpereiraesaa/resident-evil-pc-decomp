"""Local part-category proposals with exact image/mask cache identity.

These records annotate evidence. They never approve or relabel generated geometry.
"""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from room3d_review import MODELS

MODEL, REVISION = MODELS['4b']
PROMPT = (
    'The cyan outline marks a candidate region in a low-resolution game background. '
    'Classify the visible structure INSIDE the outline, using the whole scene as context. '
    'Reply with exactly one letter: A=staircase with visible treads; B=railing or banister; '
    'C=door panels; D=shelf; E=floor or wall ornament; F=other; G=uncertain. '
    'Do not infer hidden treads from a railing.'
)
CATEGORIES = {'A': 'visible_treads', 'B': 'railing', 'C': 'door_panels', 'D': 'shelf',
              'E': 'floor_or_ornament', 'F': 'other', 'G': 'uncertain'}


def signature(room, camera, item, image, mask):
    if (mask.dtype != bool or mask.shape != (image.height, image.width)
            or not mask.any()):
        raise ValueError('Expected nonempty boolean mask matching image dimensions')
    name = item['mask']
    if Path(name).name != name or '\\' in name:
        raise ValueError('Mask must be a filename')
    return {'version': 1, 'room': Path(room['source']).stem.lower(),
            'camera': camera['index'], 'mask': name,
            'image_rgb_sha256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),
            'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
            'model': MODEL, 'revision': REVISION, 'prompt': PROMPT}


def record_path(root, identity):
    room = identity['room']
    if not room.startswith('room') or not room[4:] or any(c not in '0123456789abcdef' for c in room[4:]):
        raise ValueError('Invalid room identifier')
    return root / room / f"camera{identity['camera']}" / (Path(identity['mask']).stem + '.review.json')


def read_review(path, identity):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('signature') != identity:
        raise ValueError(f'Part review image/mask/model/prompt mismatch: {path}')
    response = record.get('response', '').strip()
    expected = CATEGORIES.get(response)
    if record.get('category') != expected or record.get('status') != 'unverified_part_category':
        raise ValueError(f'Invalid part review proposal: {path}')
    return record


def annotate_segments(room, images, segments, root):
    output, proposals = [], []
    for camera, image, entries in zip(room['cameras'], images, segments, strict=True):
        view = []
        for item, mask in entries:
            if 'staircase' not in item['label']:
                view.append((item, mask))
                continue
            identity = signature(room, camera, item, image, mask)
            path = record_path(root, identity)
            record = read_review(path, identity)
            if record is not None:
                item = {**item, 'part_review': {'category': record['category'],
                        'status': record['status'], 'file': str(path)}}
                proposals.append({'camera': camera['index'], 'mask': item['mask'],
                                  **item['part_review']})
            view.append((item, mask))
        output.append(view)
    return output, proposals


class Reviewer:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch.set_num_threads(4)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=False,
            use_safetensors=True).to('cpu').eval()

    def review(self, identity, image, mask):
        rgb = np.asarray(image.convert('RGB')).copy()
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 255, 255), 1)
        annotated = Image.fromarray(rgb)
        inputs = self.processor.apply_chat_template([
            {'role': 'user', 'content': [{'type': 'image', 'image': annotated},
                                        {'type': 'text', 'text': PROMPT}]}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                            temperature=None, top_p=None, top_k=None)
        response = self.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True)[0]
        return {'signature': identity, 'response': response,
                'category': CATEGORIES.get(response.strip()), 'status': 'unverified_part_category'}


def main():
    from room3d_refine import load_depth
    from room3d_segment import load_masks
    from room3d_stair_edges import load_refinements

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--rooms', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--raw-masks', action='store_true', help='Diagnostic only: bypass cached tread refinements')
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.batch.resolve()):
        parser.error('Review output must be outside the source batch')
    if any(not name.startswith('room') or len(name) != 8
           or any(c not in '0123456789abcdef' for c in name[4:]) for name in args.rooms):
        parser.error('Expected lowercase room identifiers such as room1010')
    reviewer = None
    for name in args.rooms:
        root = args.batch / name
        room = json.loads((root / 'survey.json').read_text())
        images = [Image.open(root / f"camera{c['index']}-original.png").convert('RGB') for c in room['cameras']]
        segments = load_masks(args.batch / 'segmentation' / name, room, images)
        if not args.raw_masks and (root / 'stair-refinement/refinements.json').exists():
            depths = load_depth(args.batch / 'depth' / name, root / 'survey.json', room, images)
            segments = load_refinements(root / 'stair-refinement', room, images, segments, depths)
        for camera, image, entries in zip(room['cameras'], images, segments, strict=True):
            for item, mask in entries:
                if 'staircase' not in item['label']:
                    continue
                identity = signature(room, camera, item, image, mask)
                path = record_path(args.out, identity)
                record = read_review(path, identity)
                if record is None:
                    reviewer = reviewer or Reviewer()
                    record = reviewer.review(identity, image, mask)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
                    temporary.write_text(json.dumps(record, indent=2) + '\n')
                    temporary.replace(path)
                print(name, camera['index'], item['mask'], record['category'], flush=True)


if __name__ == '__main__':
    main()
