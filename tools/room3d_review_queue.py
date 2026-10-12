"""Run deferred local visual reviews without regenerating or mutating room exports."""

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
from pathlib import Path

from room3d_review import MODELS, PROMPT, reuse_review


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_index(roots):
    """Reuse identical image pairs across scenario variants, never similar images."""
    index = {}
    model, revision = MODELS['4b']
    for root in roots:
        if root is None:
            continue
        for path in sorted(root.glob('room*/camera*-review.json')):
            try:
                record = json.loads(path.read_text())
                if not isinstance(record, dict):
                    continue
                hashes = tuple(item['sha256'] for item in record['inputs'])
                if (record.get('model') == model and record.get('revision') == revision
                        and record.get('prompt') == PROMPT and isinstance(record.get('response'), str)
                        and len(hashes) == 2):
                    index[hashes] = path
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return index


def enqueue(room, cameras):
    jobs = []
    for camera in cameras:
        stem = f"camera{camera['index']}"
        inputs = [room / (stem + suffix) for suffix in ("-original.png", "-assisted.png")]
        jobs.append({"camera": camera["index"], "inputs": [
            {"file": path.name, "sha256": sha(path)} for path in inputs]})
    queue = {"schema": 1, "room": room.name, "status": "pending", "size": "4b", "jobs": jobs}
    (room / "review-queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    return queue


def checked_inputs(room, job):
    if len(job["inputs"]) != 2:
        raise ValueError("Review job requires exactly two images")
    paths = []
    for entry in job["inputs"]:
        path = (room / entry["file"]).resolve()
        if not path.is_relative_to(room.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("Changed or non-local queued input; regenerate this room's review queue")
        paths.append(path)
    return paths


def run_queue(queue_file, out, python, cache=None, shared_cache=None):
    room = queue_file.parent
    queue = json.loads(queue_file.read_text())
    if queue.get("schema") != 1 or queue.get("size") != "4b":
        raise ValueError("Unsupported review queue")
    if out.resolve().is_relative_to(room.resolve()):
        raise ValueError("Reviews must be outside the room export to preserve its checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for job in queue["jobs"]:
        camera = int(job["camera"])
        path = out / f"camera{camera}-review.json"
        try:
            reference, reconstruction = checked_inputs(room, job)
            source = path if path.exists() else (cache / path.name if cache else None)
            try:
                reused = reuse_review(source, reference, reconstruction, path)
            except (ValueError, KeyError, TypeError):
                # An interrupted review file must be retried, not permanently cached.
                reused = False
            hashes = tuple(item['sha256'] for item in job['inputs'])
            if not reused and shared_cache is not None:
                source = shared_cache.get(hashes)
                try:
                    reused = reuse_review(source, reference, reconstruction, path)
                except (OSError, ValueError, KeyError, TypeError):
                    reused = False
            if not reused:
                with path.with_suffix(".log").open("w") as log:
                    subprocess.run([str(python), str(Path(__file__).with_name("room3d_review.py")),
                                    str(reference), str(reconstruction), "--out", str(path), "--size", "4b"],
                                   check=True, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1"))
            review = json.loads(path.read_text())
            if shared_cache is not None:
                shared_cache[hashes] = path
            results.append({"camera": camera, "status": "reviewed", "reused": reused})
            rows.append(f'<h2>Camera {camera}</h2><pre>{html.escape(review["response"])}</pre>')
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            results.append({"camera": camera, "status": "error", "error": str(error)})
            rows.append(f'<h2>Camera {camera}</h2><p>{html.escape(str(error))}</p>')
        print(f'{room.name} camera{camera}: {results[-1]["status"]}', flush=True)
        (out / "status.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
        (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Local review</title>'
                                       '<h1>Local visual review</h1><p>Model proposals, not verified approval.</p>'
                                       + "".join(rows))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cache", type=Path, help="Existing batch containing hash-matching camera reviews")
    parser.add_argument("--rooms", nargs="+")
    args = parser.parse_args()
    queues = sorted(args.batch.glob("room*/review-queue.json"))
    if args.rooms:
        wanted = {name.lower() for name in args.rooms}
        queues = [path for path in queues if path.parent.name in wanted]
        missing = wanted - {path.parent.name for path in queues}
        if missing:
            parser.error("No review queue for: " + ", ".join(sorted(missing)))
    if not queues:
        parser.error("No completed room review queues found")
    failed = False
    shared_cache = review_index([args.cache, args.out])
    for path in queues:
        results = run_queue(path, args.out / path.parent.name, args.python,
                            args.cache / path.parent.name if args.cache else None, shared_cache)
        failed |= any(result["status"] == "error" for result in results)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
