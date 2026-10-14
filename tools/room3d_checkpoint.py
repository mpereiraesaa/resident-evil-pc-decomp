"""Content-checked room checkpoints; failed or partial rooms are never reusable."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    if root.is_file():
        return {".": digest(root)}
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_signature(args):
    options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"resume", "rooms", "recipes", "out", "jobs", "overrides"}
    }
    inputs = {}
    for key in ("assets", "authored", "detections", "segmentation", "depth"):
        root = getattr(args, key, None)
        if root is not None:
            if args.out.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Output must not be inside --{key}")
            inputs[key] = inventory(root)
    code = {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    return signature({"options": options, "inputs": inputs, "code": code})


def successful(result):
    if result.get("status") not in {"survey", "mesh", "baseline", "image_assisted", "empty_slot"}:
        return False
    if any("error" in camera for camera in result.get("cameras", [])):
        return False
    runtime = result.get("runtime", {})
    return not runtime or runtime.get("status") == "not_run" or runtime.get("passed", False)


def save(path, key, result, room_dir):
    if not successful(result):
        return
    artifacts = inventory(room_dir) if room_dir.exists() else {}
    record = {"schema": 1, "key": key, "result": result, "artifacts": artifacts}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def restore(path, key, room_dir):
    try:
        record = json.loads(path.read_text())
        if record["schema"] != 1 or record["key"] != key or not successful(record["result"]):
            return None
        artifacts = inventory(room_dir) if room_dir.exists() else {}
        if artifacts != record["artifacts"]:
            return None
        return record["result"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
