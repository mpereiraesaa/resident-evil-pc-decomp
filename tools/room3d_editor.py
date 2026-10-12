"""Generate a self-contained offline transform editor for a completed batch room."""

import argparse
import base64
import json
from pathlib import Path

from room3d_preview import camera_basis
from room3d_retouch import catalog, read_mesh


def build(room, out):
    if out.exists():
        raise ValueError("Choose a new editor filename")
    if out.resolve().is_relative_to(room.resolve()):
        raise ValueError("Store the editor outside the automatic room output")
    mesh = read_mesh(room / "assisted.obj")
    vertices, _, faces, _ = mesh
    groups = catalog(mesh)
    for group in groups.values():
        group["edges"] = []
    # Deduplicate by endpoint positions, because the baker duplicates face vertices.
    edges = {name: set() for name in groups}
    for name, _, refs in faces:
        for a, b in zip(refs, refs[1:] + refs[:1], strict=True):
            points = tuple(sorted((tuple(vertices[a[0]][:3]), tuple(vertices[b[0]][:3]))))
            edges[name].add(points)
    for name, values in edges.items():
        groups[name]["edges"] = sorted(values)
    survey = json.loads((room / "survey.json").read_text())
    cameras = []
    for camera in survey["cameras"]:
        image = room / f"camera{camera['index']}-original.png"
        cameras.append({**camera, "basis": camera_basis(camera).tolist(),
                        "image": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()})
    data = {"room": room.name, "groups": groups, "cameras": cameras}
    payload = json.dumps(data).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace("/*ROOM_DATA*/ null", payload))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.room, args.out)
    print(f"{args.out}: {len(data['groups'])} groups, {len(data['cameras'])} cameras")


if __name__ == "__main__":
    main()
