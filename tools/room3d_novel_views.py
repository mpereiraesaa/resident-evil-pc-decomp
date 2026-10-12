"""Inspect inferred surfaces from new eye-height viewpoints without camera cutaways."""

import json

import numpy as np
from room3d_preview import render


def inspect(mesh, room, out):
    out.mkdir(parents=True, exist_ok=True)
    bounds = room["boundaries"]
    x0, x1 = min(b["xmin"] for b in bounds), max(b["xmax"] for b in bounds)
    z0, z1 = min(b["zmin"] for b in bounds), max(b["zmax"] for b in bounds)
    spots = []
    for x in np.linspace(x0 + 600, x1 - 600, 11):
        for z in np.linspace(z0 + 600, z1 - 600, 11):
            if any(
                b["xmin"] - 350 <= x <= b["xmax"] + 350
                and b["zmin"] - 350 <= z <= b["zmax"] + 350
                for b in bounds
            ):
                continue
            spots.append((float(x), float(z)))
    records = []
    rows = [
        "<h1>Inferred surfaces from new viewpoints</h1>",
        "<p>No original reference exists for these views. Eye height and sampling positions are provisional, not a validated shoulder-camera implementation.</p>",
    ]
    if spots:
        center = np.mean(spots, axis=0)
        selected = np.linspace(0, len(spots) - 1, min(4, len(spots))).astype(int)
        for i, index in enumerate(selected):
            x, z = spots[index]
            target = (
                center
                if np.linalg.norm(center - [x, z]) > 100
                else np.array([x, z + 1000])
            )
            camera = {
                "index": -1,
                "origin": [x, -1500, z],
                "target": [float(target[0]), -1500, float(target[1])],
                "roll": 0,
                "fov": 207,
            }
            image, covered = render(mesh, camera)
            name = f"view{i}.png"
            image.save(out / name)
            records.append(
                {
                    "camera": camera,
                    "image": name,
                    "geometry_coverage": float(covered.mean()),
                    "qualification": "Coverage is not correctness; original collisions only screen sample positions",
                }
            )
            rows.append(
                f'<h2>View {i}</h2><img src="{name}"><p>Geometry coverage: {covered.mean():.1%}</p>'
            )
    report = {
        "status": "rendered" if records else "no_clear_sample_positions",
        "views": records,
    }
    (out / "views.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>New viewpoints</title>'
        + "\n".join(rows)
    )
    return report
