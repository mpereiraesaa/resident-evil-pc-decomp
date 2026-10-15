# Small local reconstruction comparison

Report: `build/room3d-alternatives-probe/index.html`. Original game assets and
model artifacts remain local and untracked. No images were sent to a service.

## Room surfaces

`probe_room3d_dense.py` backprojects one DA3 depth map into a textured surface.
`probe_room3d_multiview.py` compares joining those surfaces with rejecting
triangles in another camera's confident free space (8% depth tolerance).
`probe_room3d_tsdf.py` fuses signed distances in a 150-unit voxel grid, with
450-unit truncation, before extracting a surface and assigning source textures.
These exports are diagnostic, not runtime-ready game meshes.

```sh
build/room3d-tools-venv/bin/python tools/probe_room3d_multiview.py \
  --input build/room3d-local-pipeline --out build/room3d-multiview-probe
build/room3d-fusion-venv/bin/python tools/probe_room3d_tsdf.py \
  --input build/room3d-local-pipeline --out build/room3d-tsdf-probe
build/room3d-fusion-venv/bin/python tests/test_room3d_fusion.py
```

The isolated fusion environment uses Python 3.12, NumPy, Pillow, OpenCV and
scikit-image. The two synthetic checks verify camera reprojection and fusion of
a known plane observed from two translated cameras. They do not establish room
quality. Ruff and diff whitespace checks also passed.

Single-view surfaces preserve appearance but have large unseen regions. Joining
views creates overlapping surfaces because the depth estimates disagree. TSDF
reduces some fragments in ROOM1180, but holes and distorted fixtures persist in
ROOM10B0. Original-camera texture comparisons reuse their inputs; they are not
independent reconstruction accuracy. Excluded-surface comparisons omit that
camera's surface/texture, but the upstream DA3 estimates used all views.
Displaced cameras have no original reference and are not tested gameplay routes.

## Generative object experiment

The collaboration subagent ran TRELLIS.2 through the community
[trellis.cpp](https://github.com/pwilkin/trellis.cpp) v0.8.1 CUDA runtime, with
[GGUF weights](https://huggingface.co/ilintar/trellis2-gguf), on the RTX 3060
Laptop 6 GB. Streaming decoders from RAM and quantized weights allowed execution
without replacing drivers or the existing Python environments. This is not the
unmodified Microsoft reference pipeline.

Input: ROOM1180 camera 1 chest, existing SAM mask, 71x45-pixel crop. Parameters:
512 resolution, 12 steps, seed 42, 1024 PNG atlas. Three actual runs:

| Input/model | Seconds | Peak sampled process VRAM |
|---|---:|---:|
| Original / Q4 | 61.8 | 1204 MiB |
| Exposure x2.5 / Q4 | 66.9 | 1226 MiB |
| Original / DINO and structure Q8, remaining stages Q4 | 64.9 | 1702 MiB |

One-second sampling can miss brief peaks. The last run does not evaluate a
fully Q8 model. Each run generated a volumetric box on an invented platform;
exposure and partial higher precision did not eliminate it. A valid GLB or a
closed mesh does not establish fidelity.

The follow-up geometry-only cleanup detected the dominant horizontal support
plane and retained the main elevated component. `cleaned.glb` has 11,776 faces,
UV/material preserved, but 126 boundary edges remain open at the base. Three
cut tolerances retained the same principal X/Z bounds. No manual modeling or
hidden hole repair was applied. Raw outputs are preserved. The isolated object
has visible lid and side geometry, but its proportions and details are inferred;
its texture is dark and it has not been placed in the game.

Reproduction wrapper: `tools/probe_room3d_object.py`. Artifacts, exact commands,
hashes, VRAM samples, upstream verification and licenses:
`build/room3d-object-probe/README.md`, each run's `manifest.json`, and
`chest-q8-conditioning/cleanup.json`. The local cleanup and diagnostic render
scripts are `build/clean_object_support_probe.py` and
`build/render_object_fast.py`. The latter uses painter ordering and per-face
texture sampling; shaded geometry views are for shape inspection, not final
PBR or game rendering. A failed initial CLI argument attempt and terminated
slow-render attempts remain recorded separately from completed inference.

**Decision:** low-memory local object generation is viable and worth evaluating
further; current appearance and geometry are not ready for integration. None of
the tested room-surface methods is a satisfactory complete room reconstruction.
No inference, download or model-training job remains running from this probe.

Review these experiments separately from the existing room runtime changes:
room-surface probes/tests/documentation form one review unit; the generative
object wrapper and any future production cleanup form another. No PR opened.
