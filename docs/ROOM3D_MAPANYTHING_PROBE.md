# Local MapAnything probe: ROOM10B

Completed on RTX 3060 Laptop, 6 GiB, 2026-10-11. No cloud inference or
image upload. Four original backgrounds and RDT cameras were inputs;
DA3 depth was not supplied.

## Reproduce

Source checkout: `build/map-anything-validation`, revision
`3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
Model: `facebook/map-anything`, revision
`a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`.
Checkpoint SHA256 recorded by Hugging Face:
`981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0`.
Isolated environment: `build/room3d-mapanything-venv` (Python 3.12,
PyTorch 2.6.0+cu124, torchvision 0.21.0, UniCeption 0.1.7).

```sh
build/room3d-mapanything-venv/bin/python tools/probe_room3d_mapanything.py \
  --source build/room3d-local-pipeline/room10b0 \
  --model build/room3d-mapanything-model \
  --out build/room3d-mapanything-probe/NEW_RUN --size 322
build/room3d-mapanything-venv/bin/python tools/compare_room3d_mapanything.py \
  --source build/room3d-local-pipeline/room10b0 \
  --predictions build/room3d-mapanything-probe/NEW_RUN \
  --baseline build/room3d-tsdf-probe/room10b0/fused.obj
```

Use a new output directory for each inference. Successful run:
`build/room3d-mapanything-probe/run05/index.html`.
Raw tensors are preserved as four NPZ files, with measurements in manifest.json.

## Memory and failures

Run05 processed all four views at 322x238 in 23.91 seconds including model
loading (excluding downloads and comparison rendering). Peak PyTorch allocated
memory: 5,190,659,584 bytes; reserved: 5,226,102,784 bytes (4.87 GiB).
This is allocator telemetry, not total system/GPU memory.

- Run01: BF16 weights caused an indexed-assignment dtype mismatch in the
  upstream camera-conditioning code.
- Runs02/03: FP32 weights with the default AMP cache exhausted VRAM at
  resolutions 322 and 224 respectively.
- Run04: disabling AMP caching allowed the forward pass; an outer autocast
  context also affected postprocessing and caused unsupported BF16 LU solving.
- Run05: FP32 weights, upstream BF16 AMP, global AMP cache disabled, minibatch
  one, memory-efficient inference. No upstream model code was changed.

Logs are `build/mapanything-run01.log` through `run05.log`.

## Interpretation

The comparison exports per-view camera-space points at the original RDT
poses, using 1000 game units per assumed metre. It does not use the model's
predicted poses to reposition views. Textures are the preprocessed inputs;
triangles are sampled every four pixels, masked, and rejected across depth
jumps above 8%. This is a diagnostic surface union, not a fused or watertight
room. It can contain overlapping surfaces and missing geometry.

The HTML compares original backgrounds, prior DA3/TSDF surfaces and MapAnything
from all four RDT cameras plus a camera translated 1000 game units sideways.
Coverage is recorded, but is not geometric accuracy. Visual review of camera 0
already shows holes and inconsistent overlapping surfaces; completing local
inference does not establish an improvement suitable for the game.

Weights retain their upstream license; downloading ungated weights does not
make this checkpoint Apache-licensed. No weights/assets are added to the fork.
