# ROOM10B: original versus HD input to DA3

Run `tools/probe_room3d_hd.py` with the existing vision environment:

```sh
build/room3d-vision-venv/bin/python tools/probe_room3d_hd.py \
  --source build/room3d-local-pipeline/room10b0 \
  --hd build/mod-analysis/hires/bgd \
  --out build/room3d-hd-probe
```

Use a new output directory. The probe does not modify the batcher or game assets.
It processes all four cameras jointly, with the same DA3-SMALL revision and
camera baselines used in `room3d_depth.py`. No model retraining or external
inference is involved.

## Experimental controls

Inputs are 320x240 originals and 1280x960 HD backgrounds. HD camera IDs follow
`tools/data/re1_gog_usa_hd_backgrounds.csv`: 8CA9A873, 196082CE, 83C8E167,
C12E6487. Intrinsics scale by four for HD inputs; extrinsics are unchanged.
The four input pairs were visually checked for corresponding viewpoints.
The HD pack changes lighting, color, and detail, so this is not a pure
resolution experiment.

Both input sets run at processing resolutions 336 and 672, yielding depth maps
336x252 and 672x504. This separates the effect of the input pack from merely
increasing model processing resolution. All runs use CPU, four threads, seed 42.

For diagnostic geometry comparison, predicted depth is resampled to the same
320x240 grid and meshed with the same stride and 8% discontinuity filter.
All four meshes use the ORIGINAL textures to avoid attributing prettier HD
textures to improved geometry. The same cross-view free-space filter is used
for all variants. Full-resolution predicted depths are retained in NPZ files.
This normalization tests room structure, not the maximum mesh detail achievable
from high-resolution predictions.

The HTML includes every original camera and camera 1 shifted sideways by
1000 game units. Depth visualizations share one range per camera across all
variants. Coverage is reported as a diagnostic only: filling pixels does not
prove that geometry is correct. There is no ground-truth room mesh.

This experiment evaluates DA3 depth and diagnostic surface fusion. It does not
rerun the complete segmentation, primitive fitting, review and game export
stages of the batcher. It cannot by itself establish whole-pipeline quality.

Outputs: `build/room3d-hd-probe/index.html`, `manifest.json`, per-variant OBJ/MTL,
raw NPZ predictions and diagnostic renders. Log: `build/room3d-hd-probe.log`.

## Observed result (2026-10-11)

All four variants completed. Inference plus diagnostic mesh export took
4.59/19.95 seconds for original 336/672 and 3.98/19.77 seconds for HD 336/672.
Visual inspection covered all four original cameras and the displaced camera.
HD is not consistently better: camera 0 shows additional protruding overlapping
surfaces near the picture/door, and the lamp/stairwell remains fragmented.
Increasing resolution does not consistently resolve this. No batcher default
was changed. Retain originals for depth; HD appearance can be evaluated
separately as texturing. This finding applies to this pack and ROOM10B only.
