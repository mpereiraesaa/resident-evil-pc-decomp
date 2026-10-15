# GenRecon local probe

Status: input prepared; inference NOT run. No cloud job or image upload.

The official source is checked out in `build/genrecon-validation`.
ROOM10B's four backgrounds, COLMAP cameras and estimated point cloud are in
`build/room3d-genrecon-probe/room10b0`. Its manifest records input hashes and
coordinate conversion. The cloud contains 11,494 confident DA3 depth samples;
it is not SfM evidence or measured ground truth. Scale assumes 1000 game units
per metre. The game-to-world rotation maps `(X,Y,Z)` to `(X,Z,-Y)`.
Round-trip projection error is below 1e-6 pixels; this verifies conversion,
not the physical accuracy of estimated depth or the original camera model.

Reproduce into a NEW output directory:

```sh
build/room3d-fusion-venv/bin/python tools/prepare_room3d_genrecon.py \
  --source build/room3d-local-pipeline/room10b0 \
  --depth build/room3d-local-pipeline/depth/room10b0 \
  --out build/room3d-genrecon-probe/room10b0
build/room3d-fusion-venv/bin/python -m unittest discover -s tests -p test_room3d_genrecon.py
```

## Current blocker

The official pipeline uses `facebook/dinov3-vitl16-pretrain-lvd1689m`.
On 2026-10-11 its Hub API reported manual gated access and unauthenticated
HEAD requests for both config and weights returned HTTP 401. No HF login or
cached weights are present. User must request/accept access at:
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
then authenticate locally with `hf auth login`. This is download access;
it does not require cloud GPU billing or uploading backgrounds.

Local hardware: RTX 3060 Laptop 6 GiB, 215 GiB free disk at preflight.
Official code has `low_vram=True` and projection batching; fitting inference
in 6 GiB is unverified. CUDA toolkit/extensions and GPU PyTorch still need
an isolated environment. Do not treat trellis.cpp object inference as evidence
that this distinct PyTorch scene pipeline fits.

After dependencies are ready, use `reconstruct_scene.py --mode Iphone` with
`--colmap_subdir colmap --num_imgs_per_scene 4 --proj_batch_voxels 512`, the
three official stage checkpoints, and `configs/pipelines/texture.json`.
Start with seed 42 and pipeline 512; monitor GPU and system RAM. Inspect point
cleaning/chunk layout before judging reconstruction. The cloud's ERROR=0
entries are format placeholders and have no correspondence tracks.

Next validation: export GLB, transform it back to game coordinates, compare
all original views plus moved cameras against the previous TSDF result.
Only actual generated meshes should enter the comparison as GenRecon results.
