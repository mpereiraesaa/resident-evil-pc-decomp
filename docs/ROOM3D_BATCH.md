# Room reconstruction batch experiment

## Shared structure and fixture separation (2026-10-11)

Reconstruction with cached or local DA3 now adds `room3d_structure.py` after
primitive refinement. It unions rectangular wall footprints, removes internal
vertical faces between overlapping wall volumes, and emits shared exposed
planes. Original collision/script records remain read-only. Existing explicit
wall skips and owner heights are respected. This remains an architectural prior,
not proof that every collision rectangle is an actual visual wall.

Plane coordinates use confident wall-mask depth samples pooled across cameras.
Lamps and railings are excluded from those masks, including overlapping SAM
masks. A wall adjustment requires at least 30 nearby points in each of two
views, per-view medians within 300 game units, and a maximum displacement of
200 units from the collision prior. Shared coordinate mappings keep corner
intersections joined. Floor adjustments similarly require two views and are
bounded to 100 units. Without sufficient evidence, the original plane remains.
There is no claim of complete multi-level floors or watertight room recovery.

Architectural planes and the floor are tessellated in patches of at most
approximately 1000 units (up to 64 subdivisions per edge) before baking. This
keeps the geometry planar while avoiding texture loss when a large triangle's
vertices are off-screen. The first comparison exposed that failure and is
preserved at `build/room3d-structure-comparison`; v2 and later improve it.

Thin objects now include separate railing proposals, even when a combined
`staircase railing` label also supplies the existing approximate staircase
incline. A fixture cell crossing a depth jump above 15% is not generated.
Confident lamp/railing masks veto architectural texture samples regardless of
overlapping wall masks; railing pixels cannot provide staircase completion
material. Empty rejected fixtures cannot fall back to a solid bounding box.

The multi-view contradiction gate samples eight corners and the center of each
inferred fixture cell. It removes a cell only when at least seven samples have
two or more confident free-space votes and more conflicting than supporting
views. Unknown depth, image borders, depth discontinuities and surfaces behind
an observed foreground do not count as contradictions. Architectural planes and
script-owned furniture are not punched full of holes by this filter.

Each room writes `before-structure-recipe.json` and `structure.json`; the latter
is linked from the HTML and included in `result.json`. Set the room recipe's
`consolidate_structure` to false to disable shared-plane consolidation and its
fixture contradiction pass. Fixture proposals and semantic fixes remain active.

Validation uses ROOM10B and ROOM118 with their existing original-image, DA3 and
SAM caches. The new synthetic tests cover wall-union boundaries, free space vs
occlusion/unknown depth, depth discontinuities, fixture priority over overlapping
wall masks, railing material separation and empty-fixture rendering. All 34
batch tests and Ruff checks passed. No coverage or depth-agreement number should
be presented as reconstruction fidelity.

The validated run is `build/room3d-structure-validated/index.html`; the side-by-side
original/before/after report is `build/room3d-structure-before-after/index.html`.
The comparison reuses the previous `build/room3d-local-pipeline` output and checks
that the displaced camera definitions match. ROOM10B produces 24 exposed wall
runs and ROOM118 produces 14. Both retain the original floor height because the
floor-fitting evidence is insufficient. ROOM10B now has a separate 60-cell
railing proposal and a 12-cell lamp. The contradiction gate removed no fixture
cells in these two rooms; its rejection behavior is covered by synthetic tests.

Visual inspection shows recognizable door/painting detail in ROOM10B's first
view and continuous planar walls, but severe staircase/lamp errors persist in
other views, along with texture stretching, missing upper surfaces and unobserved
areas. ROOM118 remains approximate, with texture seams and coarse furniture.
This is not a faithful reconstruction or a demonstrated improvement in every view.

Both isolated Wine smoke tests passed: ROOM10B loaded the generated mesh and
changed camera; ROOM118 additionally opened and closed the chest menu. Runtime
mesh/material/texture/cutaway hashes match the reviewed artifacts, and input RDT
and background hashes match the source assets. These tests do not verify item
transfer, saves, or visual fidelity. Reproduce the cached reconstruction with:

```sh
build/room3d-vision-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-structure-validated \
  --rooms ROOM10B0 ROOM1180 --reconstruct \
  --detections build/room3d-local-pipeline/detections \
  --segmentation build/room3d-local-pipeline/segmentation \
  --depth build/room3d-local-pipeline/depth \
  --runtime-exe build/windows-cross/residentevil.exe
```

Use a new output directory for subsequent runs to preserve this comparison.

`tools/batch_room3d.py` produces reproducible, local comparison artifacts from
original USA PC assets. It exports approximate geometry without changing the
source assets, scripts or existing authored ROOM118 mesh. Optional runtime
probes launch an isolated game copy with the generated mesh. Model predictions
and hidden-surface completion are estimates, not recovered original geometry.

## Complete local pipeline

With the tools and vision environments described below installed, run:

```sh
build/room3d-tools-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-local-pipeline \
  --rooms ROOM1180 ROOM10B0 --reconstruct --local-models \
  --vision-python build/room3d-vision-venv/bin/python \
  --runtime-exe build/windows-cross/residentevil.exe
```

Use a fresh output directory for each comparison. DINO proposes objects, SAM
segments them, DA3 estimates depth conditioned on the original cameras, and
bounded geometry trials render every original view before accepting changes.
Qwen 4B reviews the resulting images locally. Its prose findings remain
unverified; numeric depth and silhouette gates govern geometry acceptance.
Weights may be downloaded; images are processed locally.

Each room links to reversible trials, material provenance, visual reviews,
novel viewpoints and runtime evidence. `--review-cache <previous-output>` only
reuses reviews whose model revision, prompt and both image hashes match.
The complete command passed for both rooms in `build/room3d-local-pipeline`:
`manifest.json` contains two successful rooms, both runtime reports pass, and
`audit.json` verifies the final artifacts against the original assets.

After it finishes, audit the evidence against current assets and the executable:

```sh
build/room3d-tools-venv/bin/python tools/check_room3d_pipeline.py \
  build/room3d-local-pipeline \
  --assets assets/USA \
  --executable build/windows-cross/residentevil.exe
```

The audit decodes the original PAKs to verify the reference images and re-renders
the mesh to check that Qwen reviewed its current appearance,
checks model revisions and input hashes, validates depth provenance, and compares
the mesh, cutaways, materials and textures copied into the Wine probe. Both room
scenarios must pass; ROOM118 additionally requires chest-menu open/close. It
checks that the final recipe matches the accepted reversible state and that the
HTML and novel-view reports exist. A passing `audit.json` establishes consistency
of these artifacts, not fidelity or exhaustive gameplay compatibility.

### Validated result

| Requirement | Evidence in `build/room3d-local-pipeline` |
|---|---|
| Local DINO, SAM 2.1 and DA3 | Per-camera detector/mask records and per-room depth arrays with pinned model revisions and source hashes |
| Local Qwen 4B review | Six camera reviews; ROOM118 camera 0 reused only after exact model/prompt/image-hash matching |
| Reversible correction | `corrections/` retains each recipe, mesh, multiview render, score and accepted/rejected decision |
| Original game data | Audit verifies original RDT/PAK hashes and reference decoding; exports change only visual geometry |
| Approximate hidden surfaces | Closed object primitives, inferred ceiling/roof and material patches with source crops; novel-view inspection |
| Runtime | ROOM10B mesh load and camera transition; ROOM118 additionally opens and closes the chest menu |
| Comparison and consistency | Both room HTML reports, new-view reports and passing `audit.json` |
| Automated checks | 27 synthetic/regression tests pass; platform-boundary check and `git diff --check` pass |

ROOM118 camera 0's capped log-depth error decreased from 0.17796 to 0.09053;
camera 1 stayed approximately 0.11716. ROOM10B camera 2 decreased from 0.29868
to 0.28904, while camera 3 increased from 0.33850 to 0.34108 within the acceptance
tolerance. These are comparisons against DA3 estimates, not true depth.

Hidden material continuation filled 2,570 vertex samples in ROOM118 and 8,340
in ROOM10B. This is activity evidence, not an accuracy measure. Unsupported
surfaces retain neutral appearance; the resulting scene is approximate and
does not establish a watertight reconstruction. Visual inspection shows coarse
fixtures, stretched/lost texture detail and an upper gap in ROOM10B. The
appearance-excluded camera comparisons are weaker than the all-view bakes;
geometry fitting still uses all original cameras.

The Wine probe logged a FAudio initialization exception and then continued to
the required gameplay events with the process still alive. `CrashLog.cpp`
registers both a vectored and an unhandled handler, so its heading alone does
not establish a terminal crash. These probes do not validate audio, item
transfer, save/load, every collision, or a shoulder-camera gameplay mode.

The remaining sections describe individual stages and historical experiments;
the evidence table above describes the final integrated run.

## Run

From the repository root (paths may also be absolute):

```sh
uv venv build/room3d-tools-venv
uv pip install --python build/room3d-tools-venv/bin/python numpy Pillow
build/room3d-tools-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-batch-comparison \
  --rooms ROOM1180 ROOM10B0 --authored tests/fixtures/room3d
```

Open `build/room3d-batch-comparison/index.html` in a browser. Every camera has
an original/automatic/authored comparison, plus an adjustable original/automatic
overlay. The authored column appears only when an existing OBJ is available.
No external web services, server or network connection are required to view it.
The original backgrounds in this local report are proprietary game assets;
they are not included in source control.

Omit `--rooms` to process every RDT. Use `--survey-only` to inventory without
rendering, or `--mesh-only` to export meshes without decoding backgrounds:

```sh
build/room3d-tools-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-batch-all --mesh-only
build/room3d-tools-venv/bin/python tests/test_room3d_batch.py
```

Each room output contains measured `survey.json`, an X/Z `plan.svg`, an OBJ/MTL
and camera visibility sidecar when generating meshes, and `reconstruction.json`
recording assumptions and corrections. `result.json` lists camera/background
availability and, when rendered, source hashes and screen coverage.
`manifest.json` aggregates successes, empty slots and errors. Errors produce a
nonzero exit status while allowing other rooms to finish. A missing comparison
background is also reported as incomplete. Four zero bytes identify empty RDT
slots; arbitrary truncated files remain errors. Stages 6/7 reuse backgrounds
from 1/2, following the game's normal cached background path.

## What the baseline geometry means (without `--reconstruct`)

- Camera origin, target, focal length and collision rectangles are measured.
- Rectangular collision types 1/5 become blocks, using estimated heights:
  6000 for records with flag 0x100 and 1500 otherwise. That heuristic is not
  a claim that the flag describes the visual height of an object.
- A flat floor covers the collision extents at estimated Y=0. There is no
  inferred ceiling, staircase, door aperture or furnishing.
- A block containing a camera is hidden from that camera as an automatic
  cutaway heuristic. Other cutaways may need explicit corrections.
- Other collision shapes remain in the measured survey and are listed as
  unresolved; they are not silently converted to walls.
- Materials are neutral diagnostic colors. Textures and baked lighting have
  not been recovered from the backgrounds.

The independent preview uses the camera basis, focal length, 320x240 centre and
near-plane rule used in `Room3D.cpp`. It supports perspective-correct depth,
UVs and vertex colors, diffuse PNG materials and camera visibility groups.
It samples textures with nearest filtering and excludes game actors, animated
objects, postprocessing and menus. Therefore it is **offline comparison**, not
proof that generated rooms work in the game. Nonzero camera roll is explicitly
unsupported, matching the current runtime room renderer.

Coverage measures how much of a rendered image contains geometry; a completely
wrong wall can cover every pixel. It must never be interpreted as a fidelity
percentage. No automatic visual-fidelity score is reported.

## Persistent corrections

Add `--resume` to a batch command to reuse completed rooms. Every successful
room gets an atomic record in `checkpoints/`; interrupted or failed rooms run
again. Resume verifies the contents of the asset tree, supplied geometry/model
input trees, Python tool sources, command options, that room's recipe and all
files in its output directory. Changing one recipe only invalidates that room.
Changing shared assets or tool code invalidates the whole batch conservatively.
Python environments and downloaded model weights are not fingerprinted: after
changing either, run without `--resume`. Model revisions remain pinned by the
individual inference tools. Do not run concurrent writers into the same output.

For a local full reconstruction (not just collision extrusion):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  build/room3d-vision-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-all-assisted \
  --reconstruct --local-models --defer-review --resume --jobs 2 --recipes corrections.json
```

This requires previously downloaded model weights and the configured local
environment. Omit `--rooms` to attempt every RDT. Inspect failures in the final
manifest; successful export does not establish visual quality or playability.
Checkpoints appear as each room completes, while the batch HTML and manifest
are written at the end. Repeating the same command resumes successful rooms.
Use `--rooms ROOM2010` to rerun a specific room, preferably into a separate
output directory when comparing alternatives. A subset run replaces the root
manifest with that subset, but retains other rooms' files and checkpoints.

`--defer-review` keeps detection, segmentation, depth and render-based geometry
fitting enabled, but records Qwen review jobs instead of waiting for inference
at each camera. Room results explicitly mark visual review as pending. Once rooms
are exported, run that separate pass with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  build/room3d-vision-venv/bin/python tools/room3d_review_queue.py \
  --batch build/room3d-all-assisted --out build/room3d-all-reviews
```

The review pass validates queued image hashes, reuses matching completed reviews,
and writes a per-room HTML/status report outside the geometry export, preserving
its checkpoint. `--rooms ROOM2010` selects a subset; `--cache OLD_BATCH` can reuse
reviews from the earlier synchronous workflow. Pending review is not approval.
The queue runner snapshots available rooms at startup; rerun it for rooms that
finish later. Without `--defer-review`, the synchronous behavior is unchanged.
Reviews are indexed by the exact ordered pair of reference/render hashes, pinned
model revision and prompt, so identical scenario variants can reuse them even
when room/camera names differ. This never skips reconstruction of either room
and does not reuse reviews merely because two pictures look similar.

`--jobs` defaults to 1. Higher values use independent spawned processes, one
room per task; geometry fitting and quality gates stay identical. Only the parent
writes the batch manifest/index, in input order, while each worker owns its room
and model-output directories. Worker count is excluded from checkpoint keys so
changing it alone does not invalidate completed outputs. Each model process may
itself use CPU threads and memory; two room workers were tested on the local
20-logical-CPU/32-GB machine. Interactive Wine probes require `--jobs 1`.

Pass `--recipes corrections.json` with lower-case RDT stems as keys:

```json
{
  "room1180": {
    "wall_height": 6000,
    "floor_y": 0,
    "obstacle_height": 1500,
    "boundary_heights": {"2": 1410},
    "skip_boundaries": [],
    "hide": [[1, "boundary_0"]],
    "boxes": []
  }
}
```

This example documents syntax; it is not a validated reconstruction recipe.
`boxes` entries are `[xmin, zmin, xmax, zmax, topY, baseY]`. Generated groups
are `boundary_N`, `floor` and `authored_box_N`. Negative Y is up. Corrections
are recorded in the room's reconstruction metadata and can be rerun without
editing the original data. Detailed authored meshes remain separate inputs;
the batcher does not pretend to regenerate their manually designed furniture.

## Retouch a completed export without rerunning models

`tools/room3d_retouch.py` edits the groups in a completed room's `assisted.obj`.
It copies materials/textures, preserves camera cutaways, saves the corrections,
and renders original/automatic/retouched comparisons for every original camera.
The source folder is read-only; each variant must use a new output directory
outside that folder. No inference or geometry fitting runs during this operation.

```sh
build/room3d-vision-venv/bin/python tools/room3d_retouch.py \
  --room build/room3d-all-assisted/room2010 --list
build/room3d-vision-venv/bin/python tools/room3d_retouch.py \
  --room build/room3d-all-assisted/room2010 \
  --edits edits-room2010.json --out build/retouches/room2010-v1
```

The edits file maps existing group names (from `--list`) to operations:

```json
{
  "primitive_1": {
    "translate": [200, -100, 0],
    "scale": [1, 0.9, 1],
    "yaw": 15
  },
  "boundary_6": {"hide": true}
}
```

This is syntax, not a recommended correction for ROOM201. Coordinates are game
world units in X/Y/Z order; negative Y moves upward. Yaw is in degrees, positive
rotating +X toward +Z. Scale then yaw operate around the group's bounding-box
center unless `pivot: [x,y,z]` is supplied; translation follows. Use multiple
group entries when one object spans multiple groups, with the same explicit
pivot when rotating/scaling them as a unit. Only positive scales are accepted.

Open the variant's `index.html` to review it. Adjust the edits JSON and generate
another variant from the same automatic source to avoid accumulating transforms.
The result is an OBJ plus its dependencies; importing it into the running game
and checking navigation remain separate validation steps. Textures stay attached
to moved geometry, so existing baked lighting moves too. Hidden or moved walls
can expose missing geometry. This tool does not fill holes, rebuild topology,
change gameplay collisions. Additional parametric objects can be inserted as
described below; this does not automatically recover arbitrary missing surfaces.

#### Insert an estimated object without rerunning models

`--additions objects.json` accepts a list of up to 32 reusable objects: `staircase`,
`railing`, `lamp`, or `column`. Each needs a unique `name`, `parameters` using the
same schema as `room3d_shapes.generate`, and an `origin` in game coordinates.
Optional `yaw` uses **radians**, as in automatic shape recipes. Transform edits
still use degrees. Inserted objects receive the group name `added_NAME`.

```json
[
  {
    "name": "stairs",
    "kind": "staircase",
    "origin": [0, 0, 2000],
    "yaw": 0,
    "parameters": {"width": 600, "run": 1000, "rise": 500, "steps": 4}
  }
]
```

The coordinates above illustrate syntax only. Choose a pose for the actual room,
or copy a shape proposal's parameters/origin/yaw. Generate an independent variant:

```sh
build/room3d-vision-venv/bin/python tools/room3d_retouch.py \
  --room build/room3d-all-generated/room2040 \
  --additions objects.json --out build/retouches/room2040-added
build/room3d-vision-venv/bin/python tools/room3d_editor.py \
  --room build/retouches/room2040-added --out build/room2040-added-editor.html
```

Use the editor to move, scale, rotate or hide the new group. `--edits` can also
transform the added group during insertion. Source meshes, UVs and dependencies
remain intact. Additions use a neutral material unless `--reproject` is supplied;
reprojection accepts the same existing SAM/DA3 folders as ordinary retouches.
New objects are recorded as estimated, unverified geometry in `retouch.json`.
When existing segmentation is supplied, added lamps can automatically associate
with a confident lamp mask using their visible silhouette. Scene depth excludes
occluded geometry; association needs at least 16 visible pixels, detector score
0.3, mask confidence 0.7, silhouette IoU 0.35, and a 0.1 margin over both competing
masks and competing objects. Ambiguous matches abstain. `reprojection.json`
records the selected camera/mask and overlap. The palette comes only from those
associated masks, rather than another lamp elsewhere in the room. Table lamps,
wall sconces and multi-light wall fixtures preserve separate shade/support
material roles. These are observed baked colors, not recovered physical light
emission or albedo. Unsupported associations retain the existing inferred or
neutral fallback; association alone does not certify object identity or shape.
Neither inserting an object nor reprojection validates room fidelity, attachment,
headroom, navigation or game integration. This is a finishing tool, not a substitute
for automatic reconstruction. The editor cannot yet create new objects directly.

After moving or hiding an object, `--reproject` recalculates appearance on the
edited geometry using the original backgrounds. Existing SAM/DA3 room directories
can preserve the original semantic/depth texture guards without rerunning models:

```sh
build/room3d-vision-venv/bin/python tools/room3d_retouch.py \
  --room build/room3d-all-generated/room1000 \
  --edits edits-room1000.json --out build/retouches/room1000-reprojected \
  --reproject --segmentation build/room3d-all-generated/segmentation/room1000 \
  --depth build/room3d-all-generated/depth/room1000
```

This can update texture visibility after edits, but does not guarantee filling
gray areas: absent/wrong geometry, incorrect semantic masks, or insufficient
source visibility remain limitations. `reprojection.json` records the bake report.
Always start from the automatic source room when comparing variants.

For a local visual interface, generate a self-contained HTML editor:

```sh
build/room3d-vision-venv/bin/python tools/room3d_editor.py \
  --room build/room3d-all-assisted/room2010 --out build/editor-room2010.html
```

Open that file in a browser, select a group and camera, and change position,
scale, pivot, yaw or visibility using the controls. The cyan wireframe updates
over the original background; it is not a textured or occlusion-correct preview.
The file embeds its geometry and images and needs no server or network access.
Save the corrections JSON, then apply it with `room3d_retouch.py` as above for
the textured comparisons. Previously saved corrections can be loaded again.
The editor uses numeric controls rather than direct dragging of objects.

Check the editor math and real browser controls with:

```sh
node tests/test_room3d_editor.js --browser build/editor-room2010.html
```

The browser check requires `google-chrome` and uses an isolated temporary profile.

## Catalog for a full-room run

Generate a combined room list with comparison/editor links as exports complete:

```sh
build/room3d-vision-venv/bin/python tools/room3d_catalog.py \
  --assets assets/USA --batch build/room3d-all-generated \
  --out build/room3d-catalog --reviews build/room3d-all-reviews --watch-seconds 30
```

Open `build/room3d-catalog/index.html` and refresh it for updated counts. Omit
`--watch-seconds` for one update. The catalog output must be separate from the
batch and assets. Completed rooms get an offline editor automatically. Counts
distinguish empty slots, pending/partial exports and image-assisted exports.
Export completion requires a matching RDT fingerprint, checkpoint and key output
artifacts; it does not establish fidelity or runtime traversal. Partial output
files do not establish that their generating process is still running.

Creative replacements, accepted single-view details and rejected proposals are
listed to prioritize manual inspection. These are review signals, not automatic
error classifications. External visual-review reports are counted only when their
input hashes match the queued images, and remain unverified model proposals.

## Stage a multi-room asset bundle

Do not flatten each room's `assisted.mtl` and `source-cameraN.png` into the same
directory: filenames repeat across rooms. Use the packager to give dependencies
room-specific paths compatible with the current `Room3D.cpp` loader:

```sh
build/room3d-vision-venv/bin/python tools/package_room3d.py \
  --batch build/room3d-all-generated --out build/room3d-mod-bundle
```

Only completed image-assisted checkpoints are selected. `--rooms ROOM1000
ROOM1010` limits a bundle. Use `--override room1000=build/retouches/room1000-v1`
to package a manual variant in place of that room's automatic output. Each call
requires a new destination and stages `StageN/roomNNNN.obj`, its camera sidecar,
and a per-room material/texture directory. It never installs into a live game.
The manifest records packaged rooms, failures and file hashes; an error returns a
nonzero exit status while other rooms can still be packaged.

The packager checks current runtime size/count limits and PNG dependencies.
This is not full runtime or gameplay validation. Offline equivalence checks
confirmed identical RGB/coverage/depth for selected views of the first three
exports after packaging; other rooms and in-game traversal remain unverified.

## Initial evidence

On the supplied USA assets:

- 348 RDT files: 320 surveyed and exported as provisional meshes, 28 empty
  slots. These counts include character variants and revisits, not 320 unique
  physical rooms.
- 1684 camera entries resolved to existing background files after stage folding.
  Inventory mode checks existence, not successful decoding of every image.
- A separate full decode audit (2026-10-11) validated all 620 distinct referenced
  background files covering those 1684 entries, with zero decoding errors.
  RDT hashes were checked against the inventory first. Evidence lives in
  `build/room3d-background-validation.json`; this checks inputs, not reconstruction
  quality or completion of the image-assisted batch.
- ROOM1180 and ROOM10B0: six backgrounds decoded and six offline comparisons
  rendered successfully. Both ROOM118 views include the existing authored
  reconstruction. The source files and the installed playable build were not
  modified by these runs.
- Five synthetic tests pass: projection basis/unsupported roll, depth ordering,
  near clipping and camera hiding, recipe/collision handling, and revisit/empty
  slot handling. Ruff checks and formatting pass for the new Python files.

Visual inspection shows useful corridor/floor volume alignment, but broad
featureless walls, absent ceiling areas, missing furniture and no reconstructed
stairs. ROOM118's authored result contains substantially more scene detail.
This experiment supports using batching for surveying, initial geometry and
comparison. It does not demonstrate unattended faithful reconstruction. The
next useful extension is multi-view surface fitting with persisted landmarks
and explicit height estimates, followed by material reconstruction and in-game
validation; simply applying the current extrusion to more rooms will not close
that fidelity gap.

## Image-assisted pipeline (experimental)

The baseline above remains available. `--reconstruct` adds image evidence and
procedural scene proposals; `--local-vision` runs a local detector automatically.
It downloads public model weights but does not upload the input images.

Dependencies are isolated. The tested detector uses CPU PyTorch and
`transformers==4.57.1`; the newer Transformers configuration loader was
incompatible with an older model configuration during setup. The model is
pinned to revision `a2bb814dd30d776dcf7e30523b00659f4f141c71` of
[`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71).
The API follows the [official Transformers documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino).
No custom model repository code is executed (`trust_remote_code=False`) and
weights are loaded as safetensors.

```sh
uv venv --python /usr/bin/python3 build/room3d-vision-venv
uv pip install --python build/room3d-vision-venv/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python build/room3d-vision-venv/bin/python \
  transformers==4.57.1 opencv-python-headless

build/room3d-vision-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-assisted-comparison \
  --rooms ROOM1180 ROOM10B0 --authored tests/fixtures/room3d \
  --reconstruct --local-vision \
  --runtime-exe build/windows-cross/residentevil.exe
```

An existing detection set can be supplied with `--detections <directory>`
instead of `--local-vision`. Pixel hashes must match the current original
backgrounds. `--vision-python` selects a separate detector environment when
running the lighter geometry environment. The baseline still needs only NumPy
and Pillow; reconstruction and its tests also need OpenCV. Model weights and
all generated outputs remain outside tracked source files.

### Five stages and their actual scope

1. **Visual proposals:** the local detector labels walls, floor, ceiling,
   doors, furniture and other requested categories. It can miss objects or
   produce false labels: ROOM118's door panel also produced window/chest
   proposals. Detection scores are not correctness probabilities. Annotated
   images are exposed in the report for review.
2. **Dimensions:** mutual SIFT matches are triangulated with the known cameras;
   positive depth, reprojection and ray-angle gates reject inconsistent points.
   Supported horizontal point clusters can propose obstacle heights. Detector
   bounds can estimate a script-anchored object's top Y. A rectangular-room
   ceiling hypothesis is fitted against at least two views; split floor plans
   are rejected. Explicit recipe dimensions retain priority.
3. **Geometry:** original init-script door, item-box and typewriter declarations
   supply semantic anchors. Closed door panels, chest bodies and typewriter
   tables/machines are generated procedurally with recorded estimated dimensions.
   Furniture whose estimated footprint overlaps a wall inherits that wall’s
   camera cutaway; this remains a placement heuristic. Each camera contributes
   at most one detection to an object-height fit.
   Accepted ceiling proposals add a roof. Conditional anchors are deferred and
   unsupported script opcodes stop decoding; the tool never scans arbitrary
   later bytes as commands. Moving game models remain game-owned. Arbitrary
   chairs, lamps, windows and staircases are **not yet automatically modeled**
   from their detector labels; those labels remain proposals.
4. **Appearance:** source-camera textures are assigned to subdivided triangles
   only after depth/visibility checks. Untextured corners retain sampled vertex
   color or neutral gray. This transfers the original baked illumination along
   with color; it does not recover independent albedo, lights or physical
   materials. Incorrect surfaces can still acquire the image of an object that
   ought to be separate geometry. Excluding the evaluation camera from texture
   sampling exposes lack of support from other views; it is not a fully held-out
   geometry benchmark because geometry fitting may use that camera.
5. **Runtime:** the optional Wine probe uses an isolated copied executable,
   generated mod and save directory. It waits for actual log observations before
   proceeding through the debug menu, then tests mesh loading, a camera change
   while walking, and menu open/close at the ROOM118 chest. It neither sends
   desktop-wide keys nor requires focusing another agent's window. No item
   transfer or save assertion is included. Other rooms explicitly report
   `not_run` until they have a validated interaction scenario.

Runtime support requires rebuilding the executable. `RE1_TEST_KEYS_FILE` polls
an opt-in command file at most 20 times/second and accepts up to 128 timed key
entries; a file change resets their relative clock. Scripted processes ignore
physical keyboard input. `RE1_ROOM3D_TRACE` emits camera/menu/transition state
changes. Both hooks are inactive for ordinary runs. Each probe records hashes
of the copied OBJ and executable, a fresh log and a separate result file.
A failed probe is not converted into a pass because the process stayed alive.

### Evidence and remaining limits

The first image-assisted run found zero acceptable sparse feature matches in
ROOM118 and one in ROOM10B, so no multiview obstacle-height proposals were
accepted. Local detector anchors nevertheless estimated ROOM118's door top
near -3382, chest top near -1551 and table top near -1599. The rectangular
ceiling hypothesis chose height 5800 with mean bounding-edge discrepancy about
11.4 pixels over two views. These are estimates, not recovered ground truth.
ROOM10B's ceiling hypothesis was rejected because an interior wall splits the
rectangle. Appearance-only improvement must not be mistaken for correct 3D.

The initial fixed-time runtime driver failed to reach the requested room and
was recorded as failed. The feedback-driven driver subsequently loaded ROOM118,
observed both cameras, and opened and closed the chest menu successfully; the
user also confirmed seeing the chest open. That is evidence for this specific
interaction, not for every room or every action.

### Review sequence

Keep this work in dependency-ordered review units, not one oversized PR:

- Geometry, appearance and comparison: batcher, preview, reconstruction,
  scene helpers, focused tests and this documentation (six files).
- Optional local detector integration: detector, batcher, tests and documentation.
- Wine feedback probe: Windows platform input, debug-menu observation,
  room-renderer observation, runtime runner, tests and documentation (six files).

The five stages now have executable paths, but the remaining object categories,
stair geometry and room-specific gameplay scenarios above are deliberately
reported as incomplete. There is no unattended fidelity-completion claim.

### Local segmentation stage (in integration)

`tools/room3d_segment.py` runs pinned `facebook/sam2.1-hiera-tiny` weights
(revision `de431c4043854a71d8101e17995dfe596bf101a5`) on CPU, with DINO
boxes as prompts. It verifies RGB hashes before inference, rejects invalid
boxes, preserves overlapping per-object masks, and records model revision,
detection hash, predicted mask IoU and inference time. Predicted IoU is a model
estimate, not a measured comparison with ground truth. No images are uploaded.

```sh
build/room3d-vision-venv/bin/python tools/room3d_segment.py \
  build/room3d-assisted-comparison/room1180/camera*-original.png \
  --detections build/room3d-assisted-comparison/detections/room1180 \
  --out build/room3d-segmentation/room1180
```

Initial ROOM118 inspection shows usable chest/floor/ceiling/wall masks, about
3.3 seconds per view after loading on this CPU. False DINO labels remain false
labels even when SAM finds a coherent silhouette. The current upstream
checkpoint declares `sam2_video`; Transformers 4.57.1 warns when loading its
image model with `Sam2Model`, but inference completes. Model integration must
continue to check compatibility rather than suppress this warning.

Pass `--segmentation build/room3d-segmentation` to the reconstruction batcher
alongside cached detections. Confident incompatible masks veto texture samples;
weak/unknown regions retain the previous behavior. Primitive OBJ groups now
retain their own identities and inherit parent-wall cutaways separately.
The first run rejected 36 camera/vertex samples in ROOM118 and 486 in ROOM10B;
these counts measure filter activity, not a fidelity improvement. Masks do not yet constrain mesh fitting. Later sections describe the depth
refinement and partial material completion now implemented; full geometry
completion and reliable visual review remain outstanding.
Reference API: https://huggingface.co/docs/transformers/model_doc/sam2


### Pose-conditioned depth stage (in integration)

`tools/room3d_depth.py` runs `depth-anything/DA3-SMALL`, pinned model revision
`e08cab65ca0ec38e7826075418411ab90cab4da3`, on CPU. The upstream source used is
`ByteDance-Seed/Depth-Anything-3` commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. It passes the actual game cameras as
world-to-camera matrices, preserves native 320x240 input, and stores output
intrinsics at the model's 336x252 resolution. Depth is aligned to the supplied
camera baselines, not assumed to be meters. RGB/survey hashes identify inputs.

```sh
build/room3d-vision-venv/bin/python tools/room3d_depth.py \
  --survey build/room3d-assisted-comparison/room1180/survey.json \
  --images build/room3d-assisted-comparison/room1180 \
  --out build/room3d-depth/room1180
```

The first two-view CPU inference completed in 1.89 seconds after model loading.
Depth and confidence arrays, grayscale previews and camera metadata are saved.
These are unverified proposals; `--depth` consumes them in the bounded
refinement described below.
The tools environment installed the upstream package with `--no-deps` to avoid
replacing CPU PyTorch with CUDA packages. Required inference dependencies added:
`einops omegaconf evo e3nn trimesh imageio plyfile pillow-heif moviepy==1.0.3
addict pycolmap`. Optional 3DGS exports are unused. Reference:
https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md

### Reversible refinement and inferred materials (in integration)

`--depth build/room3d-depth` verifies depth input hashes and searches bounded
primitive-height changes, retaining every candidate recipe, OBJ and per-camera
render under `corrections/`. A fixed target confidence mask scores capped
absolute log-depth error. Missing rendered pixels are penalized, not ignored.
A candidate must improve the mean by more than 0.0005 without worsening any
view by more than 0.005. The metric measures agreement with DA3 estimates, not
accuracy against unknown ground truth. `corrections/index.html` compares the
before/after states and links every rejected/accepted recipe. The initial
ROOM118/ROOM10B search rejected all ten tested changes and preserved its inputs.
Search currently adjusts only existing primitive heights; object generation,
width/depth/orientation fitting and a richer multiview objective remain pending.

`room3d_complete.py` supplies inferred local material patches for surfaces with
no observation. Patches come from squares strictly inside confident SAM masks,
with image borders included in the distance test. Excluded evaluation cameras
cannot supply completion patches. Entirely unobserved triangles receive a
repeated patch at an estimated scale; partially observed triangles fill unknown
vertex colors with the patch mean. The manifest records the source crop/camera
and inferred vertex count. This completes some appearance gaps, not missing
object geometry; unsupported kinds still use diagnostic gray.

`room3d_review.py` uses pinned local Qwen3-VL-2B-Instruct, selected for CPU trials
with approximately 17 GiB available system memory and unavailable NVIDIA driver.
It stores raw discrepancy proposals, prompt, input hashes, time and peak RSS;
it cannot approve corrections or claim gameplay correctness. Model revision:
`89644892e4d85e24eaac8bacfd4f463576704203`. This stage is still being tested and
has not yet been connected to the batch report/correction loop. Reference:
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Keep follow-on changes reviewable as separate dependency-ordered units:
segmentation/filtering; depth inference/refinement; hidden-material completion;
local review integration. No oversized PR has been opened.


The first Qwen 2B review completed in 40.1 seconds with peak RSS 4,845,012 KiB.
It incorrectly called the wooden floor a blue rug and made other unsupported
claims. This measured result is insufficient to authorize geometric corrections;
an identical-image negative control is being run before further integration.
The model is functional locally, but useful reviewer quality remains unproven.


The refined/completed ROOM118 mesh passed a fresh Wine probe (camera transition
and chest menu open/close), with its OBJ hash matched to the report. Evidence:
`build/room3d-refined-comparison/room1180/runtime-refined-02/result.json`.
The preceding probe was explicitly aborted after an incorrect asset-root
argument; its separate result records that error rather than claiming a pass.
ROOM10B still lacks a validated runtime interaction scenario.

The identical-image control completed in 39.8 seconds and invented four
nonexistent discrepancies. Qwen 2B with the current prompt therefore **fails
the reviewer acceptance check**. Its output must not drive corrections. The
next reviewer iteration must explicitly allow no differences and pass controls;
a larger local variant remains an option within measured memory limits.

### Mask/depth object proposals (in integration)

`room3d_objects.py` back-projects full object silhouettes through the known
camera and DA3 depth, estimates robust visible extents, then proposes barrel
volumes and lamp envelopes. Barrel hidden surfaces use a twelve-sided bulging
cylinder with top/bottom caps; lamp envelopes are still coarse boxes. Generic
box primitives now include bottom faces for views from below. The proposal
records its source camera, label, pixel support and globally confident fraction.
Whole silhouettes are needed because the global confidence threshold can retain
only an object's bottom edge and collapse its estimated height.

Proposals may replace overlapping non-wall *visual extrusions* of collision
bounds; original collision/script data remain unchanged. Each addition/replacement
passes the same reversible multiview gate. Initial ROOM118 accepted one barrel
replacement: per-view depth error changed from [0.17720, 0.11625] to
[0.08978, 0.11625]. This is agreement with estimated depth, not a percentage of
reconstruction fidelity. ROOM10B's coarse lamp proposal was rejected. Small
objects, staircases and unsupported hidden geometry remain incomplete.

Reviewer controls: the less leading two-image prompt correctly returned no
clear difference for identical inputs, but also missed an inserted red square.
A single labeled A/B contact sheet made 2B detect the square, while it still
invented additional lighting/floor differences. The 4B local variant is being
measured using the same controls (revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`); 2B remains unqualified to direct
corrections. Raw reports preserve every unsuccessful trial.

### Staircase and runtime follow-up

A staircase proposal now requires a labeled mask, at least 40 depth points
inside a previously unmodeled type-4 footprint, over 1000 units of horizontal
support, a height span over 500, and absolute height/run correlation at least
0.65. Type 4 alone is not treated as proof of stairs. Visible slope is fitted;
unseen run, width (up to 1500) and step rise (about 220) are geometric priors.
Closed stepped boxes complete hidden undersides. The current ROOM10B proposal
was accepted by the depth gate: camera 2 error 0.29868→0.28889, camera 3
0.33850→0.34108 (within the configured 0.005 tolerance). This is still visually
coarse and does not establish faithful stair dimensions.

A material bug found in image inspection matched `wall lamp` as `wall`, spreading
white bulb material onto hidden walls. Wall matching is now exact; a regression
test prevents this. Texture projection across object interiors still needs more
than triangle-corner support checks.

Runtime probes now accept ROOM10B navigation as a separate scenario (mesh load
and actual camera change); ROOM118 additionally requires chest-menu open/close.
Wrong asset roots are rejected before launch. `RE1_TEST_BACKGROUND=1` combined
with `RE1_TEST_KEYS_FILE` keeps scripted tests ticking after desktop focus loss,
normalizing activation only inside that process. Normal play retains focus-loss
pause. Opt-in trace also records player position changes for route diagnosis.
Earlier ROOM10B navigation attempts failed their camera-change assertion;
they are retained separately from the later successful run described below.

The 4B reviewer retry passed the visible-red-square control in 58.2 seconds,
peak RSS 9,231,712 KiB; it reported only the inserted square. The initial 4B
process exited with signal 15 before producing a report (cause unknown).
The subsequent identical-image control and review integration are described
below; this failed first attempt remains recorded.

### Integrated local pipeline and latest evidence

`--local-models --vision-python build/room3d-vision-venv/bin/python` now chains
DINO → SAM → DA3 → geometry/refinement/material completion → Qwen 4B for every
camera, all locally. It generates model artifacts beneath the output directory;
do not combine it with cached `--detections`, `--segmentation` or `--depth` roots.
The cached-stage path remains available for faster development. Full one-command
execution subsequently passed as recorded in the validated-result section.

`--review-python` adds a per-camera local review and hashed evidence to the HTML.
4B passed both synthetic controls: inserted red square detected without the 2B
floor/lighting inventions; identical images returned `NO_CLEAR_DIFFERENCE` in
64.1 seconds. This establishes only basic comparison ability, not expert-level
review accuracy. The six actual room reviews in
`build/room3d-reviewed-comparison` completed and remain unverified proposals.
In the subsequent local-pipeline run, the ROOM10B camera-2 review attributed
distortion to a changed camera angle even though the renderer uses the original
camera record. Such causal claims are not accepted as calibration evidence.
Model prose must not override measured cameras or gameplay data.

Texture filtering now creates per-camera/per-material source textures. Confident
incompatible pixels are filled with inferred material, preventing an object in
a triangle's interior from leaking through checks limited to its corners.
The manifest records filtered-pixel counts; a regression test places a red
object inside a green wall triangle and verifies the center remains green.

ROOM10B navigation finally passed with the generated staircase mesh. The
harness failed to recognize `room10B0.obj` because it matched lowercase only;
matching is now case-insensitive, with a regression test. Motion starts after
the room transition finishes, and the route walks forward to camera 1. Evidence:
`build/room3d-stairs-comparison/room10b0/runtime-navigation-06/result.json`.
Previous attempts remain failed, rather than being relabeled.

A fallback roof proposal completes unobserved upper coverage when there is no
accepted ceiling. Its height follows existing walls and any inferred upper
stair level; outside original cameras receive a cutaway. Raising all walls to
that height worsened camera 2 and was rejected; roof-only completion was accepted
with unchanged depth scores in all four ROOM10B views. Original wall dimensions
stay unchanged. The proposal does not establish a fully watertight mansion.

Thin fixtures now use closed prisms sampled from their masks and estimated
depth, preserving empty cells rather than filling the entire bounding box.
Back thickness remains an explicit geometric prior. Object proposals can also
pass on visible silhouette IoU of at least 0.35, provided no view exceeds the
0.005 depth-regression bound relative to the current accepted state. Two passes
allow a previously occluded proposal to be reconsidered after another accepted
replacement. Every trial records its gate and scores.

`build/room3d-silhouette-comparison` accepted the ROOM10B lamp with IoU 0.5714
and two ROOM118 barrels with IoUs 0.7823 and 0.5921. These compare against SAM
estimates, not hand-verified masks. Inspection still shows coarse stair shape,
texture stretching and gaps near the upper walls. Novel viewpoints explicitly
disable original-camera cutaways and report coverage without treating it as
fidelity; sampled positions are screened against collision bounds, not a
validated player-navigation route.

The initial ROOM10B novel-view inspection rendered four sampled positions with
97.44–100% geometry coverage. Visual inspection still shows stretched wall
textures, a coarse lamp and an upper gap. Those coverage numbers must not be
reported as reconstruction accuracy. The current local-pipeline depth trial
changes camera 2's model-agreement error from 0.29868 to 0.28904, while camera 3
changes from 0.33850 to 0.34108; cameras 0 and 1 are unchanged. The latter
regression is within the documented per-trial tolerance, not an improvement.
# Diagnóstico de detalles en el catálogo

Para repetir el piloto arquitectónico con reglas fijas en otras habitaciones:

```sh
build/room3d-vision-venv/bin/python tools/probe_room3d_architecture.py \
  --batch build/room3d-all-generated --rooms room1070 room1080 \
  --out build/mi-piloto-arquitectura
```

La salida debe ser una carpeta nueva fuera del lote original. Comprueba los
checkpoints antes y después, usa los modelos locales ya descargados, registra
las categorías y hashes del código y detecta cambios de código durante la
prueba. Produce máscaras, hipótesis, ajustes compartidos y evidencia de oclusión
por vista, además de un HTML. Es un piloto de evaluación: no instala candidatos
en el juego ni declara aceptadas las modificaciones de habitaciones.

`room3d_architecture.projected_object_evidence` distingue píxeles sin evidencia,
oclusiones propuestas por la escena, primer plano observado, acuerdo de
profundidad y conflicto de profundidad. El acuerdo con una máscara asociada
se registra por separado. Una superposición que ignora oclusión no basta para
rechazar un objeto detrás de una barandilla: la profundidad fiable puede indicar
que la barandilla está delante. Los píxeles desconocidos u ocultos tampoco
validan la identidad del objeto. En ROOM106 cámara7, los 105 píxeles fiables
indican primer plano y los otros 338 no tienen profundidad fiable; por ello
esa vista es inconclusa, no una contradicción demostrada de la columna.

La familia procedural `column` genera un fuste de doce lados con base y
capitel cuadrados; acepta altura, diámetro del fuste y altura de ambas piezas.
`room3d_architecture.fit_column_axis` ajusta una sola malla a las dos máscaras
de una hipótesis de eje, minimizando el peor ajuste de silueta entre vistas.
No usa profundidad observada para elegir la forma ni acepta cambios en la sala:
las comprobaciones de profundidad, oclusión y vistas independientes se hacen
después. El piloto de ROOM106 alcanza IoU de 0.74/0.76 en las vistas usadas
para ajustar; no es una medida de fidelidad de toda la habitación.

`room3d_architecture.column_axis_hypotheses` prueba correspondencias mediante
el eje del fuste cuando las bases visibles no corresponden al mismo punto.
Intersecta planos verticales, exige error de eje de hasta 3 píxeles, alturas
compatibles y extensión cercana al suelo, y registra alternativas ambiguas.
La intersección de dos planos no demuestra identidad: las propuestas necesitan
ajuste de forma por renderizado y evaluación en vistas adicionales. En ROOM106
una pareja de 34 supera estas comprobaciones preliminares; no se aplica a la
malla automáticamente.

`room3d_architecture.column_foot_correspondences` compara bases de columnas
entre cámaras usando el suelo compartido. Exige reproyección recíproca de
3 píxeles, descarta máscaras truncadas verticalmente, excluye cámaras
reservadas y marca parejas ambiguas. Registra también los rechazos y sus
errores. La hipótesis de contacto con el suelo y la integridad de la máscara
no están demostradas: las parejas son propuestas y no modifican la malla.

El detector local admite `--labels` para experimentar con categorías sin
modificar las predeterminadas. Por ejemplo, después de los argumentos de
imagen y salida: `--labels 'stone column' pillar archway balcony wall staircase
railing`. Cada resultado registra `prompt_labels`; las máscaras SAM quedan
ligadas al hash del resultado de detección completo. Una etiqueta y una máscara
no demuestran por sí solas que un arco sea un espacio vacío ni autorizan quitar
geometría. La prueba de ROOM106 obtuvo propuestas visibles de columnas y arcos;
todavía falta asociarlas entre cámaras y comprobar su geometría.

`room3d_assemblies.corroborated_stair_assemblies` genera propuestas conjuntas
de escalera y descansillo a partir de un borde superior observado en al menos
tres cámaras. Usa posición y orientación del borde además de su altura;
descarta duplicados, excluye cámaras reservadas y limita la búsqueda a dos
bordes y 32 propuestas. Son candidatos experimentales: el descansillo añadido
no constituye evidencia independiente de apoyo estructural. Todavía no se
incorporan automáticamente al lote ni se consideran reconstrucción validada.

La opción experimental `constrain_stair_proportions: true` penaliza durante
el ajuste los candidatos de escalera que incumplen las proporciones del
generador: contrahuellas de 120–240 unidades, huellas de al menos 200 unidades
y pendiente de 15–42 grados. Evita que una silueta más parecida gane a costa
de peldaños desproporcionados. Son prioridades procedurales, no medidas
recuperadas del fondo. Conserva las comprobaciones de profundidad, apoyos y
aceptación; está desactivada por defecto hasta su evaluación en habitaciones.

Las comparaciones de apariencia de una habitación reutilizan las profundidades
de su malla y suelo. La caché es local a la habitación y se invalida por contenido
de la malla, archivo de visibilidad o parámetros de cámara. No almacena colores
ni calibraciones de profundidad observada, por lo que cada comparación sigue
excluyendo sus muestras reservadas. La comprobación de ROOM104 produjo 46
archivos idénticos y evitó volver a renderizar profundidad en una segunda pasada.

La opción `local_fixture_appearance: true` asigna a cada lámpara procedural
aceptada una paleta de sus propias máscaras observadas. Evita compartir
una muestra de otra lámpara. Requiere máscaras de al menos 16 píxeles y confianza
SAM de 0.7; registra además la confianza del detector, sin presentarla como una
probabilidad de fidelidad. En apliques y lámparas de mesa con partes paramétricas,
asigna la mediana del cuartil oscuro al soporte y la del 15% más luminoso a la
pantalla, solo si la diferencia de luminancia alcanza 0.15. Los objetos sin
contraste conservan su mediana uniforme. La asignación de colores a las partes
es una interpretación procedural, no una correspondencia recuperada. No recupera el
albedo ni añade iluminación dinámica. Las comparaciones con cámara excluida
también excluyen sus muestras de color. El reproyectado tras un retoque conserva
esta opción de la receta. Está desactivada por defecto.

La receta experimental `single_view_wall_fixtures: true`, junto con
`creative_fallback: true`, permite aceptar una lámpara estimada desde una sola
vista si tiene anclaje comprobado a una pared, al menos 16 píxeles visibles y
IoU de silueta de 0.6. Sigue exigiendo mejora del ajuste, consistencia de
profundidad en todas las cámaras y las demás comprobaciones habituales.
Queda registrada como interpretación procedural y con
`single_view_fixture_prior: true`; no acredita reconstrucción fiel ni validación
en una cámara reservada. Está desactivada por defecto y aún necesita evaluación
visual en habitaciones adicionales antes de incorporarse al lote general.

Cada habitación exportada muestra los rechazos por categoría y las comprobaciones
que fallaron, con un enlace al informe de detalles. Un objeto puede fallar varias
comprobaciones; sus conteos no se deben sumar como objetos distintos. Los rechazos
sin comprobación fallida registrada se indican por separado. Estos datos ayudan
a priorizar retoques, pero no son una puntuación de fidelidad ni una validación
de navegación dentro del juego.


La opción experimental `fixture_part_fit: true` añade un ajuste de brazo y altura
sobre la pared a los apliques paramétricos. Requiere las imágenes originales
(el batcher las entrega automáticamente). Reserva la última observación cuando
hay tres o más cámaras; esa imagen no define el objetivo de brillo. Descarta
máscaras sin contraste, reflejos fragmentados y regiones claras recortadas por
el borde. La región clara se considera una hipótesis de pantalla, no prueba de
identidad de una bombilla. Los candidatos mantienen contacto con la pared,
silueta mínima 0.35 (0.60 en apliques de una sola vista cuando se usa
`single_view_wall_fixtures`) y no pierden más de 0.15 de IoU por vista de entrenamiento;
se exige mejorar al menos 0.05 la coincidencia de cada región clara utilizada.
Los controles finales de escena, profundidad y soporte siguen siendo necesarios.
El ajuste puede abstenerse y conservar la geometría previa. No está activado en
las recetas del lote completo; la evaluación visual y en habitaciones nuevas
sigue pendiente antes de adoptarlo por defecto.

La opción experimental `raised_support_occlusion: true` amplía la búsqueda visual
sobre las dos mejores alternativas con apoyo elevado, antes de elegir el apoyo final. Requiere `creative_fallback` y
reutiliza las propuestas existentes de menor altura para superficies provisionales
que contradicen profundidad o estructura repetida de peldaños. Mantiene fija cada
escalera y su apoyo; evalúa como máximo dos alturas por alternativa (cuatro en total),
sin ramificar de nuevo.
Solo las vistas de entrenamiento eligen la propuesta; el batcher conserva los
controles finales de profundidad en todas las cámaras. Cambia geometría visual,
no las colisiones originales. Permanece desactivada por defecto y pendiente de
validación completa en habitaciones nuevas.

`semantic_stair_guard: true` es una protección experimental contra paneles de
puerta confundidos con peldaños. Descarta una propuesta de escalera solo si al
menos el 90% de su máscara está dentro de otra máscara etiquetada exactamente
como puerta, con confianza del detector >=0.5, SAM >=0.9 y ventaja de confianza
>=0.15. Exige al menos16 píxeles; conserva etiquetas ambiguas con barandilla y
conflictos débiles. Registra cada abstención en `semantic_rejections` del informe
y en su HTML. Estas confianzas son señales de los modelos, no probabilidades de
fidelidad. Está desactivada por defecto. No reemplaza la revisión de otras
confusiones (estanterías, alfombras, mobiliario) ni los controles de geometría.

`feasible_stair_starts: true`, junto con `creative_fallback` y
`constrain_stair_proportions`, añade como máximo dos inicios procedurales para
escaleras: uno por familia de observación (máscara completa y región de
peldaños). Evita que una semilla demasiado plana o empinada quede descartada
antes de optimizarse. Conserva base, orientación, anchura, altura y descansillo;
ajusta longitud y número de peldaños a proporciones plausibles. Puede mover el
extremo superior, por lo que no demuestra una conexión con otro piso. Mantiene
los inicios originales y no modifica hipótesis de nivel independientes ni
candidatos explícitos del llamador. Está desactivado por defecto y no omite
controles posteriores. El informe identifica `feasible_mask_depth` o
`feasible_tread_region` y la forma resultante conserva su procedencia estimada.

### Revisión local de partes: propuestas sin autoridad geométrica

`tools/room3d_part_review.py` registra propuestas de Qwen3-VL-4B sobre las
regiones etiquetadas como escalera. Usa el fondo completo con el contorno de la
máscara en cian; distingue peldaños visibles, barandilla, paneles de puerta,
estantería, suelo/ornamento, otro e incierto. La revisión no cambia etiquetas,
no elige generadores y no aprueba objetos. El piloto de cuatro casos produjo
dos aciertos y dos errores; no demuestra una precisión generalizable.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  build/room3d-vision-venv/bin/python tools/room3d_part_review.py \
  --batch build/room3d-all-generated --rooms room1020 \
  --out build/stair-part-reviews-source
```

Requiere los pesos fijados por revisión disponibles en la caché local. La
inferencia se ejecuta en CPU; las imágenes no se envían a un servicio. El
directorio de salida debe estar fuera del lote de entrada. Aplica las máscaras
refinadas de peldaños cuando existen. `--raw-masks` solo sirve para diagnosticar
las máscaras originales; una revisión de máscara original no sirve para otra
refinada.

La opción de receta `stair_part_reviews` apunta al directorio de revisiones y
añade las propuestas al informe de ajuste y su HTML. Exige imágenes originales.
Comprueba habitación, cámara, nombre de máscara, hashes de RGB y máscara, modelo,
revisión de pesos y prompt. Una revisión existente incompatible provoca error;
una revisión ausente se omite. Las entradas idénticas se reutilizan sin cargar
el modelo. `semantic_stair_guard` y los controles geométricos siguen siendo
independientes, incluso cuando contradicen la respuesta del revisor.

### Escaleras con categoría específica y fondo delgado

El refinador reconoce las categorías exactas `staircase`, `carpeted staircase`,
`wooden staircase` y `stone staircase`, conservando el texto del detector. No
convierte etiquetas mezcladas como `staircase railing` o `stair treads staircase`
en una escalera. Los registros nuevos incluyen `source_label` y rechazan su
reutilización con otra etiqueta; los históricos sin ese campo conservan sus
validaciones de imagen, máscara y profundidad. Esto admite prompts específicos,
pero no activa automáticamente una segunda pasada del detector.

`suspended_stair_starts: true`, junto con `creative_fallback`, añade una
alternativa con fondo delgado por familia de inicialización observada, hasta
cuatro adicionales. Mantiene los inicios macizos y no modifica candidatos
explícitos del llamador ni hipótesis independientes de nivel. La forma usa
`structure: suspended` y `slab_depth: 100`; admite espesores positivos hasta
240 unidades. Conserva las posiciones de peldaños y descansillo, cerrando cada
peldaño como un prisma que se solapa con el siguiente sin rellenarlo hasta el
suelo. Las superficies inferiores y la estructura portante son estimadas.
No demuestra resistencia, apoyo superior ni recorrido en el juego. Está
desactivada por defecto y sigue sujeta a los controles de ajuste y de escena.

El presupuesto de optimización permite hasta cuatro inicios por estructura,
de modo que cuatro candidatos macizos mejor puntuados inicialmente no excluyan
la alternativa delgada antes de optimizarla. El informe `structure_search`
registra los índices/estructuras optimizados y la estructura seleccionada cuando
está activa esta opción. Las estructuras explícitas también se distinguen aunque
compartan el nombre de inicialización. Esto amplía la búsqueda; no garantiza que
se seleccione la variante delgada ni que una escalera pase los controles finales.


### Ajuste de profundidad y altura libre por etapas

`scene_depth_fit: true` añade durante la búsqueda una penalización por empeorar
la profundidad de cualquier cámara de la habitación, salvo la vista del objeto
reservada para evaluación. La penalización usa la geometría fija de la escena y
la profundidad estimada; no convierte esos datos en medidas reales. Se mantiene
el control final independiente de regresión en todas las cámaras.

`scene_depth_polish: true` resuelve primero forma, apoyo y techo con el ajuste de
máscaras habitual. Después refina esa forma con profundidad de escena y peso
adicional para la peor vista. No vuelve a inventar la topología en esta etapa.
Activa automáticamente el control de obstáculos conocidos sobre los peldaños
(`constrain_stair_headroom`). El control también se aplica al resultado final y
a las alternativas creativas. La falta de techo observado se registra como
altura libre desconocida, no como prueba de que se pueda recorrer la escalera.

Estas opciones son experimentales y están desactivadas por defecto. El informe
`depth_polish.initial_report` conserva el ajuste previo; los parámetros y las
puntuaciones principales corresponden al refinamiento. El techo se selecciona
antes de pulir la forma, por lo que el buscador debe respetar sus límites.
Las comparaciones iniciales de ROOM106 y ROOM10B que pasaron controles visuales
pero tenían obstáculos de techo son diagnósticos históricos; no son ensamblajes
utilizables. Los ensayos corregidos se guardan en directorios separados.


`foreground_fixture_guard: true` añade un veto opcional de textura para máscaras
explícitas de lámparas/candelabros y barandillas con score del detector>=0.3 y
confianza SAM>=0.8. Impide proyectarlas sobre otra categoría, incluidas superficies
sin clasificar. Conserva las muestras de la propia categoría; no convierte esa
confianza en permiso para generar geometría ni elegir materiales. Las etiquetas
ambiguas o débiles conservan el comportamiento previo. Se aplica tanto al bake
normal como al de vista excluida, y el informe registra los umbrales. No recupera
la textura oculta: usa las continuaciones/colores estimados del proceso existente.
Permanece desactivado por defecto y pertenece a una revisión separada del ajuste
de escaleras y altura libre.
# Descansillos creativos acotados

La receta opcional `creative_stair_landings: true`, junto con
`creative_fallback: true`, activa `room3d_landing_fit.fit_landing` después de
aceptar una escalera. Conserva el tramo ajustado y prueba descansillos de 400,
800, 1200 y 2000 unidades. Elige el mayor que pasa las vistas de entrenamiento;
después valida una sola vez las cámaras reservadas, sin volver a seleccionar
otro candidato si falla esa evaluación.

Requiere contacto con superficies designadas transitables en ambos extremos,
altura libre conocida en las muestras, puertas despejadas, al menos 90% de los
píxeles visibles previos de la escalera y degradación de profundidad <=0.005.
En la integración, esa tolerancia se mide respecto a la escena anterior al
ajuste del objeto, evitando acumular dos márgenes. También se conservan las
condiciones originales del ajuste de la escalera y se actualizan las imágenes,
profundidad y métricas finales. La ausencia de techo observado no cuenta como
altura libre certificada. El informe `landing_assembly` distingue contacto
geométrico de superficie designada transitable; un apoyo sobre una extrusión de
colisión sin esa designación provoca abstención.

El descansillo sigue siendo una propuesta visual estimada. No recupera todo el
piso superior, no demuestra soporte estructural y no modifica las colisiones ni
la lógica de movimiento del juego. La opción está desactivada por defecto y no
se ha activado en el lote general que ya estaba ejecutándose.

Comparación aislada: `build/room3d-compact-landing-fitter/index.html`.
ROOM106 selecciona 400 unidades; ROOM10B se abstiene por la base no designada
transitable. La versión texturizada de ROOM106 está en
`build/room106-compact-landing-textured/index.html`. Son pruebas de aproximación,
no una certificación de fidelidad o de recorrido en el juego.

# Evaluación de un lote todavía en ejecución

El evaluador acepta checkpoints terminados sin esperar al manifiesto final:

```sh
build/room3d-vision-venv/bin/python tools/evaluate_room3d_details.py \
  --run build/room3d-all-generated --completed-only \
  --out build/room3d-live-detail-evaluation
```

Verifica las huellas de `details/results.json` y de `assisted.json` cuando forma
parte del checkpoint. Excluye entradas parciales, con errores o con esos archivos
modificados; no certifica todos los archivos del export. Resume aceptados,
rechazados y criterios fallidos por categoría. Los criterios fallidos de una
reconstrucción descartada no se cuentan como rechazo de su reemplazo creativo
aceptado.

La salida agrupa variantes por nombre base de habitación y mantiene sus entradas
individuales. No son pruebas independientes ni garantiza que tengan geometría
idéntica. Si no hay fuentes congeladas verificables, muestra
`source_execution_snapshot: not_recorded`: el código actual del disco no prueba
qué versión importaron los workers al comenzar. Tampoco convierte coincidencia
de máscara, aceptación o exportación en un porcentaje de fidelidad.

# Placas de lámpara junto al borde de una pared

`inset_fixture_anchors: true` añade propuestas acotadas cuando una placa de
aplique queda parcialmente fuera de su pared. Prueba desplazamientos de media
altura de placa más 10 unidades y media anchura más 10, dentro del mismo plano;
conserva la propuesta válida más cercana. No extiende la pared ni usa una altura
nueva inferida de la imagen. Registra el punto original y el desplazamiento.

Junto con `creative_fallback` y `single_view_wall_fixtures`, la selección de una
sola vista prioriza soluciones visibles con contacto válido sobre una silueta
flotante. El umbral final de IoU 0.6, los controles de profundidad, puertas y
contacto permanecen vigentes. Si no hay solución apoyada, no se certifica una
geometría recuperada. La opción está desactivada por defecto.

La transferencia aislada `build/room3d-fixture-insets-transfer-203-204` consiguió
contacto válido, pero ambos objetos siguen rechazados por ajuste visual
insuficiente. En sus máscaras, los mayores grupos de píxeles brillantes ocupan
50% y 57.1% del brillo seleccionado: el refinamiento de una sola pantalla se
abstiene correctamente. Hará falta modelar y ajustar varias luces por aplique;
activar esta opción por sí sola no resuelve ese detalle.

# Apliques con varias luces

`multilight_wall_fixtures: true` añade propuestas `wall_candelabra` de dos y tres
luces sobre hasta cuatro apoyos de pared ya válidos. El generador admite de dos
a cuatro luces, con separación lateral `span`, una placa común y brazos curvos.
Úsalo junto con `creative_fallback`, `fixture_part_fit`,
`single_view_wall_fixtures` e `inset_fixture_anchors` para evaluar el camino
creativo de una sola vista; `refine_fixture_anchor` permite ajustar el punto
estimado dentro de su vecindario sobre el plano de apoyo.

El ajuste de partes puede emparejar varias regiones brillantes con pantallas
distintas. Exige regiones contenidas, contraste y cobertura suficientes; pequeños
reflejos no se convierten en luces adicionales. Las imágenes reservadas no
entran en ese ajuste. Siguen vigentes los controles finales de máscara,
profundidad y contacto; no hay un porcentaje de fidelidad certificado.

Con `local_fixture_appearance: true`, todas las pantallas y todos los brazos usan
sus respectivos roles de paleta, obtenidos de las observaciones de esa lámpara.
No se simula iluminación dinámica ni se recupera albedo independiente del fondo.
La opción está desactivada por defecto. Comparación texturizada aislada:
`build/room204-multilight-textured/index.html`. ROOM204 acepta una propuesta
creativa de tres luces; ROOM203 sigue rechazado. La tercera luz de ROOM204 no
está demostrada por los dos grupos brillantes observados.


### Experimental combined detail profile

`--detail-profile creative-v2` applies a versioned combination of the detail
improvements through the normal batch CLI. It requires `--reconstruct` and is
experimental; it is not a claim that all room geometry or navigation is solved.
`existing` remains the default and preserves the recipe contents. Explicit
per-room recipe options override the profile, including explicit `false` values.
The effective profile/recipe participates in checkpoint identity; results record
`detail_profile` and assisted reports retain the effective fitted recipe.

```sh
build/room3d-vision-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/creative-v2-trial \
  --reconstruct --detail-profile creative-v2 --defer-review \
  --detections build/room3d-all-generated/detections \
  --segmentation build/room3d-all-generated/segmentation \
  --depth build/room3d-all-generated/depth --rooms ROOM2040
```

The profile enables feasible/suspended stair starts, proportion and headroom
constraints, staged whole-scene depth polishing, compact creative landings,
wall fixtures with finite-wall insets, multiple lights and part fitting, local
fixture palettes and foreground texture guards. Existing final acceptance
criteria remain in effect. Inspect both accepted and rejected trials, original
camera comparisons and newly exposed surfaces before adoption. In particular,
success with specially selected masks does not prove the automatic all-mask
pipeline works. Cached model roots must remain stable during a reproducible run;
freeze selected inputs and source code when comparing profiles. Use a separate
output directory to preserve existing exports.

### Focused stair recovery for missing camera views

Optional recipe flag `stair_prompt_recovery: true` runs a local focused DINO/SAM
retry before object proposals and stair refinement. It is not enabled by the
combined profile. At least one existing exact stair-category mask with detector
score 0.3, mask confidence 0.65 and 150 pixels must trigger recovery. Only missing
or weak views are retried, using a fixed room-independent prompt list: staircase,
carpeted staircase, railing, wooden floor. Mixed staircase/railing labels do not
trigger retries or become stair proposals. The recovery accepts at most two
nonduplicate proposals per view, requiring the existing repeated-tread-edge and
predicted-depth tests. It preserves detector/SAM confidence rather than increasing
it and retains all original masks. These proposals still require downstream
association, fitting and final geometry criteria.

Recovery uses cached local model weights with network access disabled in the
batch subprocess. It requires segmentation/depth inputs and the vision Python
runtime. Its merged segmentation and provenance live under a uniquely named
`stair-recovery/ROOM/RUN/` directory outside the source room; failed runs can be
retried without replacing old recovery artifacts. The result records the merged
mask path and input evidence. Resume still validates the completed room normally.
A standalone trial can be run without fitting geometry:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  build/room3d-vision-venv/bin/python tools/room3d_stair_recovery.py \
  --survey build/room3d-all-generated/room1060/survey.json \
  --images build/room3d-all-generated/room1060 \
  --segmentation build/room3d-all-generated/segmentation/room1060 \
  --depth build/room3d-all-generated/depth/room1060 \
  --out build/focused-stair-room1060
```

Choose a new output directory. A recovered mask is not proof of recovered
stairs, and no successful room reconstruction is claimed until the full fitting
and visual evaluation pass.

### Optional shade size and spacing refinement

Recipe flag `fixture_size_part_fit: true`, used with `fixture_part_fit`, adds a
bounded shade-size/spacing stage after arm/anchor fitting. It preserves the wall
anchor and the mean vertical position of each shade while trying smaller shades
and alternative spacing. Whole-object silhouette, wall contact and per-view part
improvement requirements still apply; outer whole-scene depth/acceptance gates
remain mandatory. Rejected candidates are reported, including attachment or
whole-silhouette regressions. Selection uses training views only. The option is
off by default and is not enabled in `creative-v2`.

A rendered synthetic test confirms a recoverable size error can improve without
using the reserved source image. The actual ROOM204 trial kept all three existing
fixtures unchanged: no candidate met the necessary whole-shape/part criteria (one
fixture lacked reliable bright-region evidence). This is not an adopted visual
improvement. Bright pixels alone cannot be treated as the full geometry of a
lampshade; preserving dark shade surfaces while modeling local bright regions
remains unresolved.

### Optional estimated bright cores on creative lamps

Recipe flag `fixture_core_appearance: true`, together with
`local_fixture_appearance`, replaces uniformly bright shade materials with a
small centered bright pattern. It estimates the pattern area from the lamp's own
verified mask brightness and visible projected shade area. Low contrast, tiny,
unobserved and excluded-camera samples abstain. The source geometry, support
materials and all other surfaces stay unchanged. The centered pattern is an
artistic estimate, not recovered emission or light geometry. This option remains
off by default and is not enabled in `creative-v2`.

The batch passes the option to both ordinary and heldout appearance baking.
`fixture_cores` in the bake report records training masks, area measurements,
estimated fractions and generated textures. Per-face charts are continuous
across triangles from a source quad. Geometry transforms without reprojection
preserve these UVs. Retouch reprojection now preserves existing baked lamp
shade/support materials, colors and UVs rather than estimating another bright
pattern on every previously subdivided face. It recognizes only the role names
emitted by this pipeline and only interpreted lamp groups. Newly added lamps
still use their own association/palette path. `preserved_fixture_materials` in
`reprojection.json` records the preserved group/material mapping. Preservation
cannot be enabled during heldout appearance evaluation because previously baked
textures may already contain the excluded view.

For a faithful appearance rebake, supply the same cached SAM/DA3 inputs used for
the original export. Without these inputs, other surfaces may receive different
projections. `build/room204-preserved-fixture-retouch-cached/index.html` compares
no-edit reprojection and a translated lamp using those caches. Both variants
match the expected textured geometry pixel-for-pixel in all five source views,
including unchanged shade patterns and support colors. The earlier no-cache
trial preserved the lamps but visibly changed floor/wall projection; it is not
the recommended finishing workflow. Geometry transforms without reprojection
remain available when an appearance rebake is unnecessary.

Controlled comparisons are in `build/room204-fixture-core-trial/index.html`,
`build/room1060-fixture-core-trial/index.html`,
`build/room3080-fixture-core-trial/index.html` and
`build/room4010-fixture-core-trial/index.html`. The same rule reduces solid bright
shades in the first three rooms, and makes no changes in ROOM401. Exact vertex,
coverage/depth, non-shade UV/material and visible-shade change checks are stored
in each `baked-material-verification.json`. These are appearance comparisons,
not a new successful whole-room reconstruction or a fidelity percentage.

### Checkpoint-bound visual findings outside exports

Keep visual review files outside completed room folders so they do not invalidate
their checkpoints. `room3d_visual_audit.py` registers explicit findings for views
actually inspected and binds them to the complete export, including its rendered
images and all texture/material dependencies:

```sh
build/room3d-vision-venv/bin/python tools/room3d_visual_audit.py \
  --room build/room106-tread-direction-trial/room1060 \
  --audits build/my-visual-audits \
  --status needs_further_geometry_work --camera 1 --camera 7 \
  --reviewer assistant_visual_review \
  --finding 'Incorrect flight dimensions and missing gallery geometry.'

build/room3d-vision-venv/bin/python tools/evaluate_room3d_details.py \
  --run build/room106-tread-direction-trial --completed-only \
  --audits build/my-visual-audits --out build/my-evaluation
```

Register a finding only after inspecting those exact views. A visual pass covers
the listed views, not hidden geometry, fidelity percentage or traversal. If any
export artifact or checkpoint changes, the review becomes `stale_or_invalid_audit`
and cannot approve the new version. New review directories retain earlier reviews
without overwriting them. Numerical acceptance stays distinct from visual review.

### Explicit horizontal layout prior for tread evidence

Experimental recipes can specify `"tread_horizontal_axes": [[1, 0], [0, 1]]`
alongside `"stair_refinement": true`. These are XZ tread-width axes, not ascent
directions. The refiner intersects each image line's backprojection plane with
a horizontal plane and keeps lines within 12 degrees of a supplied axis. No DA3
depth scale or confidence is changed. The same option is available directly as
repeated `--horizontal-axis X Z` arguments to `room3d_stair_edges.py`.

Supply axes only as an explicit layout assumption; diagonal flights may need
rotated axes or no prior. It is disabled by default and in existing profiles.
Refinement caches record normalized axes and reject reuse under a different
prior, including reuse without a previously applied prior. This filter removes
some incompatible edges; it does not prove that retained edges are treads.
The local four-room comparison is at
`build/room3d-layout-tread-probe/index.html`: it rejects misleading ROOM106 lines,
retains real ROOM10B treads, but still admits a ROOM210 railing edge. No improved
ROOM106 geometry or full batch fidelity is claimed from that evidence.

### Successive retouches and reprojection

A new retouch variant retains `retouch-appearance.json` whenever its source has
appearance semantics. This carries material-category settings, fixture policy
and the semantic kinds of manually added groups. It does not describe current
primitive poses: the edited `assisted.obj` is authoritative for geometry.

Use the previous variant as `--room` for another transform/hide operation,
including `--reproject`. Continue to supply the same original `--segmentation`
and `--depth` directories when using them; this avoids changing the appearance
policy between edits. Earlier variants remain untouched. Older variants that
lack this semantic context can still be transformed with their UVs preserved;
reprojection requires the initial export's semantics, and now fails before
creating a variant if those are unavailable.

This enables successive edits of existing groups, including an object added in
the first variant. Additional insertion steps reuse the canonical neutral
additions material library without duplicate references. Existing object names
remain reserved: duplicate names fail before creating an output. If that neutral
library has been manually customized, insertion refuses to overwrite it; use a
separate variant to preserve those material changes. Reprojection already bakes
materials into a new library and retains the semantic types of earlier additions.
Each variant's `retouch.json` records only that step's edits. Unit coverage checks
three sequential retouch/reprojection operations, unchanged source inventories,
exact retained lamp appearance after moving geometry, and classification/palette
preservation for a previously added lamp. Keep this separate from the tread-prior
change: four files (retouch helper, fixture/addition tests, this document).

The real cached ROOM204 check is at `build/room204-sequential-retouch/index.html`.
It runs two successive full retouch/reprojection calls; the second variant moves
one existing lamp by 150 units. In all five views, its rendered pixels exactly
match the expected transformed previous variant; coverage/depth also match and
all six fixture-role materials are preserved. Model caches, original room and
first variant remain unchanged. Camera0 was visually inspected. This confirms
appearance/geometry preservation for that sequence, not traversal or fidelity
of the underlying room. Six fixture-appearance, six addition and three retouch
tests pass, along with Ruff for the three Python files.

Successive insertion is a separate four-file change (addition helper, retouch
helper, addition tests, this document). Its tests add stairs in one variant,
insert a railing in the next, then transform/reproject the combined scene.
They check exact unchanged earlier vertices/materials, unique library references,
inherited semantic group types, duplicate-name rejection and preservation of
customized material files. Eight addition tests now pass; the underlying room's
navigation/collisions remain unchanged by visual retouches.

### Experimental creative staircase replacement

`"creative_stair_replacement": true` allows exact stair-category masks with
detector score at least 0.3, original SAM confidence 0.65–0.8 and at least 150
pixels to act as explicitly approximate design cues. Their confidence is never
increased and they do not become verified tread observations. Other categories
and merged labels keep the existing acceptance rules.
Masks already admitted through accepted repeated-tread evidence retain their
ordinary path instead of activating creative replacement.

For training groups containing such a cue, the fitter explores suspended flights
on `tread_horizontal_axes` (defaulting to an explicitly estimated orthogonal
design). It tries both ascent signs and two bounded widths, keeps each seed's
horizontal center/base height, regularizes step dimensions, and constrains width
to 4000 units and alignment to 12 degrees. Layout/proportion constraints remain
active during depth polish and final selection, including procedural fallback.
Ordinary support, consistency, depth and heldout gates still apply. No reserved
camera determines whether this prior is activated.

The option is off by default and absent from existing profiles. The full frozen
ROOM106 experiment completed in `build/room106-creative-replacement-trial`.
That experiment froze the initial candidate-generator implementation before
later reporting/final-fallback guards and the explicit proportion predicate;
its outputs alone cannot verify the final current pipeline. Source hashes retain
that distinction. No version is promoted or claimed visually improved yet.
ROOM106 still rejects every staircase candidate and lacks the flight. A later
full ROOM300 control rejects three misleading pavement/planter stair hypotheses,
but its overall courtyard reconstruction remains incomplete. The combined
checkpoint-verified visual report is
`build/room3d-creative-replacement-evaluation/index.html`; neither result is a
complete reconstruction or a measured fidelity percentage.

The experimental creative route now also initializes from shared camera/mask
volumes inside the existing room bounds. Only training views participate; no
DA3 values enter this localization. A 96×96 floor/height grid and a three-pixel
mask tolerance generate both-axis/sign candidates with regular proportions.
This path can produce starts even when depth-derived initialization yields none.
For these approximate designs, silhouette visibility uses actual rendered scene
occlusion; model depth remains in the residual and global safety checks rather
than censoring the silhouette itself. Existing/default fitting is unchanged.

The initial ROOM106 seed diagnostic produced 32 bounded candidates, but its best
scene-visible mask IoUs are only about 0.100/0.093 in training views1/6. It is not
an accepted reconstruction. Current-source full trials are running at
`build/room106-mask-volume-full-trial` and `build/room300-mask-volume-validation`;
their results are required before any promotion or visual improvement claim.


Experimental visual enclosure proposals
--------------------------------------
With `creative_stair_replacement` enabled, an estimated creative stair layout
observed in at least two training cameras may propose lower visual boundary
extrusions. A diagnostic must show at least 150 projected pixels, 60% overlap
with its training mask, at least 20% scene occlusion and 50 pixels attributed
to a known generated boundary. The proposal tries heights 750/1500/3000,
bounded to the three highest-voted groups. Source collision records, floor,
script-owned primitives and held-out observations are untouched. This is an
estimated architectural alternative, not evidence of the original wall height.
Ordinary fitting, contact, held-out and all-camera depth acceptance still apply;
existing/default paths retain their previous depth/tread-based proposals.

`build/room106-visual-enclosure-probe/index.html` records a frozen diagnostic:
reducing the three automatically attributed visual occluders raises the seed's
training silhouette IoUs from 0.100/0.093 to 0.354/0.247 at height750. All eight
cameras remain within the existing depth tolerance, but the gallery and upper
geometry are visibly incomplete. Textured comparisons are supplemental, use
the same estimated appearance policy and do not constitute adoption. The
ROOM300 mask-volume control completed; its checkpoint-bound visual audit
rejects overall fidelity despite avoiding an invented stair flight.


Experimental joint creative stair landing
----------------------------------------
`joint_creative_stair_landings: true`, together with
`creative_stair_landings: true`, allows the creative fallback to propose a
landing before rejecting an otherwise plausible flight for its missing top
connection. This option is off in every named profile and by default. The
existing generator tries 400/800/1200/2000 depth landings, requires walkable
contact at both ends, known headroom, door clearance and retention of the stair
silhouette. It chooses using training cameras and checks reserved cameras once.
The combined fallback then rechecks placement and every camera against the
original baseline at the stricter .005 depth tolerance. Original collision
records, stair endpoints, support and placement are unchanged; regular tread
counts retain the existing creative fallback policy. An accepted joint assembly
is reported and is not passed again through the later landing-only stage.

The frozen `build/room106-joint-creative-landing-probe` reproduces the rejected
full-trial fallback candidate and baseline geometry exactly. Its existing landing
fitter accepts all four sizes and selects2000, while retaining all-camera depth
checks. This is a component diagnostic, not a full batch result or a visual pass.
A separate integrated fallback replay is at
`build/room106-joint-interpretation-probe`; do not treat preliminary files as
a completed reconstruction.


Creative assembly evaluation
----------------------------
The detail evaluator now records the selected creative alternative separately
from rejected fit metrics. Its `creative_geometry` field includes the final
shape, measured placement, acceptance gates, end connections, baseline/candidate
depth errors and selected landing checks. Known headroom comes from the selected
combined assembly, rather than the earlier rejected flight; absent, malformed
or rejected selection metadata leaves the evidence unknown. Old mask IoUs and
held-out fit scores are still omitted when a creative replacement supersedes
their geometry. The HTML presents these checks separately from the checkpoint-
bound visual audit; none of them constitutes original fidelity or traversal
proof. Nine evaluation tests and Ruff pass. Keep this reporting change scoped
to the evaluator, its tests and this document.

`build/room106-joint-interpretation-probe/component-evidence.json` exercises the
evaluator against the real accepted component replay. Its selected2000 landing
has known sampled headroom and walkable end contacts. This record is explicitly
component-scoped; it does not count as an adopted full-room reconstruction.


Experimental visual collision proxy cleanup
------------------------------------------
`visual_stair_proxy_cleanup: true` requires parametric details and runs after
object fitting, before appearance projection. Defaults and named profiles keep
it off. It considers only non-wall collision rectangles fully inside the real
XZ footprint of a ground-level parametric stair and its directly attached
landing. AABB containment alone is insufficient. Structural/shared walls,
explicit boundary heights, already skipped boxes, raised-flight supports and
fixture anchor groups are excluded. The original collision records and all
actual primitives remain unchanged; only provisional visual boxes may be omitted.

Only training cameras can nominate a box: an exact floor-category mask needs
detector score>=.3,mask IoU>=.9 and>=500pixels; the box must block>=500floor
pixels, with>=50 reliable model-depth pixels and the existing12%/180 depth
contradiction margin. At most three boxes and four alternatives are tested.
Selection uses fixed training evidence and the original baseline, requiring
no camera regression above.005 and a mean improvement above.0005. Reserved
cameras check the selected alternative once and can reject it without a second
selection. If every camera is reserved, the stage reports no training evidence.

Cleanup writes its own report and recipe. Detail render measurements are marked
`before_visual_proxy_cleanup`; the evaluator omits their old mask IoUs, held-out
scores, creative placement and old scene-depth measurements rather than calling them final-scene
evidence. Parametric shape and unchanged intrinsic geometry checks remain
separate. Numerical cleanup acceptance never overrides visual inspection.
Keep this step scoped to the new cleanup helper/tests, batch hook, evaluator/
evaluation tests and this document (six files).

The current cleanup step passes58tests (3cleanup,45batch,10evaluation) and
Ruff. Frozen component replays in `build/room3d-visual-proxy-cleanup-replays`
accept the ROOM101 visual proxy replacement with exact flight/landing footprint
containment, and leave ROOM106/ROOM300 unchanged because they have no eligible
ground-flight proxy. Original exports, input caches and primitive geometry are
verified unchanged. The full integration trial at
`build/room101-proxy-cleanup-full-trial` completed successfully with unchanged
inputs, frozen sources and original export. It is another trial of the same room,
not a fresh unseen-room validation. All six final ROOM101 cameras were
reviewed: proxy removal reveals more floor under the stairs, but major wall
obstructions and incorrect stair pose/proportions remain. Its checkpoint-bound
audit is `needs_further_geometry_work`; this is not an overall visual pass.

### Additional fixed-floor depth veto (experimental)

`floor_depth_guard: true` adds a second depth check to the opt-in visual stair
proxy cleanup. Raw depth checks and training ranking remain in force. Before
comparing candidates, each view's reference is calibrated against the unchanged
baseline's visible floor, using exact floor-category masks with detector
score>=.3, mask IoU>=.9 and>=500 pixels. Existing calibration requires enough
visible samples, checkerboard validation, scale within .5–2 and validation
90th-percentile log residual<=.08; insufficient evidence keeps the raw reference.
Candidates cannot change calibration samples or scale. A regression above .005
against the fixed reference vetoes the candidate even if raw depth improves.
Reserved-view calibration runs only after training selection and may reject
the selected candidate without choosing another. Input depth arrays are never
modified. This is a model-bias sensitivity check, not ground truth or a replacement
for silhouettes, geometry constraints or visual review. No default profile
enables it, and it does not alter intermediate search objectives or ranking.

The reproducible ROOM106 diagnostic
`build/room106-floor-wall-calibrated-ranking` fixes calibration before comparing
wall-height alternatives. Its 3000-unit alternative passes raw depth checks but
fails the calibrated reference in corridor cameras 3 and 4. Lower alternatives
still have incorrect architecture; none was adopted. This follow-up is scoped
to the depth-guard helper/test, cleanup helper/test and this document (five files).

Two depth-guard unit tests and three rendered cleanup tests pass, including
reserved-view rejection without changing training trials. Ruff passes for the
four Python files. Frozen component replays in
`build/room3d-floor-guard-cleanup-replays` retain the positive ROOM101 cleanup
and leave ROOM106/ROOM300 unchanged, preserving inputs, source snapshots,
primitives and completed exports. The actual helper replay at
`build/room106-floor-guard-wall-replay` evaluates all eight cameras of the fixed
wall-height alternatives: 750/1500 pass both checks; 3000 passes raw depth but
fails the additional guard. These are component validations on previously used
rooms; they do not prove full reconstruction or validation on a fresh positive room.

### Selected detail assembly veto

The same option now checks each accepted final detail assembly in `fit_objects`,
after support selection, creative fallback and optional stair landing fitting.
It fixes the reference to the scene immediately before that object. Training
views are checked first; a training regression stops evaluation before reserved
views are calibrated. If training passes, reserved views check that one selected
assembly. Failure restores the previous recipe, marks the object rejected with
`floor_depth_safe: false`, and retains the candidate meshes and reports for
inspection. There is no second candidate choice after the veto. Subsequent
objects fit against the restored scene. This applies to selected fixture,
stair/landing and railing assemblies, not just visual proxy removal.

Four depth-guard tests and49shape tests pass, with Ruff on the affected Python
files. The actual selected-scene helper replay in
`build/room106-selected-scene-guard-replay` accepts the fixed750-unit wall
alternative and rejects3000 on the reserved checks. Cameras3/4 were deliberately
reserved after earlier diagnostics to exercise this code path; this is not
independent held-out generalization. Inputs, candidates, sources and completed
exports are unchanged. A frozen full integration trial is running at
`build/room101-floor-depth-guard-full-trial`; its final output and visual result
remain unverified until completion. This follow-up is scoped to the depth-guard
helper/test, shape fitter and this document (four files).

The evaluation HTML now includes selected-assembly depth-veto decisions and
training/reserved checks. If a later proxy cleanup changes visibility, it keeps
the historical decision/reason but omits the old depth measurements, explicitly
marking their scope. Eleven evaluation tests pass, including that distinction
and rendering the rejection reason in HTML; Ruff passes. This reporting
follow-up is scoped to the evaluator, its test and this document (three files).

Metadata-only validation selection is frozen in
`build/room3d-floor-guard-validation-selection.json`: the lexicographically first
completed entry outside known experimental stems with a detail observed in at
least three views is ROOM10E0, with a lamp group in cameras1/2/3. No acceptance
filter or image inspection determined selection. The frozen full-room trial
is running at `build/room10e-floor-depth-guard-validation`. Subsequent input-only
inspection confirms a desk/table lamp in all three originals, despite cached
`wall lamp` labels. The existing baseline reserves camera3; the new run's final
association, mounting choice, silhouette and actual reserved checks remain
unverified until completion. The input review is not a final visual audit, and
the frozen trial was not tuned after viewing it.

ROOM10E validation has since completed with unchanged sources, selection,
caches and original export. The new fit retains training cameras1/2 and reserves
camera3. It proposes a table-lamp family anchored to a horizontal furniture
surface despite the cached label. Mask IoUs improve from.607/.293/.276 to
.825/.487/.572 across the three views, but all-camera raw depth rejects the
assembly: camera3 error increases from.0912 to.1192. No lamp or railing is
adopted. All four final views were inspected and the checkpoint-bound audit is
`needs_further_geometry_work`, with large intrusive furniture/wall surfaces and
distorted textures. This is a useful negative validation, not a successful
positive reconstruction. The full evaluation is
`build/room3d-floor-guard-independent-evaluation/index.html`.

After this validation, ROOM10E is treated as a development example. Controlled
ablations at `build/room10e-lamp-depth-rejection` exactly reproduce the original
before/selected meshes: in camera3 the lamp replacement alone adds.00922 depth
error, support-height change alone adds.01930, and the combined candidate
adds.02800. The lamp-only ablation may float above the unchanged support and is
not an admissible design. Region attribution at
`build/room10e-lamp-regression-regions.json` finds.02616 of the combined penalty
outside the selected source lamp mask and.00185 inside. This does not support
ignoring only luminous lamp pixels or weakening the global gate. A component
probe at `build/room10e-fixed-support-lamp-probe` now tests fitting on the original
support without height alternatives. It completed, selecting an unanchored
generic shade rather than an attached table lamp. Training IoUs are.693/.493
and reserved IoU.604, but camera3 depth still regresses from.09123 to.09909,
failing the unchanged.005 gate. This does not establish a physically supported
or admissible replacement. These
diagnostics preserve inputs, original exports and collision data.

The complete ROOM101 depth-guard integration trial also finished successfully,
preserving five accepted lamps, the stair/landing assembly and proxy cleanup.
All six final rendered images are byte-identical to the previous completed
proxy-cleanup trial (`final-render-equivalence.json` records the hashes). All
six current views were inspected and remain `needs_further_geometry_work`:
wall obstructions, incorrect stair pose/proportions and stretched appearance
persist. This verifies full-run integration without losing that existing
positive example; it does not prove general visual quality or a new positive
reserved-view case. The current checkpoint-bound evaluation is
`build/room3d-depth-guard-final-evaluation/index.html`.

### Local creative fixture pedestals and proxy-free fixture reference

Two separate opt-in options are under development; no default profile changes:

* `local_fixture_pedestals: true` adds a bounded local pedestal alternative to
  each fitted table-lamp support-height proposal. It preserves the original
  furniture height/footprint and collision records. The pedestal covers the
  rotated lamp-base footprint plus one stem-thickness margin, must lie entirely
  within an existing non-wall parent footprint, and may bridge only a positive
  lift of at most1000units. Parent contact and fixture contact are both checked.
  Wall, skipped, shared-wall, below-surface and out-of-footprint supports are
  excluded. Whole-support-height alternatives remain available. Training scores
  are recomputed on each actual assembly; equal loss prefers the local change.
  Reserved scores are recomputed only for the selected local assembly.
* `fixture_proxy_depth_reference: true` uses the background without the matched
  inferred lamp proxies as the raw/floor-calibrated depth reference when a lamp
  has at least two observations. It retains native, script-owned and other
  geometry. No depth pixels are omitted and the.005tolerance is unchanged.
  Source-mask before/after comparisons still use the original visible scene;
  original proxy-scene depth errors are retained separately. Reports and the
  evaluation identify the alternative reference explicitly. This prevents
  coarse proxies inferred from DA3 itself from becoming an independent truth
  standard for a shared replacement. It is not evidence that DA3 is wrong or
  permission to ignore other scene geometry.

The component probe `build/room10e-local-fixture-pedestal-probe` constructs a
supported local pedestal without reoptimizing the selected lamp. It passes
depth against the proxy-free background reference but still regresses against
the original proxy scene; both comparisons are recorded. Same-policy bakes and
the comparison HTML show a cleaner single procedural lamp instead of overlapping
bright proxies, while major furniture/wall errors remain. The component is not
adopted and does not establish full-room fidelity or independent generalization.
The full development trial is running at
`build/room10e-local-pedestal-full-trial`; final selected assembly, checks and
images remain unverified until completion.

Three pedestal tests verify physical contacts, exclusion cases, unchanged
furniture and invariance of training selection when reserved mask/depth change.
The49shape,4depth-guard and11evaluation tests also pass; Ruff passes for the
affected Python files. Keep this feature scoped to the pedestal generator/test,
fixture fitter, shape fitter, evaluator/evaluation test and this document
(seven files).

The full ROOM10E local-pedestal trial has now completed with unchanged frozen
sources, inputs, selection and original export. It selects the local pedestal
and one shared table lamp, preserving furniture height/footprint and native
collision. Training cameras1/2 choose the assembly; reserved camera3 retains
mask IoU.572 and valid contact. Raw and floor-reference checks pass against the
explicit proxy-free background, while the recorded original-proxy comparison
still regresses. All four final views were inspected: the overlapping bright
lamp proxies are replaced by a cleaner procedural lamp, but furniture/wall
blocks and stretched textures remain. The checkpoint-bound audit is therefore
`needs_further_geometry_work`, not an overall pass. Current comparison/evidence:
`build/room3d-local-pedestal-evaluation/index.html`.

A separate metadata-only selection outside known experiment stems is recorded
in `build/room3d-local-pedestal-validation-selection.json`, choosing ROOM1130
with a lamp observed in cameras0/1. The frozen full trial completed at
`build/room113-local-pedestal-validation`, with source, input, selection and
original export hashes unchanged. All four source/final views were inspected
and the audit is bound to the completed checkpoint. A wall fixture is accepted,
but the shared fixture in cameras0/1 is a ceiling fan, outside the table-pedestal
family. This is not a positive validation of local pedestals. Oversized fixture
shade, fragmented fan, furniture/wall blocks and stretched bathtub textures
remain; overall status is `needs_further_geometry_work`. The two fan views do
not provide a reserved camera. Comparison:
`build/room3d-local-pedestal-validation-evaluation/index.html`.
No default profile promotion follows from the ROOM10E development improvement.

### Experimental ceiling-fan family

Recipe option `ceiling_fan_fixtures: true` adds independent fan starts to the
lamp fitter. Defaults and existing profile flags are unchanged. The generator
uses a ceiling stem, hub, one lower light and three to six closed tapered radial
blades. Radius, height, blade width/thickness, count and yaw are adjustable;
invalid overlapping dimensions are rejected. Shade/support material roles keep
light appearance separate from the blades. Existing multiview selection and
final acceptance gates still apply; generating a fan is not accepting it.

The local ROOM113 focused detector diagnostic
`build/room113-ceiling-fan-detection` uses the exact `ceiling fan` prompt with
unchanged DINO/SAM thresholds. Visual inspection finds useful proposals in
cameras0/1, a false toilet proposal in camera2 and a whole-image false proposal
in camera3, despite high detector/segmentation scores. Original caches are
unchanged. These masks have not been automatically merged or promoted; a
category prompt alone does not prove identity or correspondence. ROOM113 is
now a development room. Rendered synthetic tests cover actual family selection,
reserved-mask/depth independence, transformed ceiling contact and material roles;
unseen-room generalization remains unverified.

Controlled ROOM113 development fitting completed in
`build/room113-ceiling-fan-family-probe`: the same manually identified pair of
focused masks0/1 is fitted with and without the new family. Fan training IoU is
.482/.525 versus legacy .523/.288. All four raw-depth checks pass against both
the original proxies and the stripped background. This is not full acceptance:
the fitted stem is 885 game units below the inferred ceiling (100-unit proximity
limit), and no view is reserved. All four source/rendered views were inspected.

The initial appearance comparison made fixtures white; a subsequent own-mask
palette comparison made the tiny light dark because blades dominate its mask.
For exclusively `ceiling fan` palette sources, the upper quantile is now .98
instead of .85. Other categories retain .85. The .15 contrast requirement and
own-mask/heldout restrictions are unchanged; this is estimated baked appearance,
not emission or recovered albedo. Geometry was not refitted for these bakes.
The final component comparison, with black blades and a bright central light,
is `build/room113-ceiling-fan-bright-core-probe/index.html`. All four views were
inspected; component findings and image hashes are stored externally in
`build/room3d-component-visual-audits`. Native inputs, prior exports and frozen
sources are verified unchanged. Fan pose/proportions, ceiling attachment and
the larger room artifacts remain unfinished. No automatic focused-mask recovery,
default promotion or unseen-room positive validation follows from this trial.

### Finite ceiling connection for fan fixtures

Recipe option `ceiling_fan_anchors: true` replaces free fan starts with proposals
attached to existing finite horizontal ceiling surfaces. It is opt-in alongside
`ceiling_fan_fixtures`; default profiles remain unchanged. The fan's new optional
`light_drop` parameter separates hub/light level from total suspension height;
omitting it preserves the previous `.35 * height` geometry. The nearest eligible
underside per ceiling group may extend the stem by at most 2000 game units.
The full stem cap must contact the actual generated ceiling triangles. Omitted,
absent or out-of-footprint ceilings cannot supply support. Anchor refinement
slides within the support plane and retains the existing displacement bound.

Regardless of that option, final fan acceptance now requires a ceiling prior
and finite stem-cap contact. Unattached fans cannot pass simply because their
silhouette/depth agreement improves. Connection is to an **estimated ceiling**;
this is not proof of recovered roof height or mechanical support.

Frozen development replay `build/room113-ceiling-fan-attachment-probe` compares
the previous floating fan, stem-only attachment, and subsequent constrained
fitting with the same own-mask appearance. Attachment closes the 885-unit gap
without changing any hub/light/blade vertices. The refined mask IoUs are
.508/.610 and stem-cap contact passes. All four raw-depth checks pass against
both the floating candidate and the original proxy scene. Inputs, prior complete
export and frozen sources are verified unchanged. All four final source/rendered
views were inspected: the fan is recognizable but placement remains approximate;
large furniture/wall blocks and bathtub projection errors remain. Comparison:
`build/room113-ceiling-fan-attachment-probe/index.html`.

The same replay records automatic association of **all four** focused detector
proposals: cameras0/1 form a shared object, while the false proposals2/3 remain
single-view groups. This is useful grouping evidence, but the component fit still
uses the previously inspected pair0/1. Automatic focused-mask recovery, replacement
of prior coarse proxies and a complete batch export remain unverified. No reserved
view or new-room generalization claim follows from this development replay.

Rendered tests cover body preservation, finite roof contact and omitted roofs,
anchor sliding, reserved-mask/depth independence, and real final-scene acceptance
or rejection of contacting/floating fans. Only the expensive parameter optimizer
is controlled in the final-gate test; association, rendering and acceptance gates
are exercised. Geometry, appearance and navigation limitations remain explicit.

### Automatic focused fan-mask recovery

Recipe option `ceiling_fan_prompt_recovery: true` queries the local pinned
DINO/SAM models for `ceiling fan` when an existing confident fixture mask is
wide, near the upper part of a view (width >=80 pixels, aspect >=3, median Y
within .45 of image height, >=150 mask pixels). This is a query trigger, not a
fan classifier. The option requires segmentation/depth, `parametric_details`,
`ceiling_fan_fixtures` and `ceiling_fan_anchors`. Defaults remain unchanged.
Optional `ceiling_fan_recovery_cache` names a parent directory containing
`roomXXXX` focused SAM caches; their image hashes are verified before use.
Without that cache, focused inference runs locally/offline in a subprocess.

`tools/room3d_fixture_recovery.py` rejects weak/nonfinite confidence and tiny or
frame-filling proposals (normal .3 detector/.8 SAM gates, 150 pixels to .3 of
image area). It retains only exact-category proposals associated in at least
two original cameras by the existing finite-ray/depth grouping. Neither model
confidence is increased. This is correspondence evidence, not proven identity,
and occluded/unseen views are not forced into a group.

Recovery writes a separate cache. A contained old lamp/chandelier observation
is superseded only when >=.85 of its pixels lie in exactly one retained fan
proposal, with at most 10x area growth. Ambiguous overlaps remain unchanged.
The original source cache is never edited. Old entries and mask files remain
in the recovery cache; aliases shadow covered old observations **only during
detail fitting**. Their complete entries are also retained in `recovery.json`.
Baseline object proposals, structural consolidation and refinement consume the
original segmentation. New mask names are distinct and old-name aliases let the
fitter tentatively replace already inferred visual proxies when present; failed
fitting leaves those proxies in the scene. Native/script-owned geometry is not
selected by observation aliases. Focused fan masks retain the normal .35 minimum
IoU even when aliases match an older coarse proxy. Standard finite attachment,
raw depth and floor-reference gates still decide whether geometry is kept.

Before baking, `appearance_segments` promotes recovered masks only when the
final recipe contains geometry referencing them. Otherwise original observations
are restored for appearance. This prevents a rejected false fan hypothesis from
erasing ceiling texture semantics. The selected/discarded masks are recorded in
`fixture-recovery-appearance.json`. Accepted fans can use their own dark blades
and small bright light instead of unrelated fixture colors.

The initial full development run, before the rollback correction below,
`build/room113-automatic-fan-recovery-full-trial` completed successfully from
all four previously computed, image-bound focused model outputs. The program
selects views0/1, rejects the toilet/whole-image false proposals, and supersedes
four old detector observations without manual view/mask choice. A shared
five-blade fan is accepted with final mask IoUs .541/.525, finite ceiling contact
and all-camera raw/floor-reference depth safety. Only two views observe the fan,
so there is no reserved-view claim. All four final views were inspected and
bound to the complete checkpoint; overall audit remains
`needs_further_geometry_work` because wall/furniture blocks and bathtub texture
distortion remain. Original caches/export and frozen execution sources are
verified unchanged. Comparison/evidence:
`build/room3d-automatic-fixture-recovery-evaluation/index.html`.

A separate metadata-first control selection is recorded in
`build/room3d-fixture-recovery-validation-selection.json`: ROOM1170 is the first
completed >=2-view entry outside known development/inspected stems with a query
trigger and no exact background reuse from those stems. Selection precedes
source-image inspection or new focused inference and does not filter by
acceptance. The frozen full trial completed at
`build/room117-fixture-recovery-validation`, with fresh local model inference
and no supplied focused cache. The detector incorrectly associates ceiling
architecture in views1/3 as a fan. The candidate is rejected (IoUs .124/.263);
no detail object is accepted. All eight final views were inspected and bound to
the completed checkpoint: large wall blocks, missing ceiling detail and distorted
projections remain. Inputs, original export, selection and frozen sources are
verified unchanged. This is a negative new-room control, not a positive fan
example or justification for promotion; ROOM117 is now a development room.

That negative control exposed premature removal of one older detector observation
and promotion of rejected recovery masks into baseline/appearance processing.
The corrected implementation now keeps old observations for baseline geometry
and conditions recovered appearance on retained geometry, as described above.
Frozen full replays completed successfully at
`build/room113-fan-recovery-rollback-full-trial` and
`build/room117-fan-recovery-rollback-full-trial`, reusing all image-bound local
focused masks without manual selection. Paired recovery-disabled controls also
completed at `build/room113-fan-recovery-disabled-control` and
`build/room117-fan-recovery-disabled-control`. Sources and input contents are
identical within each pair; normalized commands match, and the sole functional
recipe difference is `ceiling_fan_prompt_recovery`. Complete checkpoints and
whole-run input/export inventories were verified unchanged after comparison.
These are development replays, not fresh new-room validations.

For ROOM117, all eight final assisted PNGs are byte-identical with recovery
on/off. No detail object is accepted, and rejected focused masks are discarded
from appearance. This verifies non-regression for this negative control; it does
not repair the existing wall blocks, missing ceiling detail or stretched textures.

For ROOM113, both recovery-on and recovery-off produce a creative five-blade fan
and a wall lamp. The strict evidence fit is rejected in the corrected recovery
run; its top-level evidence-fit diagnostics describe that rejected candidate,
not the final creative shape. Rendering the **actual final retained geometry**
against the same focused masks gives fan silhouette IoUs .551/.472 with recovery
and .520/.410 without it (views0/1). Both actual fans have finite ceiling contact;
views2/3 are byte-identical. Recovery improves this development example modestly;
it is not necessary for the fan family to produce a recognizable fan. There is
no reserved fan view and these scores are not fidelity percentages.

All source/final views in the four corrected/control exports were inspected and
bound to complete checkpoints. Every overall audit remains
`needs_further_geometry_work`. Paired immutable-input verification, actual-final
measurements and contact sheets:
`build/room3d-fixture-recovery-paired-verification/`.
Complete-export comparison:
`build/room3d-fixture-recovery-paired-evaluation/index.html`.

Further ROOM117 development diagnostics are recorded in
`build/room117-floor-occlusion-probe/results.json` and
`build/room117-partition-height-probe/results.json`. The central collision-derived
partition (`boundary_4`) blocks many segmented floor pixels, but view7's floor
classification includes a close-up painting: semantic confidence alone does not
justify wall removal. Six lower partition-height alternatives (1500 through
5250) all fail the all-camera raw/floor-reference depth checks. None is adopted;
inputs, original collision records and complete source export remain unchanged.

A metadata-first integrated-detail validation completed successfully at
`build/room209-integrated-details-validation`. Selection and background-reuse
checks precede source-image inspection and new fitting:
`build/room3d-two-view-details-validation-selection-checked.json`. It selects a
lamp associated in views0/2, with no exact background reuse from recorded
development/audited stems. There is no reserved object view with only two
observations. An earlier >=3-observation metadata scan found only four candidates,
all reusing prior development backgrounds (including ROOM60E/ROOM10E); they are
not counted as new-room validations. Its rejected-candidate trace is
`build/room3d-integrated-details-validation-selection-checked.json`.
ROOM209 accepts two creative wall fixtures, one shared in views0/2 and one from
view1. All three source/prior/final views were inspected and bound to the complete
checkpoint. Beds, walls and cabinets remain severely distorted, and lamps are
simplified/oversized; overall status is `needs_further_geometry_work`. Acceptance
does not establish visual fidelity. Inputs, baseline export, selection record and
frozen sources are verified unchanged. Comparison:
`build/room3d-integrated-details-new-room-evaluation/index.html`; contact sheet:
`build/room209-integrated-details-visual-review/comparison.png`.
ROOM209 is now a development room. This trial predates the structural fan-mask
correction and optional final-detail measurements described below.

Structural wall/floor masks also exclude confident exact `ceiling fan`
observations, matching the texture foreground guard. Overlapping background masks
cannot turn a recognized fan into wall/floor plane evidence. Non-finite detector
or mask-quality values are ignored in structural masks and texture support,
including the optional foreground guard. This is a semantic-evidence correction, not
new fan geometry or a claim that existing frozen exports have been rebuilt.

The opt-in recipe `final_detail_metrics: true` writes `final-details.json` after
all accepted detail/proxy changes. It measures retained parametric groups from
the actual `fitted.obj` against their linked observation masks, recording missing
observations explicitly. It never reuses rejected-candidate silhouettes or
changes fitting/selection. OBJ and camera-visibility hashes bind the report to
that geometry; the comparison evaluator refuses mismatches. Original fit reports
retain reserved-view status. Final measurements are post-selection diagnostics,
not new reserved-view tests, fidelity scores or navigation approval.
An external read-only component check at
`build/room113-final-details-module-validation/` applies the new measurement
function to the completed ROOM113 export. It reproduces the independent actual
final fan measurements (.551/.472), preserves mask/depth inputs and the complete
checkpoint, and writes its report outside the frozen run. Synthetic coverage
checks actual exported triangles, camera-specific hiding, missing observations
and immutable inputs. Evaluator coverage rejects changed OBJ or visibility
sidecars. No existing frozen run is retroactively modified to add this report.

The optional `creative_fixture_part_fit: true` runs part refinement **after** a
wall fixture is accepted as a creative interpretation. It uses existing creative
placement bounds rather than claiming the rejected strict whole-mask fit passed.
Three shade sizes and up to 56 anchor/arm alternatives each are searched from
training highlights; the anchor cannot move more than min(1000, fixture height)
from its accepted position. Bright-region IoU must improve in every contributing
training view. Original strict part fitting keeps its existing criteria.

One selected assembly is then checked for observed placement, finite wall contact,
door-panel clearance, all-camera raw depth and a fixed floor-depth reference.
Failure preserves the previously accepted object; reserved evidence never chooses
another alternative. Final reports distinguish the refined geometry from earlier
interpretation trials. Brightness remains an appearance hypothesis, not proof of
bulb identity. This option is not promoted into the default profile.

ROOM209 component evidence is at
`build/room209-creative-size-component-validation/`: the shared lamp's bright-part
IoUs improve from .224/.167 to .279/.247, and the other lamp from .161 to .245,
with 20% smaller shades and unchanged all-camera raw/floor-reference errors.
Source/mask/before/after images were inspected; this is a modest fixture change,
not a repair of the room's beds or architecture. Source cache, export and frozen
execution sources remain intact. A full development replay with this option and
`final_detail_metrics` completed at
`build/room209-creative-fixture-parts-full-trial`. Both accepted creative fixtures
pass post-acceptance part safety; all-camera raw-depth deltas are zero. The actual
final whole-object mask IoUs are .480/.638 for the shared fixture and .676 for the
single-view fixture. These are different targets from the bright-core IoUs above.
All three final source/prior/after views were inspected and bound to the complete
checkpoint; overall audit remains `needs_further_geometry_work` because beds,
walls and cabinets are severely distorted. Input caches, original export,
selection and frozen sources are verified unchanged. The final metric hook is
now exercised by a complete real batch run. Comparison:
`build/room3d-creative-fixture-parts-full-evaluation/index.html`.

A metadata-first further control selects ROOM20A0 from completed entries outside
recorded development stems and exact prior background reuse, without looking at
source images or filtering acceptance. Its trace is
`build/room3d-creative-parts-new-room-selection-checked.json`. The frozen trial at
`build/room20a-creative-fixture-parts-validation` and exact-source part-fitting-disabled
control at `build/room20a-creative-fixture-parts-disabled-control` completed with
unchanged inputs, selection, original export and frozen source inventories.
Only `creative_fixture_part_fit` differs functionally within this pair.
All six source/OFF/ON camera comparisons were inspected and both exports have
checkpoint-bound `needs_further_geometry_work` reviews. Final whole-object mask
IoU falls from .571 to .426 in camera1 and from .633 to .328 in camera2, despite
improved brightness-component fits and passing depth gates. Oversized shades,
thick supports, distorted walls and furniture remain. This is negative evidence
for promoting the option, not an overall visual improvement. Comparison:
`build/room3d-creative-parts-room20a-paired-evaluation/index.html`; immutable
paired verification: `build/room20a-creative-parts-paired-verification/results.json`.
This is one room/control pair, not two independent validation rooms; ROOM20A
is now development data. Changed fixtures have single-view observations, so
these results do not validate shared-object or reserved-camera generalization.

The creative part safety wrapper now also renders the selected complete object
before/after against each observed whole-object mask. A loss exceeding .05 IoU
vetoes the selected refinement and restores the exact prior recipe/object. This
uses observed masks, not brightness hypotheses or additional candidate searches;
training views are checked first and reserved views can only veto. A synthetic
render test exercises training and reserved silhouette vetoes, reserved depth
veto, successful acceptance, one optimizer call and immutable inputs. A fresh
frozen-source development replay completed at
`build/room20a-creative-parts-silhouette-guard-trial`. Both regressing refinements
were vetoed, and all six final PNGs exactly match the disabled control. Its inputs,
original export and frozen sources remain unchanged. The six camera comparisons
were explicitly inspected and the completed export has a checkpoint-bound
`needs_further_geometry_work` review. Comparison:
`build/room3d-creative-parts-silhouette-guard-evaluation/index.html`; verification:
`build/room20a-creative-parts-silhouette-guard-verification/results.json`.
This validates rollback on a development room, not improved underlying geometry
or new-room generalization. The gate limits regression relative to an already
accepted creative fixture; it does not establish its identity or fidelity.
For reviewable publication, split this work into the standalone joint part-search
API/tests first, then its safety wrapper, fitter/evaluator integration and tests;
each dependency step stays usable without exceeding seven changed files.

The all-room driver has terminated with 348 RDT entries covered: 318 image-assisted
exports, 28 empty slots and two errors (ROOM3100/ROOM3101). Full checkpoint artifact
inventories and original RDT bindings for all 318 exports were verified by
`build/room3d-terminal-inventory-verification.json`. This run used a live worktree,
so it is not recorded as a frozen-source reproducibility experiment. Variants and
repeated stage backgrounds are not independent physical rooms or validations.
ROOM310's singleton DA3 failure has a separate fix, but its subsequent visibility
trial still has no visible geometry; those drafts are not merged as successful
volumetric exports. The goal remains incomplete.

An additional read-only PNG inventory found fully black renders in ROOM3110/1,
and one black camera each in ROOM40C0/1 and ROOM7180/1. Sources for ROOM3110's
three cameras, ROOM40C0 camera7 and ROOM7180 camera2 were explicitly inspected
and contain scenery. Actual geometry-depth renders confirm zero coverage in all
ten affected views, including when the visibility sidecar is ignored. This is
missing projected geometry, not merely dark textures or a sidecar suppression.
ROOM311's exported XZ extent is only [0,36] by [0,18], matching its tiny native
collision bounds, while its cameras are thousands of units away. ROOM40C and
ROOM718's affected cameras also lie outside the exported extent. These are
diagnostics, not permission to reinterpret native collision bounds as complete
architecture or evidence of script usage. Checkpoint exports remain unchanged.
Reports: `build/room3d-terminal-image-visibility-audit.json` and
`build/room3d-black-view-diagnostic/geometry-results.json`; inspected images:
`build/room3d-black-view-diagnostic/comparison.png`. Existing 318 export counts
therefore describe generated artifacts, not 318 visually usable reconstructions.

Experimental visual surfaces outside collision extents
----------------------------------------------------
`visual_surface_recovery: true` requires cached segmentation and depth. It runs
only when the current generated geometry has zero projected pixels in every
original camera. Native collision, script geometry, cameras and previous
surfaces are retained; this option adds estimated visual surfaces. It does not
repair partially empty rooms such as ROOM40C or ROOM718. No named profile enables
it by default, and existing all-room exports are not overwritten.

Two training cameras must support a floor and a reconciled shared vertical wall.
When there are three or more cameras, the last camera's masks and depth do not
choose planes, floor height or spatial extents. The proposal extends a rectangular
floor over the training extent and joins the shared wall to it; these additions
are explicitly estimated. Floor-height spread is checked against a provisional
allowance of max(300 units, 5% of median training floor depth). If estimates
disagree, the existing recipe floor may be retained only when every training
estimate admits that same height within the unchanged allowance. Otherwise the
proposal abstains. The tolerance is a model-consistency heuristic, not a measured
world-height error bound. Raw depth and minimum projected coverage then check
training views followed by reserved views; a veto restores the prior recipe with
no candidate reselection. Against an empty baseline, depth non-regression alone
is weak evidence, so shared semantic support and coverage remain essential.

The batcher and retouch projection pass structural-surface material categories
to the texture baker, allowing observed floor/wall material continuation instead
of unclassified gray. This extends appearance to unknown surfaces; it is not
recovery of their original textures. Synthetic rendered tests cover shared planes,
inaccessible reserved masks, height disagreement with/without a compatible prior,
reserved visibility veto, already-visible abstention, input preservation and JSON
serialization of final reports/recipes.

ROOM311 development diagnostics are at
`build/room311-semantic-plane-probe` and
`build/room311-semantic-surface-appearance-aware/index.html`. Source/all-source/
reserved-appearance images were inspected: wall and floor become visible in all
three cameras, but sky, trees, railings, door geometry and original paving remain
incomplete. The first full batch trial at
`build/room311-visual-surface-recovery-full-trial` abstained on floor-height spread.
The floor-prior trial exposed a report serialization error; its failed export is
retained separately. The corrected frozen replay at
`build/room311-visual-surfaces-floor-prior-verified-trial` completed successfully,
with unchanged cached inputs, original export and frozen sources. Its original
three cameras now have 28,561 / 74,295 / 74,125 projected geometry pixels, including
reserved camera2. All three source/baseline/final rows were explicitly inspected
and the export has a checkpoint-bound `needs_further_geometry_work` review.
Sky, trees and railings remain absent, door geometry is unrecovered and floor
projection seams remain. The OBJ is 1,760,705 bytes, with 17,304 vertices and
5,768 triangular faces; this is a static size check, not an in-game smoke test.
Comparison: `build/room3d-visual-surfaces-room311-evaluation/index.html`.
These are development replays, not independent new-room fidelity results.
Publish this as a separate five-file dependency step (visual surface module,
batcher, retouch, rendered tests, documentation), after the three-file creative
part-silhouette guard step. Do not combine unrelated cleanup with either step.

Recovery tests exercise real projected multiview masks, false full-frame masks,
cached positive recovery, aliases and confidence preservation, immutable inputs,
image-hash rejection, ambiguous supersession and a negative room that loads no
models. Wider fixture coverage, missed triggers and false shared proposals still
require independent room evaluation.

An automatic metadata scan outside prior detail-trial room stems found no
additional eligible exact ground-flight/proxy footprint among the currently
completed all-room entries. The selection trace is
`build/room3d-proxy-independent-selection.json`; it is metadata discovery, not
a frozen reconstruction evaluation or new positive validation. Proxy cleanup
therefore has one real positive component example (ROOM101), two conservative
controls (ROOM106/ROOM300), and rendered synthetic reserved-view coverage.
Do not turn these into a claimed broad fidelity improvement or unseen positive
room count.

Experimental continuous and segmented handrails
----------------------------------------------
`handrail_candidates: true` adds two bounded initialization alternatives to
railing fits; existing baluster starts are retained. The default railing family
is unchanged. The `handrail` profile generates a continuous beam without
inventing posts. A training-only PCA seed supports arbitrary horizontal direction
and zero rise, equally samples views and estimates beam thickness. When at least
80% of each observation falls on a visible native boundary face, an additional
existing-wall design prior may be proposed. A single planar wall produces one
beam; multiple supported planes may produce a `handrail_segments` assembly of
up to four beams. RANSAC wall patches require at least 24 samples and 10% of the
combined source cloud, using the existing 1%-of-depth plane tolerance. These are
estimated wall/rail interpretations, not original wall or attachment proof.

Segments use a centered local frame so pose fitting rotates around the assembly
rather than the world origin. Their topology remains fixed during coordinate
search. The existing multistart render objective and final depth/visibility gates
select the family. Reserved observations never initialize it. Accepted continuous
rails retain an explicit `estimated` qualification for hidden supports, joins,
attachment, fidelity and navigation. The creative fallback does not add posts
to a continuous-rail profile. After the independent ROOM405 trial and paired
creative-profile integration below, `creative-v2` enables this bounded
alternative. `existing` remains unchanged, and explicit room recipes can set
`handrail_candidates: false` to disable it.

Metadata selection at `build/room3d-railing-new-room-selection.json` chose ROOM4000
before source inspection, excluding recorded development stems and exact reused
backgrounds without filtering prior acceptance. Only two observations remain
for this selected group, so there is no reserved-camera room-validation claim.
Local Qwen4B review of both image-bound outlined regions proposed wall-mounted
handrails (`build/room400-railing-identity-probe`); those responses are diagnostics
and are not consumed as automatic geometry approval.

Initial continuous-beam trials selected the old baluster fit because the model
depth seed was hidden by generated walls. Visible-wall points also exposed a
nonplanar corner layout, so a single native-wall plane abstained. The frozen
segmented component trial at `build/room400-handrail-segments-component-trial`
improved whole-mask IoU from .378/.362 to .758/.792 and modestly reduced raw
scene-depth error in both observed cameras; the other four raw scores were
unchanged. Its source/control/selected isolated-object overlays were explicitly
inspected: false posts disappear, but rail thickness, corner joins and supports
remain approximate. These overlays are not full textured room renders. The
initial component used a world-origin segment frame; subsequent full trials use
the corrected centered frame and are separate experiments.

Frozen full-batch paired exports completed at
`build/room400-handrail-full-trial` and
`build/room400-handrail-disabled-control`, with identical source snapshots and
cached inputs; only `handrail_candidates` differs functionally. Final mesh IoUs
are .758/.792 with the option and .378/.362 without it. The shared rail is
accepted in both runs, and complete exports verify unchanged. All six
source/OFF/ON comparisons were inspected: invented posts disappear in cameras0/1;
cameras2/3/4/5 are byte-identical. Wall, ceiling and furniture distortions persist,
so both exports have checkpoint-bound `needs_further_geometry_work` reviews.
Comparison: `build/room3d-handrail-room400-paired-evaluation/index.html`; verification:
`build/room400-handrail-paired-verification/results.json`. ROOM400 is now
development data. Rendered tests cover straight and corner families, arbitrary
direction, horizontal rails, native-wall proximity, unavailable reserved depth
points and actual multiview family selection. Keep this as a separate six-file
publication step (generators, handrail priors, fitter, creative fallback, tests,
documentation), independent of the earlier silhouette guard and surface recovery.

Independent metadata selection at
`build/room3d-handrail-independent-selection.json` chose ROOM4050 next, without
filtering prior acceptance, and excluded ROOM400 and its exact background reuse.
The algorithm snapshot was frozen before inspecting ROOM405 source images or
fitting, and no target-specific edits were made. The shared continuous-rail
component improved mask IoU from .267/.014 to .766/.655; all eight raw scene-depth
differences relative to control stay within .005. Both source/control/selected
object-overlay rows were explicitly inspected: erroneous posts disappear and
the rail follows the wall trim more closely, with estimated width and supports.
These are object overlays, not a full room or navigation approval; there are
two observations and no reserved rail view. Component results and bound review:
`build/room405-handrail-independent-component-trial/index.html`.
Full paired batch validation using that exact pre-inspection snapshot completed
at `build/room405-handrail-full-trial` and
`build/room405-handrail-disabled-control`. The option retains the shared continuous
rail with final mesh IoUs .766/.655; control rejects it. Other retained lamp
geometry is unchanged. Complete exports, inputs and source snapshots verify
unchanged and identical across the pair apart from the recipe option. All eight
source/OFF/ON rows were inspected and both exports have checkpoint-bound
`needs_further_geometry_work` reviews. Cameras3/4/5/6/7 are byte-identical. Camera2
also shows the rail and changes; it passes the depth gate but has no linked
mask observation, so it is not a reserved silhouette or appearance test. Width,
material and attachment remain approximate, and substantial architectural
distortions persist. Comparison:
`build/room3d-handrail-room405-paired-evaluation/index.html`; verification:
`build/room405-handrail-paired-verification/results.json`.
ROOM405 is now development data for subsequent edits. This supports the bounded
rail alternative on one independent room; it does not establish whole-room
fidelity or justify adopting all experimental options in named profiles.

The subsequent creative-v2 integration pair at
`build/room405-handrail-integrated-creative-trial` and
`build/room405-handrail-integrated-creative-control` completed with identical
frozen sources/inputs and one functional option difference. The accepted rail
retains final mesh IoUs .766/.655, the same creative lamp remains in both runs,
and cameras3/4/5/6/7 are byte-identical. All eight source/OFF/ON rows were inspected
and both exports have checkpoint-bound `needs_further_geometry_work` reviews.
Comparison: `build/room3d-handrail-creative-integration-evaluation/index.html`;
verification: `build/room405-handrail-integrated-paired-verification/results.json`.
This is configuration integration on development data, not another unseen room.
An actual rendered open-baluster control selects exactly the same original
shape with the option on/off; continuous rails are alternatives, not replacements
forced on every category. Explicit-disable and per-room independence checks pass.
Profile adoption is a separate three-file publication step (profile, profile
tests, documentation), dependent on the handrail-generator step. Other experimental
fixture/architectural options retain their existing activation policies.

Per-mask model score provenance
-------------------------------
New detail reports retain detector score and SAM predicted-IoU per source mask,
including each mask merged into a shared observation. Boolean-mask pixel hashes
identify the actual mask used. Refined stair masks distinguish their SAM result
from `original_predicted_iou`; neither is upgraded to a geometry probability.
Nonfinite or out-of-range model scores cannot enter fitted observations. Accepted
and rejected objects retain the same provenance, so failures can be grouped by
category and raw score ranges without filtering to successful reconstructions.

The evaluator HTML displays per-object records and category failure summaries;
the catalog reports raw detector/SAM ranges and counts objects with incomplete
scores. Legacy missing values stay `not_recorded` or `partial`. Invalid values
are omitted explicitly rather than creating NaN JSON or inferred confidence.
The underlying model scores are not calibrated semantic or fidelity confidence.
Rendered fit tests exercise merged-mask preservation, rejected-object propagation,
pixel hashes, refinement-vs-original scores, malformed values and legacy summaries.
The frozen development replay at `build/room405-model-confidence-full-trial`
completed successfully. Read-only verification checks all six recorded mask-score
bindings against the image-bound segmentation cache and actual Boolean-mask
pixels, including accepted and rejected objects. Both complete exports retain
their checkpoint inventories; frozen sources and inputs retain their hashes.
All 751 PNGs, including the eight final camera views, are byte-identical to
`build/room405-handrail-full-trial`. The earlier explicit visual review is carried
forward by verified image identity and still reports `needs_further_geometry_work`.
This is a development metadata replay, not an additional unseen-room result;
refined-mask handling is covered separately by tests, not this replay.
Verification: `build/room405-confidence-provenance-verification/results.json`.
HTML: `build/room3d-confidence-provenance-evaluation/index.html`.
Keep this as its own six-file publication step (fit observations, confidence
summary helper, evaluator, catalog, tests, documentation).

Catalog export integrity
-------------------------
The catalog verifies the complete checkpoint inventory before labeling an
image-assisted room exported. Checking only the OBJ, survey, HTML and result
files omitted consumed failure reports, textures and visibility sidecars. A
changed detail report, missing texture or extra unrecorded file now produces
`incomplete_or_changed` and cannot feed accepted-detail or confidence summaries.
Regression cases use actual checkpoint save/restore, including recovery after
removing an unrecorded file. Full artifact hashing costs more than the previous
four-file shortcut; do not replace it with an export-key-only cache, which would
miss external file edits. Verified integrity still establishes neither visual
quality nor gameplay traversal.
