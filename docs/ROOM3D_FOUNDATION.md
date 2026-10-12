# Offline room reconstruction foundations

These modules provide independently usable foundations for the room batcher.
They do not change the game renderer or reconstruct a room on their own. Later
changes add detail fitting and complete reconstruction orchestration.

Procedural shapes
-----------------

`tools/room3d_shapes.py` generates serializable columns, stair flights with an
optional landing, railings and lamps. Its only dependency is NumPy. For example:

```python
from room3d_shapes import generate

stairs = generate("staircase", {"width": 1200, "run": 3600, "rise": 1800,
                                "steps": 9, "landing": 900})
```

The result contains closed eight-vertex parts, bounding limits and the parameters
used to generate them. Coordinates use negative Y upwards; origin and yaw apply
one rigid transform to all parts. Stairs support solid or suspended construction;
railings support posts, a continuous handrail or a bounded assembly of segments.
Lamp profiles include wall fixtures, candles and ceiling fans. These are estimated
design priors, not recovered original meshes. This library does not locate objects
in images, choose a fitting family, export OBJ or establish traversability.

`test_room3d_primitives.py` separates the existing generator invariants from the
later fitting integration suite. It checks regular steps and landing continuity,
suspended-flight clearance, valid closed columns, rigid transforms, rail contact,
lamp symmetry and rejection of invalid dimensions. Run it with:

```sh
python -m unittest discover -s tests -p test_room3d_primitives.py
```

Offline rendering and checkpoints
---------------------------------

`tools/room3d_preview.py` renders an existing OBJ with an explicit game camera:

```python
from pathlib import Path
from room3d_preview import render

camera = {"origin": [0, -2000, -4000], "target": [0, -1000, 2000],
          "roll": 0, "fov": 240}
image, visible, depth = render(Path("room.obj"), camera, return_depth=True)
image.save("preview.png")
```

Run with `PYTHONPATH=tools`. Install NumPy and Pillow in a Python environment.
Projection uses the game's 320×240 logical viewport and near plane. OBJ material
textures and optional per-camera `.cameras` visibility sidecars are supported.
This CPU diagnostic is for automatic comparisons; it does not establish runtime
appearance or whether the player can traverse the generated geometry.

`tools/room3d_checkpoint.py` saves an export inventory only for supported completed
statuses. `restore(checkpoint, key, room_dir)` returns the saved result only when
the key and complete file inventory still match. It verifies existing files and
does not repair an altered export. A successful checkpoint records completion
and integrity, not visual fidelity. The caller must derive the key from all
relevant inputs, source versions and options; `run_signature` supports the batch
CLI's argument namespace.

Run the standalone regression checks without game assets:

```sh
python -m unittest discover -s tests -p test_room3d_clip.py
python -m unittest discover -s tests -p test_room3d_checkpoint.py
```

The clipping check covers polygon order, near-plane boundaries, interpolated
attributes and immutable inputs. Checkpoint checks cover changed/missing
artifacts, changed keys, failed/partial results, empty slots and interrupted JSON.

Evidence reports and visual reviews
-----------------------------------

`tools/evaluate_room3d_details.py` builds a comparison HTML from completed export
manifests. It can inspect an existing run independently of the fitting code:

```sh
python tools/evaluate_room3d_details.py --run path/to/run --out build/evaluation \
  --completed-only --audits path/to/external-audits
```

`--completed-only` requires the matching checkpoint and unchanged full export
inventory. Actual final-detail measurements are checked against the current OBJ
and visibility sidecar hashes; scores from a replaced shape are not reused as
evidence for its replacement. Missing acceptance, reserved-view measurements or
visual review remain explicitly missing. This report does not rerun the fitter.

`tools/room3d_confidence.py` summarizes original detector and SAM scores per mask
and rejected category. These raw scores are not fidelity probabilities. Missing
legacy scores remain unknown, and invalid values cannot become nonfinite JSON.

`tools/room3d_visual_audit.py` stores explicit image-review findings outside the
input run and binds them to its complete checkpoint and reviewed image hashes.
Changing the export invalidates the binding. `approximate_visual_pass` is an
explicit review status, never an automatic consequence of a successful export.
Use `python tools/room3d_visual_audit.py --help` for its required review fields.

The report and audit regression checks run without game assets:

```sh
python -m unittest discover -s tests -p test_room3d_evaluation.py
```

Publication follows dependency order in small changes. This foundation is
separate from runtime integration and the full batcher; those remain subsequent
steps until their source and validation are included. Generated room exports,
original game assets, downloaded model weights and caches are not source files.

Image evidence and structural proposals
---------------------------------------

`tools/room3d_segment.py` loads image-hash-bound, native-size SAM masks and
provides semantic texture support. Generating new masks uses CPU-local inference
and the pinned SAM revision declared in that module; loading cached masks needs
only NumPy and Pillow. Detector scores and masks remain proposals, not truth.

`tools/room3d_refine.py` loads depth arrays bound to the survey, camera and RGB
image, then compares candidate geometry on fixed valid-pixel support. Missing
geometry retains a penalty. `tools/room3d_structure.py` distinguishes shared
surface support, free space, occlusion and unknown evidence. Collisions are a
topology prior; they are not automatically visible walls.

`tools/room3d_stair_edges.py` uses internal tread structure to propose and validate
stair masks. Its masks remain estimated; an overlapping railing label cannot
become a solid stair flight. Cached refinements are bound to their source images
and depth, and model confidence is retained rather than raised to force acceptance.
These three geometry modules also require OpenCV, NumPy and Pillow.

Run the native-scale stair-label and cached-refinement checks without downloading
models or supplying game assets:

```sh
python -m unittest discover -s tests -p test_room3d_stair_labels.py
```

Known-camera matching and projected appearance
----------------------------------------------

`tools/room3d_reconstruct.py` triangulates image features with the supplied game
cameras and optionally constrains them by collision bounds. Repeated feature
orientations do not count as separate evidence. These functions use NumPy and
OpenCV and are independent of model inference.

After geometry is fixed, `bake_projected` writes a textured OBJ from that mesh and
the original camera images. Its visibility/depth tests avoid projecting nearer
objects onto farther surfaces, and projection angle can reject grazing views.
The `exclude_camera` argument reserves a view for appearance evaluation. Geometry
is not optimized from that withheld image. `BakeDepthCache` binds cached raster
depth to the mesh, camera and visibility sidecar so it cannot survive changes to
those inputs.

`tools/room3d_complete.py` supplies local material continuations and per-object
fixture colors. `tools/room3d_fixture_appearance.py` preserves separate lamp
support and shade roles and can build a bounded bright-core patch. Inferred
textures on unseen parts are estimated appearance, not recovered original art;
the source images remain unchanged. These helpers require NumPy, Pillow and
OpenCV, without model weights.

Run the matching, palette-isolation and depth-cache regression checks:

```sh
python -m unittest discover -s tests -p test_room3d_correspondence.py
python -m unittest discover -s tests -p test_room3d_fixture_tints.py
python -m unittest discover -s tests -p test_room3d_bake_cache.py
```

Batch survey and baseline export
--------------------------------

`tools/survey_room3d.py` exports native cameras, collision bounds and an SVG plan
from a USA PC RDT. The existing decoder is preserved; original assets stay
read-only. Collision records describe gameplay constraints, not scenery meshes.

`tools/batch_room3d.py` provides resumable room selection and baseline export:

```sh
python tools/batch_room3d.py --assets path/to/USA --out build/room-survey \
  --survey-only
python tools/batch_room3d.py --assets path/to/USA --out build/room-mesh \
  --rooms ROOM4050 --mesh-only
```

The first command inventories all RDT entries. The second exports a provisional
collision extrusion and its survey. Neither command runs a vision model or
modifies game files. `--resume` reuses only matching input/code/recipe keys and
complete, unchanged exports. An empty RDT slot is distinct from a failed export.

`tools/room3d_floor.py` subtracts explicit openings from floor patches.
`tools/room3d_scene.py` supplies decoded room anchors and provisional object
priors. `tools/room3d_profiles.py` merges versioned options without mutating
caller recipes; explicit room settings override profile defaults. The `existing`
profile retains existing recipes. Profile isolation and CLI checks run with:

```sh
python -m unittest discover -s tests -p test_room3d_profiles.py
```

This publication step supports surveys and baseline meshes. The complete CLI
already contains reconstruction hooks; their remaining fitting, model and review
modules arrive in subsequent dependency steps. A baseline export is not a
completed reconstruction or an approval of its appearance or navigation.

Shared observations and bounded detail fitting
----------------------------------------------

`tools/room3d_shape_fit.py` forms per-mask observations, associates compatible
observations across cameras and fits one parameterized object to them. Shape
candidates are rendered with the original projection; the objective compares
silhouette, edges and estimated depth. When a group has at least three observed
cameras, its last camera is reserved for validation rather than initialization
or optimization. Acceptance thresholds remain agreement checks, not fidelity.

`tools/room3d_objects.py` supplies depth-derived object proposals.
`tools/room3d_connections.py` checks finite surface contact and stair headroom;
`tools/room3d_assemblies.py` provides connected landing and stair-proportion
priors. `tools/room3d_fixtures.py` anchors fixture hypotheses to finite surfaces.
These checks cannot establish gameplay traversal or recover never-visible parts.

The core fitter preserves each original mask's detector/SAM scores and pixel
hash, including masks merged into one observation and rejected fits. Invalid
model scores are excluded, refined-mask and original scores remain separate,
and failures retain evidence for category summaries. Run this integration check:

```sh
python -m unittest discover -s tests -p test_room3d_confidence.py
```

This step provides the default fitting functions. Experimental creative/support
options in the full source use additional modules supplied by subsequent steps;
the complete `creative-v2` profile is not yet the isolated publication target.

Supported stairs and independent depth vetoes
---------------------------------------------

`tools/room3d_supports.py` jointly considers a stair flight and nearby candidate
supports, retaining native collision records and an explicit training/validation
split. `tools/room3d_supported_occlusion.py` searches bounded raised alternatives
using the same tread objective as the initial fit. A foreground occluder does not
become proof of an object's recovered geometry.

`tools/room3d_depth_guard.py` checks a selected assembly against independent raw
depth and a conservative floor reference. Optional floor calibration requires
consistent evidence and cannot turn disagreement into a favorable score. The
reserved view can veto the selection; it cannot select another optimized winner.
These modules are opt-in support/safety components, not proof of navigation.

```sh
python -m unittest discover -s tests -p test_room3d_supported_occlusion.py
python -m unittest discover -s tests -p test_room3d_depth_guard.py
```

Creative flights and landing assemblies
---------------------------------------

`tools/room3d_levels.py` proposes finite floor/ceiling levels from observed views
and existing topology. `tools/room3d_creative_stairs.py` turns uncertain stair
evidence into bounded replacement-flight candidates, with regular treads and
explicit support, dimensions and clearance checks. These are approximate design
alternatives; the code does not assert that a creative flight is the original one.

`tools/room3d_interpretation.py` evaluates creative replacements separately from
the rejected literal fit. `tools/room3d_landing_fit.py` fits an attached landing
and its necessary opening as an assembly. Heldout evidence can veto the selected
assembly without participating in training selection. Reports retain estimated
geometry qualifications and the checks used to accept or reject the replacement.
Native gameplay collision and scripts are retained.

```sh
python -m unittest discover -s tests -p test_room3d_creative_stairs.py
python -m unittest discover -s tests -p test_room3d_landing_fit.py
```

These tests exercise training-only selection, reserved-view rejection, immutable
inputs and actual rendered contact/depth cases. They establish those invariants,
not fidelity across the full game's rooms.

Fixture parts with final-scene rollback
---------------------------------------

`tools/room3d_fixture_parts.py` fits recognizable lamp parts from bounded bright
regions, keeping ambiguous evidence separate. Its optional creative search can
resize shades jointly with their orientation and offset. The selected shape is
checked in the full scene by `tools/room3d_creative_fixture_parts.py`: raw depth,
floor reference, finite attachment, door clearance and whole-object silhouette
must remain within their declared bounds. A failed training or reserved-view
check restores the exact earlier recipe and object, without selecting a new
candidate using reserved evidence.

`tools/room3d_fixture_pedestals.py` proposes estimated finite support pads for
table fixtures when enabled; they are design alternatives, not original assets.
These features remain opt-in. The final-scene guard was added after the ROOM20A
trial revealed that brighter, smaller shades could worsen the object's overall
silhouette; the guarded replay restored every final image to the disabled control.

```sh
python -m unittest discover -s tests -p test_room3d_fixture_parts.py
python -m unittest discover -s tests -p test_room3d_creative_fixture_parts.py
```

Wall handrails and missing surface recovery
-------------------------------------------

`tools/room3d_railings.py` adds continuous and segmented handrail candidates to
the existing open-baluster family. Candidates use training observations only;
visible native wall patches can supply an explicitly estimated placement prior.
The existing fitter chooses among the families and applies the same silhouette
and depth checks. The rendered open-baluster control retains its original shape.
Full ROOM400 and independent ROOM405 comparisons improved the observed handrail
silhouettes; their overall room geometry still needs work. The bounded alternative
is enabled by `creative-v2`, with explicit per-room disabling preserved.

`tools/room3d_enclosure.py` fits shared architectural planes. The optional
`tools/room3d_visual_surfaces.py` uses them only when every baseline view lacks
visible geometry and adequate multiview surface evidence exists. It does not
silently repair partially empty exports. A conservative existing-floor prior
requires consistent support in all training views, and the reserved depth view
can veto the proposal. New walls and floors remain estimated visual surfaces;
native collision and scripts stay unchanged.

```sh
python -m unittest discover -s tests -p test_room3d_railings.py
python -m unittest discover -s tests -p test_room3d_visual_surfaces.py
```

The detailed local experiments and limitations remain in `docs/ROOM3D_BATCH.md`,
which accompanies the complete pipeline publication after its dependencies.

Local model inference and review queues
---------------------------------------

`tools/room3d_vision.py`, `tools/room3d_depth.py` and `tools/room3d_review.py` supply
local Grounding DINO detections, DA3 depth proposals and Qwen visual discrepancy
proposals. The modules declare pinned model revisions. Inference requires their
Torch/Transformers/DA3 dependencies and downloaded weights; cached evidence can
be consumed by the fitting pipeline without running these models again. Images
are processed locally, and model output never acts as authoritative acceptance.

DA3 aligns multiview depth to input camera baselines. A single-view room instead
uses an explicit estimated camera-target-distance editing prior; that scale is
not claimed to be a metric reconstruction. Neither path changes native cameras.

`tools/room3d_review_queue.py` separates completed exports from pending reviews
and reuses a review only when both image hashes, model revision and prompt match.
`tools/room3d_novel_views.py` renders provisional new viewpoints; these have no
original image reference and do not establish shoulder-camera gameplay.

```sh
python -m unittest discover -s tests -p test_room3d_review_queue.py
```

With these dependencies present, the batcher can finish image-assisted exports
using cached masks/depth and `--defer-review`. Pending review remains visible in
the report rather than being reported as a visual pass.
