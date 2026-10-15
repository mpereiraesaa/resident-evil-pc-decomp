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

Non-destructive room retouching
-------------------------------

`tools/room3d_retouch.py` edits generated OBJ groups in a separate output folder.
It supports finite translation, positive scaling, yaw and hiding, while retaining
the original export. Optional reprojection updates appearance for the changed
geometry. The assisted mesh is the editing authority; its appearance context and
fixture material roles survive sequential edits. Unsupported indices, paths or
transforms are rejected rather than silently normalized.

`tools/room3d_additions.py` adds explicitly estimated procedural objects when
requested. It does not send images elsewhere or change native collision/scripts.
See `python tools/room3d_retouch.py --help` for the operation schema and switches.
Retouching helps finish approximate outputs; it does not mean every generated
room is already correct enough for only small changes.

```sh
python -m unittest discover -s tests -p test_room3d_retouch.py
python -m unittest discover -s tests -p test_room3d_additions.py
python -m unittest discover -s tests -p test_room3d_fixture_appearance.py
```

These checks cover actual rendered edits, source-export immutability, sequential
reprojection, bounded additions and preservation of fixture support/shade roles.

Offline editor and integrity-checked catalog
--------------------------------------------

`tools/room3d_editor.py` generates a self-contained HTML wireframe editor using
the exported OBJ groups and original camera backgrounds. Its template has no
external script or image dependencies. It exports edits for the retouch CLI;
the original room export remains unchanged. Generate one with:

```sh
python tools/room3d_editor.py --room path/to/run/room4050 --out build/editor.html
```

`tools/room3d_catalog.py` lists completed, pending, failed and empty entries,
creates per-room editor links, and summarizes rejected details and original model
scores by category. Before marking an image-assisted room exported it verifies
the complete checkpoint inventory, including detail reports, textures and
visibility sidecars. Missing, changed or extra files invalidate completion;
checking only four main files or caching solely by export key would miss edits.
Full integrity hashing has a cost and does not establish visual quality.

```sh
python tools/room3d_catalog.py --help
python -m unittest discover -s tests -p test_room3d_catalog.py
node tests/test_room3d_editor.js
```

Committed-source validation
---------------------------

The first thirteen dependency steps passed 153 Python tests from an isolated
checkout. A full cached ROOM405 `creative-v2` replay used only the 150 tools files
from commit `d00c81d1671cb8dc04edd75c808474542973bbaf`. All 751 generated PNGs,
including eight final camera views, and actual final mesh measurements match the
earlier creative integration export. Six mask-score bindings match the original
image-bound cache. Both complete export inventories remained unchanged.

The prior explicit visual review is carried forward by verified image identity
and remains `needs_further_geometry_work`. This proves reproduction from versioned
code on development data, not another unseen-room success or gameplay validation.
Local verification: `build/room405-versioned-creative-verification/results.json`;
comparison: `build/room3d-versioned-creative-evaluation/index.html`. Generated
exports and original game assets are not included in source publication.

The evaluator accepts both legacy per-module source hashes and a complete
committed-tree `source_inventory`. Complete inventories verify auxiliary files
and reject changed, missing or extra files. Legacy hashes require actual archived
source files; a matching file from the current checkout cannot substitute for a
missing snapshot. The full ROOM405 report was regenerated with these checks.

### Experimental linear wall fixtures

A room recipe can opt into `linear_wall_fixtures: true`. The fitter adds
wall-mounted elongated housing candidates with one or two closed tube parts.
Training mask rays determine housing extents in a finite wall plane; ordinary
render agreement, finite plate contact, scene-depth and reserved-view checks
still determine acceptance. Tube count, cross section and hidden housing are
estimated design choices. The default remains disabled pending independent-room
validation. ROOM502 exposed the missing family and is now development data:
its original run accepted a nine-arm shade profile for a fluorescent assembly.
A better ROOM502 result alone will not establish generalization.

The first committed-source ROOM502 ON/OFF replay was negative: no eligible
linear initialization was generated. The available wall-plane projection of
both masks spans approximately 2874 by 1269 game units (aspect 2.26), failing
the generator's required elongation of 3. All four final rendered views are
byte-identical to the pre-change run and retain its incorrect nine-arm shade
family. This establishes abstention, not a native quality improvement. A
source-compatible orientation/support initialization and additional-room
validation remain necessary before enabling this option by default.

The initializer can additionally use elongated, parallel bright components
inside training masks when original images are available. This proposes a
housing extent without replacing the object masks used by fitting or changing
acceptance tolerances. Flat regions, round glints and fragmented highlights
abstain. Cropped components are recorded as partial extent evidence; brightness
is not proof of tube identity or emission. Only training cameras supply these
pixels, and the complete observed object and scene-depth gates remain active.

When at least two training cameras agree on the number of dominant bright
strips, that count initializes the linear family. Continuous fitting retains
each initialized count; it cannot silently collapse two observed strips into
one tube. This is a brightness-based structural hypothesis, not bulb identity
proof. Linear fixtures also export explicit support/tube material roles so
the existing local palette bake keeps their bright tubes and dark housing.

The native ROOM502 replay at source `a9cc663` exports two tube parts with
explicit shade roles. Actual final-mesh rasterization exposes 1130 and 1166
white shade pixels in its two training views. Explicit review of all four
source/OFF/ON views confirms that the earlier dark-material regression is
resolved, but the bright assembly is still thick and fine tube separation is
weak at native resolution. The other two final views are byte-identical.
This is a development-room result, not a whole-room visual pass.

A metadata-selected ROOM506 replay at the same source is an independent
negative control: all six final views are byte-identical with the option ON
and OFF. Explicit review shows round suspended source lamps, so this does not
validate positive fluorescent reconstruction. Selection excluded recorded
development room stems and exact cached RGB reuse without filtering fit scores
or acceptance. Neither selected lamp has a reserved observation. Both rooms
still need substantial geometry work; the option remains disabled by default.

### Vertical ladder geometry prior

`room3d_shapes.generate('ladder', parameters, origin, yaw)` generates two
vertical rails, regularly spaced open rungs and four wall brackets. Required
parameters are `width`, `height`, `thickness`, `stand_off` and integer `rungs`
(2..64). The base origin lies on the local wall plane Z=0; rails sit at
Z=`stand_off` and extend upward along negative game Y. Rungs intersect the
rails, brackets reach the wall plane, and gaps remain actual empty geometry.
Parts retain explicit rail/rung/bracket roles and export through the ordinary
room mesh path. Width, spacing and stand-off checks reject degenerate solids.

This is a reusable geometry foundation. Automatic ladder recognition, shared
wall placement, render fitting and independent native-room validation are
still required; the existing batch classifier does not select this family.
Bracket depth and unobserved parts remain estimated, and the generator adds
no climbing or navigation behavior.

`room3d_ladder_fit.proposals` adds training-only shared-rail hypotheses. Long
mask edges must agree with projected game gravity; matching rail planes from
two cameras triangulate a pair of world axes. A viewport-clipped rail may
extrapolate outside the image at the comparison row. Height uses the union of
training extents rather than requiring a partial view to reproduce the whole
ladder. With at least three observations the final view remains reserved and
cannot affect initialization. Ill-conditioned baselines, implausible widths
and heights, and inconsistent extra training axes abstain.

These proposals tolerate a filled SAM envelope in one image and visible rung
holes in another. They do not yet replace the batcher's stair fit: candidate
selection still needs actual render comparison, reliable scene depth, finite
wall/contact checks and independent-room validation. Rung spacing and the
inferred plane behind the rails are estimated; triangulation does not prove
that such a support wall exists.

`room3d_ladder_rungs.refine_count` compares actual rasterized rung edges with
original image edges inside the shared ladder's interior. It excludes rail
sides and favors edges parallel to projected rungs. Candidate counts preserve
the shared pose, rail dimensions and brackets. Selection requires a consistent
training improvement; when a reserved observation exists, missing or worse
reserved evidence vetoes the selected candidate without choosing another count.
Flat/insufficient training images abstain. All trials remain inspectable.

This refines an initialization, not an accepted room replacement. Reliable
whole-scene depth, finite support/contact and independent-room checks remain
necessary before automatic batch integration. Image edges can come from the
background or baked lighting and are not proof of individual rung identity.

The edge target uses local perspective direction rather than one averaged
orientation for the whole ladder. Short noisy fragments do not establish
rungs: coherent segments must span a substantial fraction of the projected
interior width. Controlled raster tests recover nine rungs from a seventeen-
rung initialization and confirm reserved-image rollback. Strong background
grain can still cause abstention. In the ROOM502 development control the
filtered training evidence is insufficient, so the seventeen-rung proposal
is retained exactly. This is a recorded limitation, not a native quality gain.

`room3d_ladder_spacing.refine_spacing` provides a separate structural cue when
image edges are weak. It rectifies a narrow central strip of each training
mask using the shared rail plane. Filled envelopes supply no count evidence.
At least three separated occupied bands with intervening holes must admit a
regular period; one omitted rung per interval is allowed and recorded. Hidden
rungs and total count remain inferred. Inconsistent counts across informative
training views abstain. With at least three observations, the final mask is
reserved and can only veto the training selection, without reselection.

Actual rung rasterization selects count and phase while keeping rails, brackets
and pose fixed. Optional `rung_phase` (-.45..+.45, default 0) shifts the regular
sequence by a fraction of its pitch while leaving positive end clearances.
This remains an initialization tool; scene-depth/contact checks and independent
native-room validation are required before batch replacement integration.

The frozen ROOM502 development replay at source `0114a7f` selects seven rungs
with phase -0.1 instead of seventeen. One open mask supplies three occupied
bands with gap multiples 1 and 2; the other training mask is filled and supplies
no count. Actual strip-raster agreement increases from 0.239 to 0.557. Explicit
four-view review shows less excessive rung density, and all-camera model-depth
checks pass. This is not a whole-room visual pass: brackets remain up to 667
game units from the current finite walls, so wall contact fails. Support and
independent-room validation still prevent automatic replacement integration.

`room3d_ladder_supports.anchor_brackets` adapts only four bracket wall caps and
links to nearby finite vertical surfaces. Rails, rungs and pose stay unchanged.
Every cap corner and center must lie within 10 game units of its assigned wall
group. A missing or excessively distant support causes exact rollback of the
whole attachment; partial attachment is not returned. Original surfaces are
not extended. The local `bracket_parts` parameter stores the four fitted closed
prisms so regenerating the shape preserves them. Invalid/nonfinite/zero-volume
overrides are rejected. Optional ordered `bracket_levels` define ordinary
bracket heights when no fitted override is present.

Wall-aligned pose alternatives are available separately; bounded bracket-height
adjustments retain finite cap contact without extending the wall. They still
need render/depth selection. Neither finite contact nor a short link proves
mechanical support or absence of intersections with provisional wall volumes.
Existing scene occlusion, source appearance and independent-room checks remain
necessary before a reconstructed room is accepted.

The frozen ROOM502 replay at source `17bbd5f` adapts all four brackets while
preserving rails, seven rungs and pose exactly. Finite cap distance is below
1e-12 game units, rail feet remain on the floor, and all four model-depth
checks pass. Explicit four-view review retains `needs_further_geometry_work`:
provisional volumes still occlude part of the ladder and contact does not
establish absence of intersections. Wall-aligned alternatives were also tried
but worsened close-view mask agreement. A fixed shared-rail scan of three new
metadata groups found no eligible positive ladder case; no independent positive
generalization or automatic batch integration is claimed.

### Opt-in batch vertical ladders

Room recipes can enable `vertical_ladders: true` with parametric details,
segmentation and depth inputs. After ordinary detail fitting, the batcher
associates staircase observations, proposes shared upright rails, estimates
open-mask rung spacing, and anchors brackets to finite existing walls. Filled
masks constrain the silhouette envelope; open masks constrain actual geometry.
Raw detector/SAM confidence remains attached to each source observation.

Candidate choice uses training masks only. The final observation is reserved
when at least three exist; reserved evidence and scene gates can veto the
fixed selection without choosing another candidate. Retention requires minimum
mask agreement, sufficient visible geometry, all-camera fixed-support depth
safety, finite bracket/floor contact and clear door volumes. Only inferred,
matched staircase primitives can be removed; script-owned geometry is retained.
Repeated execution detects an existing shared ladder instead of duplicating it.
Rejected groups leave the recipe unchanged.

Reports live in `vertical-ladders/results.json` and the exported fitted recipe
records the ladder, evidence and reproducible bracket parameters. The texture
baker recognizes ladder pixels through staircase/ladder semantic masks and
retains the foreground-fixture veto. Post-selection detail measurement includes
the actual exported ladder group. The option remains disabled by default;
retention is estimated reconstruction, not a whole-room visual pass, mechanical
support, independent positive generalization or navigation validation.

The frozen ROOM502 full-batch ON/OFF trial at source `dbebeaf` retains one
seven-rung ladder. All four fixed-support depth gates, finite floor/wall
contact, door clearance and observed visibility pass. Source archives match
the committed Git blobs; cached inputs and baseline exports remain unchanged.
Explicit inspection of all four final textured views still records
`needs_further_geometry_work`: provisional walls hide part of the rails and
rungs, with coarse bracket appearance and duplicated source ladder projection.
The no-ladder view is byte-identical. These gates are not a fidelity approval;
the option remains experimental and disabled by default.

The metadata-selected independent ROOM303 helipad negative control also
replays from source `dbebeaf`. It produces no supported shared-rail proposal
and retains the previous recipe: all eight final PNGs are byte-identical
between ON and OFF. Explicit eight-view review confirms no positive ladder
case and records substantial existing architecture/appearance defects.
Committed-source, cached-input and complete-checkpoint verification pass.
This proves safe abstention for this control, not positive generalization.

### Reconstructed object texture ownership

The opt-in recipe `reconstructed_object_appearance: true` prevents retained
shared vertical ladders from being projected a second time onto architecture.
Only exact camera/mask references belonging to present, retained ladder
primitives qualify. Finite detector and mask confidence must meet the existing
0.3/0.8 thresholds used for high-confidence object observations. An overlapping wall mask cannot override this exclusion.
Wall, floor, ceiling and door projection samples and filtered source textures
exclude owned pixels; inferred architectural patches also exclude them before
patch selection. Original images, geometry and raw model confidence are unchanged.
The object's own appearance eligibility is unchanged. Full and excluded-view
bakes and retouch reprojection use the same rule; excluded source cameras never
supply ownership evidence or material patches. Unobserved wall material remains
an estimated continuation, and filled object masks can exclude gaps between
rails. This cannot repair provisional wall occlusion or recover hidden art.
The option remains disabled by default; real-view comparison is described below.

The first frozen full-batch trial at `a72b415` changed no ROOM502 PNGs: its
0.4 detector cutoff incorrectly excluded a retained 0.3403-score observation.
The subsequent `0023b5e` replay aligns ownership with the existing high-confidence
object criterion (0.3 detector, 0.8 mask), without changing raw confidence.
Both source observations then qualify. Explicit four-view inspection finds
less ladder projection on architectural surfaces in cameras 0, 2 and 3;
camera 1 stays byte-identical. Fitted mesh files and rendered all-camera depth
are unchanged, and OFF output matches the earlier frozen ladder-on export.
This is not a global fidelity gain: the actual ladder remains gray and partly
occluded, so removal of painted duplicates exposes the underlying geometry
failure. The room retains `needs_further_geometry_work`. The earlier `a72b415`
ROOM303 negative control preserves all eight PNGs and was explicitly reviewed;
it is not a positive case or a new-room validation of the later policy revision.
Current code passes 212 room-tool tests. Further wall/object visibility fitting
and object appearance are required before enabling this option by default.

### Bounded wall retreat proposals

`room3d_wall_recess.proposals` supplies experimental geometry alternatives
behind retained vertical ladders. It considers only finite axis-aligned shared
wall runs named by the existing bracket attachment, with training cameras on
one side and ladder rails strictly inside the run's horizontal extent. Each
retreat stays within the original collision footprint, preserves the ladder
and adds finite side connectors at the old wall ends. Collision records and
neighboring walls are unchanged. Door-bearing groups, unshared volumes,
invalid geometry and opposing training cameras abstain. Roof geometry remains
unknown; the tool does not fabricate a ceiling.

These are proposals only, not accepted batch replacements. They require
training-render selection, reserved-view veto where available, fixed-support
all-camera depth checks and bracket refitting/contact checks before retention.
The proposal evidence explicitly marks the new wall location as estimated.

When a wall changes, `refit_support` regenerates only the procedural bracket
seed before finite attachment fitting. An already fitted oblique link cannot
be used as the new wall-normal prior. Rails, rungs and rail-end attachment
centers must regenerate unchanged; otherwise the function abstains. Failed
support leaves the original object intact. Initial ROOM502 trials from
`19a3de5` could not refit any brackets using their old oblique shafts; this
failure is retained as development evidence, not a rejected depth experiment.

Optional `local=True` limits the retreat span to the retained rail/rung bounds
plus two rail thicknesses of margin on each side. It retains outer wall strips
on their original plane and connects the inset to them, avoiding an unsupported
whole-wall translation. This remains a proposal requiring the same scene vetoes.
The whole-run ROOM502 experiment at `d259f0e` chose a 270-unit retreat on training
renders but failed fixed-support depth checks in three cameras; it was rejected
without reselection. The original completed room export remains unchanged.

The local-span frozen ROOM502 experiment at `d3e2d31` selects a 270-unit
retreat using training renders, then passes unchanged fixed-support depth
tolerance (+0.005) in all four cameras. Visible ladder fraction rises from
0.548 to 0.852 and from 0.617 to 1.000 in its two observed views; these are
visibility fractions, not fidelity percentages. Rails/rungs and rail-end
attachment centers remain unchanged. Finite bracket contact is verified on
the actual final baked mesh, rail feet remain on the floor, all 162 archived
tool files match their Git blobs, and original export/input hashes remain
unchanged. Explicit four-view final texture inspection still records
`needs_further_geometry_work`: gray ladder/brackets, smeared wall continuation,
missing upper architecture and distorted cabinet appearance persist. This
is a development diagnostic, not automatic batch integration, independent
positive validation or runtime/navigation approval.

### Opt-in batch object/wall fitting

Room recipes may enable `object_wall_fitting: true` with parametric details,
segmentation and depth. After vertical ladder reconstruction, the batcher runs
versioned local wall proposals and training-render selection before texture
baking. Only retained shared-rail interpretations with at least two referenced,
high-confidence source views qualify. Multiple referenced masks in a camera
are combined, preserving their original confidence and pixel hashes.

The last observation is reserved when there are at least three. Selection
uses mean visible-mask agreement plus a visibility term, with per-view
regression limits and a required visibility gain. Once chosen, a candidate
can only be vetoed by the reserved view, fixed-support all-camera depth,
finite wall/floor contacts or detachment of an existing supported ladder or
lamp. The fitter never tries another candidate after a veto. Failed objects
leave the prior recipe unchanged. A retained object is marked to prevent
repeated wall retreats. Script/manual geometry is not selected for this fit.

`object-wall-fitting/results.json` records trials, the frozen training choice
and every veto; `after-object-walls-recipe.json` contains retained geometry.
Post-selection measurements and appearance baking use the final mesh. These
reports are separate from the earlier ladder and ordinary detail fit reports.
The option is disabled by default. Estimated visual wall location, model
agreement and contact do not establish original fidelity, hidden geometry,
mechanical support, independent generalization or navigation.

The frozen full-batch ROOM502 ON/OFF replay at source `13dfbae` retains one
local wall fit and reproduces all four final RGBA views of the earlier
`d3e2d31` local-retreat diagnostic exactly. All source files match frozen Git
blobs; input hashes and original exports remain unchanged. OFF output matches
the prior ownership-on full export. Finite contact is verified on the actual
final baked mesh; rail floor distance remains zero. Explicit four-view review
retains `needs_further_geometry_work` because gray ladder/brackets, smeared
local wall appearance, incomplete upper architecture and stretched cabinet
projection persist. The separate previously reviewed ROOM303 negative control
has no eligible ladder: all eight ON/OFF final PNGs remain byte-identical, with
explicit eight-view review. This is safe abstention, not independent positive
generalization. All 223 room-tool tests pass; the option stays disabled by default.

### Local ladder material

The opt-in `local_ladder_appearance: true` recipe requires segmentation and
`reconstructed_object_appearance`. The baker intersects each retained ladder's
own high-confidence source masks with its actual visible raster geometry.
Known foreground fixtures are excluded even inside a filled staircase mask.
Each eligible view supplies a source-pixel median; their equal-view median
becomes a uniform material. Raw detector/SAM confidence, shape, placement and
wall fitting remain unchanged. The reported tint is baked appearance, not
recovered albedo, metallic reflectance, texture detail or lighting. Unsupported
objects retain their previous appearance.

Full and excluded-camera bakes use the same sampler; excluded source cameras
cannot contribute palette evidence. Already baked ladder materials survive
repeated retouch reprojection without the original masks, alongside existing
fixture materials. Preserved material cannot be used in heldout appearance
evaluation. Reports expose sampled visible pixels, confidence and sample-mask
hashes. Hidden material appearance remains estimated. The option stays disabled
by default pending real-room and independent validation.

The frozen `9513173` ROOM502 fixed-mesh comparison estimates RGB `[22,24,24]`
(on the 0–255 scale) from 612 and 1876 own visible source pixels in cameras
0 and 2. Detector/SAM confidence stays unchanged. All four rendered depths
and silhouettes are identical ON/OFF, and color changes are confined to visible
ladder geometry. Excluding either observed camera leaves only the other camera
as palette evidence. Explicit four-view inspection finds a more recognizable
dark ladder, but uniform tint does not recover highlights, wear or reflectance;
coarse brackets, smeared wall continuation, missing upper architecture and
distorted cabinet projection remain. The previously reviewed ROOM303 negative
control has no eligible ladder and keeps all eight final images unchanged.

A separate full-batch ROOM502 run from the same frozen source reproduces all
four reviewed material-ON RGBA images exactly. Its fitted geometry and cameras
remain byte-identical to the previous object/wall-fitting export. Git tool
blobs, frozen input hashes and completed original exports were verified.
All 227 room-tool tests pass, including three successive retouch reprojections
and excluded-source sampling checks. Both room reviews retain
`needs_further_geometry_work`; two object observations provide no third
reserved view, and neither comparison establishes independent positive
generalization, whole-room fidelity or runtime/navigation approval.

A metadata-first recheck at `ff2a80d` uses the current observation, association
and rail-initialization code on 13 additional cached scenario-0 room entries
with high-confidence staircase masks in at least two cameras. Previously
reviewed room stems and exact reused RGB hashes are excluded before geometric
analysis. Of these entries, two have associated multi-view staircase groups;
neither initializes a valid vertical ladder. Enumerating previously unassociated
camera pairs with the same unchanged rail predicate also yields no proposal.
This does not justify relaxing association or rail-size gates: most observations
lack a gravity-compatible rail pair, while one shared group exceeds the current
ladder width prior. The diagnostic freezes Git tools and verifies 992 input
file hashes before and after analysis. It leaves completed exports unchanged
and supplies no independent positive validation. Cached detector proposals and
exact-image exclusions do not establish unseen training data or perceptual
independence; ordinary staircases still require their separate shape generator.

### Estimated elongated ceiling lights

The `ceiling_linear` lamp profile adds a closed mounting plate, two suspension
links, a housing and one to three separate tubes. Its parameters are `height`
(ceiling-to-tube-underside drop), `radius` (half housing length), `housing_width`,
`thickness`, `shade` (tube size prior) and `arms` (tube count). Positive suspension,
elongated housing and separate tubes are required. Explicit support/shade roles
work with the existing `local_fixture_appearance` mesh export option.

`tools/room3d_ceiling_linear.py` proposes attachment only to supplied finite
horizontal ceiling groups. All mounting corners and their center must contact
the same group; a center-only contact is insufficient. A bounded upward extension
changes only the mounting plate and suspension, retaining housing/tube vertices,
pose and source evidence. It does not fabricate missing ceilings. Hidden supports,
tube count and roof height remain estimated design choices. This is a geometry
and finite-contact foundation: automatic image fitting, final scene gates,
native-room comparisons and independent positive validation are still required
before enabling automatic replacement.

Four actual-geometry tests cover closed exported parts and material roles,
unchanged rendered depth under role coloring, rotated mounting contact,
center-only/far/sloped/wrong-group rejection, unchanged visible body under
extension and invalid dimensions. All 231 room-tool tests pass. ROOM512 was
explicitly inspected as development data for this category. Its current fitted
recipe does contain a ceiling box (`primitive_1`), estimated from detector boxes.
The earlier claim that it had no ceiling inspected only `structural_surfaces`
and missed that primitive. The actual exported finite underside is available
for attachment; its height remains estimated rather than original geometry.

`ceiling_light_fixtures: true` adds bounded ceiling-light initializations to
the existing shared object fitter. Only training masks intersect the finite
ceiling plane; inferred elongation, pose and dimensions initialize one- and
two-tube design alternatives at four suspension drops. A separate compact
`ceiling_panel` profile provides a rectangular housing with one inset light
face for masks that do not establish elongated tube geometry. These are design
alternatives, not recovered light types or verified component counts. The final observation
is reserved when there are at least three. Existing silhouette, raw depth,
support and final attachment checks still decide retention, without using a
reserved view to select another candidate. Ceiling attachment remains finite
through continuous fitting, and its contact cache includes the actual mounting
plate rather than a suspension link. The option is disabled by default and
requires native-room and independent positive validation.

Four additional actual-render tests cover panel dimensions, finite proposal
contact, default-off behavior, unchanged training initializations and selected
shape when the reserved mask changes, and contact-cache rejection when a wider
mounting plate leaves its finite roof although its suspension link is unchanged.
All 235 room-tool tests pass. These tests verify fitting/attachment invariants;
they do not establish native-room visual improvement or positive generalization.

The frozen `71fa33a` full-batch ROOM512 ON/OFF pair retains one compact panel
attached to the actual finite ceiling underside (`primitive_1`). Measurement
on the final baked geometry raises its two source-mask IoUs from 0.898/0.910
to 0.922/0.926. Final images change by 655/783 pixels in cameras 0/1; camera 2
remains RGBA-identical. Explicit inspection of all three source/OFF/ON images
finds a cleaner shared rectangular silhouette than the fragmented OFF shade,
but its flat pale face loses tube highlights and frame detail. Distorted door,
sink, rail appearance and gray side geometry remain. Both bound reviews retain
`needs_further_geometry_work`. Raw-depth gate scores stay unchanged although
actual rendered depth changes in cameras 0/1; that gate does not independently
recover the light's depth. Git blobs, frozen inputs and completed checkpoints
were verified without modifying original exports. This development room has
two object observations, no reserved object view and no independent positive
validation; the ceiling prior itself used detector boxes in all room cameras.

The separate metadata-selected ROOM715 pair at the same frozen source completes
without initializing or retaining a ceiling-light candidate. All eight final
RGBA images and rendered depths remain ON/OFF-identical. Explicit inspection
of all eight source/OFF/ON views retains `needs_further_geometry_work`: angular
pale fixtures, missing furniture, stretched walls and gray obstructions persist.
This is independent negative evidence, not positive ceiling-light generalization.
Selection preceded first explicit source-image inspection, although cached
proposals had previously undergone linear-highlight eligibility checks. Four
camera pairs have identical origin, target, focal length and roll but differently
lit backgrounds. Their final renders are identical within each pair as well;
there is no camera-dependent cutaway mismatch. Such lighting variants must not
be counted as independent geometric viewpoints.

```sh
python -m unittest discover -s tests -p test_room3d_ceiling_linear.py
```

Portable overrides and replay
-----------------------------

`tools/room3d_overrides.py` stores manual corrections as a portable, versioned
override file instead of a one-off retouch directory. The file names the room
by RDT stem, stage, room number and scenario player; fingerprints its inputs
(RDT SHA-256, measured camera/collision layout, background image hashes); and
freezes the generator identity (tools digest, boolean recipe settings, baseline
mesh hash). Each corrected object is recorded by kind, script `source_offset`
when the batcher anchored it to the room script, and bounds, never by the bare
`primitive_N` index alone. The history is an ordered list of steps, each with
the editor's transforms, explicit parametric additions and the identities those
transforms referred to. Files contain no machine-local paths; validation rejects
them. The tracked copies live in `room3d/overrides/` with `room3d/status.json`.

```sh
python tools/room3d_overrides.py init --room out/room1180 --out room3d/overrides/ROOM1180.json
python tools/room3d_overrides.py add --overrides room3d/overrides/ROOM1180.json \
    --room out/room1180 --edits edits-room1180.json --additions additions.json --reproject
python tools/room3d_overrides.py replay --overrides room3d/overrides/ROOM1180.json \
    --room out/room1180 --out corrected/room1180 --segmentation seg/room1180 --depth depth/room1180
python tools/room3d_overrides.py check --overrides room3d/overrides/ROOM1180.json --room other/room1180
python tools/room3d_overrides.py compare corrected-a corrected-b
python tools/batch_room3d.py ... --reconstruct --overrides room3d/overrides
```

Replay always starts from a fresh automatic export and writes a separate
directory, so transforms cannot accumulate twice. A step is recorded against
the automatic export only while the history is empty; later steps must be
recorded against the replay of the current history (`replay.json` carries the
history hash), and recording refuses an object that another object of the same
kind within the tolerance could be confused with. During replay, script-anchored
objects resolve by offset, structural groups (`floor`, `boundary_N`, `added_*`)
by stable name with unchanged bounds, and other objects by kind and bounds
within 50 room units. Zero or several candidates, a changed structural group or
an export of a different room layout are reported as conflicts and the replay
fails without output unless `--allow-partial` is given, in which case resolvable
steps are applied and the rest listed. A scenario variant with the same layout
is accepted and reported as `scenario_variant`; differing backgrounds are noted.
An explicit pivot moves with the object when its bounds relocated.

The batcher's `--overrides DIR` replays `DIR/<ROOM>.json` into
`<out>/corrected/<room>/` after the automatic export or its resumed checkpoint,
using the batch segmentation and depth roots for reprojection when the file
asks for it. The automatic directory and its checkpoint are unchanged; the
override directory is excluded from the batch key, so editing a correction does
not invalidate generation. Conflicts appear in the manifest and index page.

ROOM1180 proof on 2026-10-14 (local, `build/room3d-overrides-proof/`): the
room was regenerated twice from the original game files with cached detector,
mask and depth evidence (22 s and 16 s); both `assisted.obj` exports are
byte-identical. One recorded step adds a wall sconce where camera 1 shows the
lamp and raises the small barrel by a factor of 1.5; replays onto both exports,
with reprojection, are byte-identical to each other (20,532 vertices) and
differ from the automatic export only in `primitive_4` (bounds delta 289.5) and
`added_wall_lamp`. The batch hook on the resumed export produced the same bytes.
The corrected render is an approximation: the gray sconce is small and the
rest of the room keeps its automatic defects, so the room is tracked as
`needs_work`, not approved, and gameplay is `not_validated`.

```sh
python -m unittest discover -s tests -p test_room3d_overrides.py
```

Nine actual-mesh tests cover portability (no local paths, room identity),
replay equal to a direct retouch and repeatable byte for byte, UV preservation
without reprojection and deterministic reprojection, re-indexed and nudged
objects resolved by identity, missing/ambiguous/changed/source conflicts with
no output written, partial replay, scenario-variant acceptance, sequential
steps composing from the replayed state only (refusing stale bases), record-time
ambiguity refusal, and the batch hook keeping automatic and corrected outputs
distinct. They do not establish visual fidelity or gameplay.

### Room catalog and review status

`tools/room3d_rooms.py catalog` reads every RDT directly, fingerprints its
layout and background files, and groups scenario pairs. Over the USA assets:
348 RDT files, 28 empty slots, 116 unique layouts, and every one of the 320
non-empty rooms has a scenario variant with the same layout and backgrounds,
so one override file can serve a pair (replay still checks the fingerprints).
`room3d/status.json` tracks `review` (`unreviewed`, `needs_work`,
`visually_approved`) separately from `gameplay` (`not_validated`, `validated`,
`broken`); `tools/room3d_rooms.py status ROOM1180 --review needs_work` updates
it and unknown states are rejected. The catalog page lists links to automatic
and corrected pages when a batch root is given. Shared layout is a measured
identity, not a fidelity or playability claim.

```sh
python tools/room3d_rooms.py catalog --assets assets/USA --out build/room3d-rooms \
    --overrides room3d/overrides --status room3d/status.json --batch build/room3d-all-generated
python -m unittest discover -s tests -p test_room3d_rooms.py
```

### Layout variants, shared overrides and the block family

Stages 6 and 7 revisit stage 1 and 2 rooms with the same measured layout and
the same background files (ROOM6180 is ROOM1180 again), but their script
offsets differ, so `source_offset` anchors identify objects only within one
RDT. Replay therefore accepts a `layout_variant` whose layout and background
hashes both match and resolves anchored objects by kind and bounds there; a
room with the same layout but different images is still a source mismatch.
When a batch room has no override file of its own and exactly one file in the
override directory matches its layout and images, that file is replayed and the
manifest records `shared_from`. The catalog lists `shared_layout_with` across
groups and marks corrections shared this way. Over the USA assets the 320
non-empty rooms have 116 unique layout-and-image combinations, so three override
files currently cover eight rooms.

`room3d_additions.py` accepts a `block` family (width, depth, height, optional
elevation) for furniture-sized stand-ins; it is an explicit manual estimate.

First manual pass, 2026-10-14, over the sixteen rooms with one or two cameras
(`build/room3d-rooms-pass1/`, regenerated with cached evidence in about 20 s
per room): two rooms were accepted as approximate (ROOM4100, ROOM50E0), the
rest are tracked as `needs_work` with specific notes in `room3d/status.json`.
Recurring defects: 1.5 m collision slabs standing in for furniture (bathrooms
ROOM4020/4070/40B0), furniture merged into wall-height slabs (ROOM1120 bed),
single-camera duct views that cannot be reconstructed (ROOM50D0, ROOM7000), and
whole-room distortion that needs reconstruction rather than retouch (ROOM10D0).
The reusable correction found so far is scaling a furniture collision slab to
its real height about the floor, which ray casts from the cameras can locate:
ROOM4070 and ROOM40B0 record it, and the reprojected texture lands on the slab.
No gameplay validation was performed for any room.

### Second pass: every unique layout reviewed

On 2026-10-14 one representative RDT per unique layout-and-image combination
was regenerated with cached evidence (`build/room3d-rooms-pass2/`, 103 rooms,
four workers, 80 minutes; ROOM3100 fails because no cached DA3 depth exists for
it, as in the original batch). Each room's original and automatic views were
inspected side by side and the result recorded with `room3d_rooms.py status`.
Together with the first pass that covers all 116 unique layouts: 47 rooms are
accepted as approximate reconstructions and 73 need work, with the defect named
in the note. `room3d_rooms.py propagate` then gives each unreviewed room the
status of its reviewed sibling with identical layout and background files,
marked `inherited_from` and "not reviewed separately"; gameplay validation is
never inherited. The automatic exports of such siblings are close but not
byte-identical (ROOM1180 versus ROOM6180 differ by 0.22 and 0.09 mean
intensity per view, because cached model outputs and script anchors are per
RDT), so an inherited status is a presumption to confirm when that room is
replayed. Corrections recorded in this pass: ROOM1050 (dining table slab
scaled to table height) and ROOM20F0 (table block on the skipped collision box).

Recurring defects across the 73 rooms, in order of frequency: furniture
rendered as bare 1.5 m collision slabs or merged into wall-height slabs;
misplaced gray slabs from collision boxes that are not visible walls; cameras
that render nothing or black (close-ups, exteriors, duct views); caves and
waterways (ROOM3070, ROOM30B0, ROOM30F0) that produce no usable geometry; and
chandeliers or lamps scattering into fragments. The reusable manual fix remains
scaling or adding a slab or block at the furniture's collision footprint; the
other defects need geometry work beyond the retouch tools.

### Runtime smoke tests on corrected meshes

The root checkout's `tools/check_room3d_runtime.py` drives the Wine build with
scripted input and only passes when the room mesh loads, a camera transition
happens and, for ROOM1180, the chest menu opens and closes. On 2026-10-14 the
replayed ROOM1180 correction (`build/room3d-overrides-proof/auto-b/corrected/`)
passed in 49 s and the automatic ROOM10B0 export passed in 43 s
(`build/room3d-runtime-corrected-1180/result.json`,
`build/room3d-runtime-auto-10b0/result.json`, executable
`build/windows-cross/residentevil.exe`). Both rooms are recorded as
`validated` with the scope in the note: an automated smoke test, not a human
play-test, with no visual, item or save assertions. No other room has a runtime
scenario; `propagate` never copies gameplay status.

### Furniture slab corrections, third pass

Five more rooms whose only recorded defect was furniture rendered as 1.5 m
collision slabs received the slab-scaling correction (beds to 0.4 of the slab
height, desks and dressers to 0.5 to 0.7, pivot on the floor at the slab
centre) with reprojection: ROOM4010, ROOM40A0, ROOM4060, ROOM5070 and
ROOM2090. All replayed through the batch hook; the first four now read as
furniture with the original texture on the lowered top and were moved to
`visually_approved` (approximate); ROOM2090's bed tops come out partly gray.
`propagate` now refreshes inherited entries when their source room changes.
Catalog: 136 approved, 184 need work, 28 rooms covered by 11 override files.

Fourth pass, same correction: ROOM1080 (display cases, cabinet, desk),
ROOM1110 (sofa, side table), ROOM7160 (reading table), ROOM4020 (tub, toilet,
sink), ROOM7190 (chair) and ROOM1120 (dresser). Slabs were mapped to objects
by projecting each slab's top centre into the cameras and reading the
original image at that pixel. ROOM1080, ROOM4020 and ROOM7190 moved to
approved (approximate); the other three keep defects the slab fix cannot
reach (globe fragments, misplaced slabs, a bed merged into a wall slab).

Fifth pass, same correction: ROOM4040 (billiard table, bar, barrel), ROOM1130
(tub, stool, stand), ROOM71C0 (kitchen table, counters), ROOM6190 (desk,
chair), ROOM30E0 (trunk, desk) and ROOM5060 (counters, desk, cabinet). The
first four moved to approved (approximate); ROOM30E0 and ROOM5060 keep a
fragmenting typewriter and a room-splitting wall slab. After this pass 154
rooms are approved and 166 need work; 23 override files cover 60 rooms. The
remaining needs-work rooms have no furniture slab left to scale: their
defects are black or empty cameras, caves, stairs and scattered fixtures.

### Hiding scattered fixture primitives

A lamp or chandelier primitive that scatters into white fragments reads worse
than no fixture, so the `hide` edit is the right correction when the original
fixture cannot be rebuilt. Applied on 2026-10-14 to ROOM10E0 (three table-lamp
primitives), ROOM10B0, ROOM2010, ROOM2080 and ROOM1160 (chandeliers), ROOM1130
(ceiling fan, as a second override step recorded against the replayed state)
and ROOM1110 (the globe, a creative removal). ROOM10E0, ROOM1110, ROOM10B0 and
ROOM1160 are approved as approximate; ROOM2010 and ROOM2080 keep flat stairs
and an unreconstructable deer head. The corrected ROOM10B0 mesh passes the
Wine smoke test (`build/room3d-runtime-corrected-10b0/result.json`), so its
gameplay record now refers to the corrected mesh. Catalog after this pass: 166
rooms approved, 154 need work, 80 rooms covered by 30 override files.

### 15-bit colour decode fix

Play-testing ROOM10B0 against the game showed tan walls and orange parquet
where every extracted background and reprojected texture was blue. The PC
background TIMs, like PSX TIMs, keep red in the low five bits and blue at bits
10-14 (see `src/game/TextureLoader.cpp`). `tools/pak_view.py` and
`tools/dump_tim.py` read them the other way round, so until this fix every
`camera*-original.png`, every reprojected texture, and every detector,
segmentation and depth run in `build/room3d-all-generated` saw red and blue
exchanged. `tools/bss_to_pak.py`, `tools/dump_ivm.py` and
`tools/dump_item_pix.py` already used the correct order.

Consequences: geometry fitting is largely hue-independent, but the cached
vision outputs are pinned to the pixel hash of the swapped images, so a room
regenerated with the fixed decoder must rerun the local models
(`--local-models --vision-python build/room3d-vision-venv/bin/python`) or the
batcher refuses the stale cache. Visual approvals in `room3d/status.json` were
made against the swapped colours; they judged layout, not hue, and stand until
a room is regenerated and looked at again. The handmade ROOM10B0 palette below
was set from the corrected colours.

### Hand-authored ROOM10B0

`tools/build_room10b_mesh.py` writes the 1F right-stairs room from code, with
no batcher involvement: the walkable outline comes from the RDT blocking
rectangles, and the stair, hall height, furniture, materials and lighting are
an authored interpretation inspired by the four views. Textures are procedural
(numpy/PIL) and written beside the mesh; lighting is baked into vertex colours
because the Room3D loader has no lighting, and every large surface is cut into
tile-aligned cells so UVs stay inside 0..1 under the loader's clamped sampler.
The chandelier ring was fitted to the four globe sprites the game overlays in
camera 2 (1.5 px rms), since the engine draws the original foreground masks
over the mesh. `--rdt` renders `preview-camera*.png` through
`room3d_preview.py` for comparison against channel-corrected originals.

```
build/room3d-tools-venv/bin/python tools/build_room10b_mesh.py \
    --out build/room3d-handmade/room10b0 --rdt assets/USA/Stage1/ROOM10B0.RDT
```

The result passes `tools/check_room3d_runtime.py --room 0x0B` and was
play-tested in the Wine build on 2026-10-14: all four cameras, the stair
trigger into ROOM2070 and the return.

### Hand-authored ROOM1060 and the shared module

`tools/room3d_handmade.py` now holds the mesh writer, tile-aligned cell
splitting, baked-lighting hook, balustrade/door pieces, texture noise helpers
and the preview/compare utilities; `build_room10b_mesh.py` was rewired onto it
with byte-identical output. `tools/build_room1060_mesh.py` builds the main
hall from ROOM1060.RDT's collision: the side colonnades (type-3 rectangles),
the north arcade piers, the central stair block with its parapet strips, and
the entrance and side doors from the door records. Storey heights, arches,
the gallery, the fanlight and all materials are authored. The mesh is cut at
1500-unit cells to stay under the loader's 4 MB OBJ limit (3.7 MB, 23971
triangles). `tools/check_room3d_runtime.py --room 0x06` passes.

Evidence used for the hall after play-testing: the mask sprite tables of
ROOM1060 (floor-cast through each camera to place columns, the stair foot
posts and the gallery rail) and the 2F hall ROOM2030 (its collision gives
the gallery depth, void edge and stair top; its views show no gallery
columns, carpeted galleries and one arched north window). `room3d_handmade.py`
now reports a per-camera edge-agreement score between preview and original
(precision/recall of edge pixels within 3 px) so layout regressions show up
before a play-test.

### Hand-authored ROOM1180

`tools/build_room1180_mesh.py` rebuilds the storeroom on the shared module:
walls from the blocking rectangles, the chest body (lid stays game-owned),
the yellow drum, boxes and tins, the typewriter table and the wall lamp.
Object positions were cast from the original views onto the wall planes
(floor casts carry a depth bias in this room, so walls follow collision).
Camera 1 sits behind the north wall, so that wall, the door and the table are
hidden for it; the drum is hidden for camera 0. Both cameras score above 0.95
edge precision. Passes the smoke test including the item-box menu.

### Room readings

`room3d/readings/ROOMxxxx.toml` records what a person read from a room's
evidence: walkable zones with floor and ceiling heights and finishes, doors,
windows, paintings, arcades, slabs (galleries, landings), stairs, objects,
sconces, chandeliers, lamp posts and per-camera hides. `tools/room3d_reading.py`
builds any reading into a mesh through `room3d_handmade.py` with procedural
textures (families with per-reading tints) and a light rig derived from the
fixtures; it checks the loader limits and prints per-camera edge scores.
`room3d_reading.py survey ROOM.RDT --originals DIR` prints the evidence used to
write a reading (cameras, collision, floor-cast mask sprites) and writes the
corrected originals with a pixel grid. The three hand-authored rooms are now
readings (ROOM1180, ROOM10B0, ROOM1060) and reproduce the approved meshes;
the per-room build scripts remain as reference.

### Readings for every room

`room3d/readings/` holds a reading for all 160 non-empty RDTs: stages 1 to 5
read from their evidence (survey dump, gridded corrected views, collision,
door records); stages 6 and 7 are cloned from 1 and 2 where the RDT exists in
both (identical collision and cameras, regenerate with the clone script after
editing a stage 1/2 reading), and the 16 return-visit rooms with no stage 1/2
twin (ROOM6100, ROOM6190, ROOM7000, ROOM70C0, ROOM7130 to ROOM71C0) plus the
event rooms ROOM3100 and ROOM3110 are read on their own. The return stages
ship no backgrounds; the survey and the batch read stage 1/2's.
`tools/room3d_readings_batch.py` builds them all, scores each camera, installs
the meshes into a mod tree with room-prefixed textures and, with
`--runtime-exe`, loads every room in the Wine build through the generalised
smoke test; it writes `report.md` and `report.json`, and `--merge` keeps the
entries of rooms not rebuilt in a partial rerun.

Library additions made while reading the last rooms: `bookcase` (wooden
carcass with book-spine faces on any sides), `ladder`, `trophy` (mounted deer
head, scalable), rectangular `rect` windows with mullions and a transom,
skirted round tables (`cloth`), `books` and ceramic `tiles` texture families,
and `peel`/`ivy` options on plaster for the ruined rooms. Fireplaces on east
and north walls used to draw their opening flush with the mantel face (hidden);
it now sits just in front.

Wall-mounted things (doors, windows, paintings, sconces, wall lamps, wall
bands, trophies) take an optional `line` naming the wall's coordinate when the
outermost wall at that span is not the one meant; paintings take `group` for
per-camera hides.

A reading can inherit another one shifted vertically with
`base = { reading = "ROOMxxxx", y = dy }`: the base's vertical keys (floors,
ceilings, tops, sills, springs, light and sconce heights...) move by dy, its
hides and backdrops are dropped, then the reading's own tables override and
its lists extend the base. ROOM2030 (main hall 2F) inherits ROOM1060 moved
down 6000, since both RDTs share one frame; its cameras now look into the real
hall instead of a walled shaft.

A reading's `backdrop = [N, ...]` writes `backdrop N` lines into the room's
`.cameras` file and the runtime keeps the original pre-rendered background
for those cameras (`Room3D_IsActive` returns false). It is used for close-ups
and cutscene views that a room mesh cannot stand in for: the keypads in
ROOM2010 camera 7 and ROOM4080 camera 5, the aerial shot in ROOM7180 camera 2,
and the event rooms ROOM3100 and ROOM3110 (all cameras). The batch does not
flag backdrop cameras as weak.

Placing furniture: `tools/room3d_evidence.py blocks ROOMxxxx --originals
DIR --out PNG` draws every collision block's floor outline and number on each
original view, and `floor ROOMxxxx CAMERA PX PY` gives the room point under a
pixel. Readings written by eye from the views were mirrored or misplaced in
many rooms (bathrooms, bedrooms, the guardhouse bar, the heliport); every
reading was re-checked this way on 2026-10-15.

Character variants: every ROOMxxx1.RDT (Jill) has the same cameras and
collision as ROOMxxx0, and the runtime asks for `roomSRRV.obj` with the
variant digit, so the batch installs each mesh under both names (the Jill
copy reuses the Chris material library). `tools/check_room3d_runtime.py
--character jill` exercises that path.

Later kinds and options: `pipe` (cylinder between points, with `via`),
`ring` (annular rim with an optional fill), `spiral_stair`, `globe` (sphere or
egg via `stretch`), `tub`/`toilet`/`sink`, `plant`; zones take `open_sides`,
`dado`, `dado_material` and `chair_rail`; the shell takes
`skirting_material`. A reading's `[shade]` table ("camera" = factor) writes
`shade N F` lines and the runtime scales the baked colours for those cameras,
for lights-off camera sets over the same geometry (trophy room, 1F study, the
B2 room, the small dining room).

Review status 2026-10-15: every room was re-read against its originals with
the evidence tool, the full batch rebuilt and loaded all 160 rooms, then two
independent reviews of every compare sheet found 28 rooms with visible
problems; all were fixed, rebuilt and loaded again (160/160 built and loaded,
no weak cameras). 86 close-up or cutscene cameras keep their original
backgrounds through `backdrop`.

Caveats: outdoor terrain is simplified (open zones with water surfaces,
walls, beds and blocks); painted lettering and stencils are not modelled;
furniture is assembled from primitives; one steep camera in the closet study
(ROOM7190 camera 1) disagrees with its own background, which was rendered
from a slightly different viewpoint than the stored camera.

### Original backgrounds projected onto the mesh

The procedural materials only approximate the rooms; the fixed-camera
backgrounds are the real art, and HD or AI-upscaled packs replace exactly
those images. `tools/room3d_projection.py` paints the mesh with them:

- Every face is clipped against the frames of the room's cameras (backdrop
  and `shade` cameras excluded) and halved where something occludes part of
  it, down to pieces of about 150 units. Each piece takes the camera that
  sees all of it, unhidden in that camera's depth map, at the highest texel
  density; pieces seen nearly edge-on (|cos| < 0.2) by every camera, and
  surfaces no camera saw (most ceilings, the walls behind the cameras), keep
  the procedural material.
- The split mesh replaces `roomSRRV.obj`; the builder's output stays as
  `roomSRRV.base.obj` (the split file starts with a marker line, so a rerun
  splits the base again). `roomSRRV.proj` holds `proj1 <faces>`, one
  character per OBJ face (`0`-`9`, `a`-`v` the camera, `-` none), and
  `tint <material> r g b` lines: per material, the art's colour over the
  procedural colour on its projected pieces (a brightness factor with a
  bounded hue shift), applied to the unprojected pieces so they do not stand
  out as a different wallpaper or floor.
- The batch runs it after scoring each room (`--no-projection` skips it) and
  installs the `.proj` under both character names.

At runtime (`Room3D.cpp`) a room with a valid `.proj` loads every in-room
camera's background through the normal asset resolution: the mod's
`StageN/RCSRRC.webp` when one resolves (any installed pack, e.g. the HD
backgrounds), otherwise the original PAK's TIM, with the same revisit-stage
fold as `load_room_bg`. Projected triangles get UVs from where their corners
land in that camera's 320x240 frame and white vertex colour (the lighting is
in the image). The modern camera draws them with bilinear filtering; a fixed
camera draws every face with its own background, mapped by screen position,
so it shows the original image exactly, with the mesh's depth behind the
actors. `RE1_ROOM3D_PROJECTION=0` ignores the sidecars.

The faces no camera saw would otherwise keep the generic procedural
textures next to sharp original (or HD) art. They repeat a square of the
room's own art instead (an exemplar):

- Unseen pieces are grouped by wall (material and plane). A wall larger than
  1 m^2 (1e6 square units) takes its exemplar from its own projected part,
  even when a camera sees that wall at a slant, so the pattern's scale and
  colour carry on across the edge with the projected part. Smaller surfaces,
  and walls no camera paints, use the material's best square anywhere: seen
  square-on (|cos| >= 0.5) and sharp.
- A square (350 to 1600 units, aligned to the wall's horizontal axis and up,
  or x/z on floors) may span several of the builder's ~600-unit faces. Every
  sample of it must be inside the camera's frame and on the surface that
  camera's depth map recorded for that material alone (a per-material depth
  render), so it never takes in a neighbouring door, sconce or painting.
- Squares whose art does not look like the material are rejected: hue
  more than 0.08 (chromaticity L1) from, or brightness more than 2x off, the
  material's area-weighted median over its projected pieces. This catches
  spots where the reading and the art disagree (ROOM1090's first wallpaper
  exemplar sat on a door painted where the mesh has wall).
- The tiled vertices are lit from the art: each takes the colour of the
  nearby projected pieces of its material (inverse square distance, 600-unit
  softening), fading to the material's median 4000 units away, over the
  exemplar's colour. These factors are baked into the split OBJ's vertex
  colours, scaled into 0..1 by the patch's colour factor.

`patch <index> <material> <camera> <side> <origin> <a> <b> <r g b>` lines
record the exemplars (at most 62 per room) and `tiles <string>` the patch of
every face (`0`-`9`, `a`-`z`, `A`-`Z`, `-` none). At load the game resamples
each square from its camera's background as loaded (so from the HD image
when a pack provides one), divides out its large-scale lighting, mirrors it
into a seamless 2x2 tile and box-maps it over its faces with the bilinear
wrapping sampler (`MARNI_SAMPLER_LINEAR_WRAP`). Materials no camera shows at
all (most ceilings) still use the procedural texture and tint.
