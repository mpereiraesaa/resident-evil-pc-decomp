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

```sh
python -m unittest discover -s tests -p test_room3d_ceiling_linear.py
```
