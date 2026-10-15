# Reusable detail reconstruction workstream

User objective: implement all five improvements from the attached plan, preserving
automatic operation across rooms: category generators, shared multiview objects,
render-based fitting, geometry/appearance separation, and evaluation on unseen
rooms. An 80% fidelity result is not yet established or measurable against ground
truth. Mask agreement is not a substitute for that claim.

## Updated user direction: creative procedural fallback

The user explicitly permits creative interpretations when evidence cannot support
an identical reconstruction. Preserve basic placement, scale, function and the
room's visual style; exact object identity and ornamentation are no longer
required in low-confidence cases. This supersedes earlier fidelity-only rejection
as the desired final behavior, while retaining all five automation objectives.

Implement two separate acceptance paths: evidence-fitted reconstruction and
procedural interpretation. Low-confidence or poorly fitting objects should trigger
category-specific generation (stairs/landings/rails or lamp supports/arms/lights),
followed by placement and render checks. Do not merely lower every existing gate
or describe a generated object as recovered geometry. A fallback needs coherent
physical support, visibility, scale and preserved gameplay access. Disagreement
with original decorative silhouettes can be allowed; invisible geometry, blocking
a doorway or mismatched traversal endpoints cannot be excused by creativity.

Record the chosen path, evidence, reason for fallback, procedural parameters and
deterministic seed in the per-object report. Use shared category/style rules across
rooms, not hand-authored replacements for specific room IDs. Validation must
separately assess basic spatial/gameplay compatibility and stylistic plausibility,
rather than treating high image IoU as the sole success criterion. This fallback
is newly authorized and still pending implementation; existing accepted shapes
must not be retroactively relabeled as validated creative replacements.

## Delivery sequence

Keep each eventual PR independently reviewable, within the parent AGENTS.md's
3–7-file scope. Existing unrelated worktree edits are outside this workstream.

1. Geometry priors and observations: `room3d_shapes.py`,
   `room3d_shape_fit.py`, `test_room3d_shapes.py`, this document.
2. Connect fitting and acceptance to the batcher, verify texture visibility,
   and add reproducible per-object comparison/export tools and tests.
3. Run a frozen configuration on new rooms, inspect original and displaced
   views, record category failures/confidence, and validate accepted meshes
   inside the game. Keep failures in the report, not just accepted cases.

No PR has been opened and no existing runtime mod has been replaced.

## Current implementation (2026-10-11)

Generators produce closed parts compatible with the existing OBJ exporter:
regular stairs with an optional landing, inclined railings with repeated posts,
and lamps with a support, repeated arms and tapered shades. Parameters and
pose are serialized with a shape schema version. These are limited families;
spiral/multiple-flight stairs and ornate curved lamps are not implemented yet.

Observation grouping uses category and complete-link overlap of uncertain
world-space bounds, permitting at most one observation per camera in a group.
Ambiguous associations stay separate. This is provisional; it still needs
cross-projection/appearance validation to avoid mistaken associations.

The fitter performs bounded searches over pose, dimensions and discrete part
counts, using the original camera projection. It compares visible silhouettes,
boundaries and confident predicted depth. When a group has at least three views,
one is excluded from initial shape construction and parameter optimization and
reported separately. Association still sees all observation bounds, so this is
not an entirely unseen-camera reconstruction experiment.

Ten synthetic tests pass for geometric continuity, pose invariance, rail contact,
lamp symmetry, invalid dimensions, association ambiguity, empty-geometry penalties
and foreground occlusion. These tests do not establish visual reconstruction
quality or completion of the user objective.

## Development evidence and remaining work

`build/probe_shapes.py` is an initial local diagnostic, not the final reproducible
entry point. `build/room3d-shape-development` contains its candidate meshes and
scores. ROOM10B produces one associated lamp pair, another single-view lamp,
a single-view railing, and an associated staircase pair. One lamp seed exceeds
the allowed extent. Other fixture fits are poor (silhouette IoU below 0.1 in the
initial one-round run). Stair agreement differs substantially between views.

The observations expose segmentation problems: `staircase railing` mixes part
and whole labels; another staircase mask covers most of its image. Global depth
confidence originally eliminated almost all fixture observations, so initialization
now uses finite mask depths while the depth loss still uses confident pixels.
Do not interpret a better mask score as a corrected staircase.

Pending before integration: stronger part/whole mask handling and association,
multiple viable shape initializations (including existing collision-supported
stairs), baseline-vs-candidate gates, serialized per-object evidence/confidence,
per-view and held-out diagnostics, and actual visual review. Lamp geometry needs
additional initialization work before it is useful. Existing projective baking
already uses camera angle, mesh visibility and semantic guards; verify those
behaviors on the new shapes and estimated-depth foregrounds.

Held-out assets were prepared by `build/prepare_shape_holdout.py` for ROOM1010,
ROOM1070 and ROOM10A0. Selection used type-4 collision records only; that is not
proof of a visible staircase. Do not tune against held-out outcomes and then call
them unseen. If inspection requires tuning, reclassify those rooms as development
and choose a fresh evaluation set. Original ROOM10B is the development room.

The active goal remains incomplete until the five requested improvements are
integrated, evaluated on new rooms, and supported by report/runtime evidence.

The development contact sheet `build/room3d-shape-development/review.jpg` was
visually inspected. Current generated stair/railing candidates remain badly
positioned or occluded; the lamp is only a coarse incomplete approximation.
Do not enable these candidates as the default output yet. All three held-out
rooms now have detector, SAM and DA3 caches; their fit outcomes have not been
used for development. Next: improve development initialization using existing
collision-supported stair seeds and reliable multiview lamp pose constraints.

## Integrated experimental path

Set `parametric_details: true` in a room recipe to run the detail fitter after
shared architecture consolidation and before texture projection. Default output
is unchanged while this path is under evaluation. It requires segmentation and
depth caches. `before-details-recipe.json`, `after-details-recipe.json` and
`details/index.html` preserve parameters, observations, all candidate outcomes,
per-view untextured comparisons and acceptance evidence. The main room HTML links
to this report.

Initializations now also include collision-supported stair hypotheses and
planar/triangulated lamp hypotheses. Triangulation rejects poorly conditioned
rays and inconsistent projected centers; mask centers are approximate matches,
not exact feature correspondences. Combined `staircase railing` labels supervise
only railing shape, preventing solid stairs from fitting a thin railing mask.

Replacements require lower mean training loss, bounded regression in every
observed/held-out view, adequate per-view silhouette support and bounded depth
regression in every original camera. New single-view objects cannot be accepted
as proven shared objects. Only inferred props may be replaced; script-owned
furniture is excluded. Primitive cutaway indices are remapped after replacement.
Confidence records mean strength of observations, not calibrated accuracy.

For experimental detail runs, texture baking additionally rejects samples with
confident, smooth estimated foreground depth. Unknown/discontinuous depth cannot
veto texture sources. Geometry remains fixed during baking. The existing angle,
mesh-visibility and semantic tests remain in place. This does not separate baked
lighting from albedo.

There are now 16 shape/association/render/appearance tests plus the existing 34
batch tests passing. Ruff passes for all changed Python modules. The first
integrated ROOM10B run rejected all four candidates: some improved a mask score,
but none passed the full cross-view/depth/support gates. This is evidence that
initialization/masks/search still need improvement, not evidence of 80% fidelity.

The first frozen independent-room evaluation is running in
`build/room3d-parametric-evaluation` on ROOM1010, ROOM1070 and ROOM10A0.
`frozen-config.json` records source hashes and the development/evaluation split.
Its recipe file enables the same configuration for all rooms. Reproduce with:

```sh
build/room3d-vision-venv/bin/python tools/batch_room3d.py \
  --assets assets/USA --out build/room3d-parametric-evaluation \
  --rooms ROOM1010 ROOM1070 ROOM10A0 --reconstruct \
  --recipes build/room3d-parametric-evaluation/recipes.json \
  --detections build/room3d-shape-holdout/detections \
  --segmentation build/room3d-shape-holdout/segmentation \
  --depth build/room3d-shape-holdout/depth
```

Pending: inspect these independent outcomes and report failures by category;
resolve duplicate lamp observations and poor broad staircase masks; verify
texture appearance and novel views; then run accepted final geometry in Wine.
The goal remains active. No claim of generalization or runtime validation for
these new parametric meshes has been made yet.

## Search correction and initial outcomes

A synthetic known-shape experiment exposed a coordinate-search bug: trying a
new dimension could undo a previously accepted different dimension. The search
now rebases independent changes on the accepted state, tries distinct discrete
topologies independently, and reduces step sizes across five passes. An ideal
four-arm synthetic lamp now reaches 0.85 held-out-camera silhouette IoU (from
about 0.42 initially), with low depth error. This measures synthetic recovery,
not Resident Evil fidelity. A regression test exercises this full render/fit
path; 18 shape tests and 34 batch tests pass.

Overlapping same-camera fixture masks are merged only with high containment and
bounded area ratio; all source mask names are retained. This removes duplicate
lamp proposals without merging disjoint nearby objects.

The corrected development run is `build/room3d-parametric-development-v3`:
ROOM10B accepted one shared lamp; its stair and railing candidates remain
rejected. Its original-camera renders were inspected: the lamp has connected
arms and shades, but looks coarse, with some stretched/high-contrast textures.
It is an incremental improvement, not a faithful fixture replica. The generated
mesh passed the ROOM10B Wine load/camera-transition test in `runtime-details`;
mesh, material, texture and cutaway hashes were verified against the export.

The first frozen external evaluation completed: ROOM1010 (13 proposals),
ROOM1070 (7), and ROOM10A0 (2) accepted none. Its exact source snapshots are in
`build/room3d-parametric-evaluation/sources`; subsequent optimizer changes must
not be attributed to that run. Visual inventory also showed these rooms are
not useful staircase evaluation examples despite type-4 collision records.
Do not call that collision type a stair detector.

`tools/evaluate_room3d_details.py` generates a combined HTML/JSON evaluation,
including rejected categories and failed gates, and checks frozen source hashes:

```sh
build/room3d-vision-venv/bin/python tools/evaluate_room3d_details.py \
  --run build/room3d-parametric-evaluation \
  --run build/room3d-parametric-development-v3 \
  --out build/room3d-detail-evaluation
```

A fresh corrected-code confirmation run is live at
`build/room3d-parametric-confirmation` for ROOM1080 (hallway lamps). Separate data
preparation for ROOM1060/ROOM1000 targets the main hall and staircase views.
The ROOM119 background has stairs but its Stage1 RDT is an empty slot; the
actual usable return-mansion room is ROOM6190. It has not yet been evaluated.
Pending: confirm generalization with the corrected search, improve actual stair
and railing masks/initialization, inspect novel views, and retain only validated
improvements. The full objective remains incomplete.

## Tread-guided segmentation and contact limitation

The corrected-code ROOM108 confirmation accepted one lamp replacement out of
nine proposals. It has one observing camera and no held-out object view; this is
limited cross-room evidence, not proof of multiview reconstruction accuracy.
Visual inspection shows that the shade-based family poorly matches its candle
chandelier, despite passing numeric gates. Candle-style fixture priors remain a
useful next improvement.

The optional `stair_refinement` recipe setting now supports `true` (automatic
local inference using the vision interpreter, or the current interpreter) or a
path to an existing refinement directory. `room3d_stair_edges.py` detects repeated
approximately horizontal edges with an inclined depth trend; the original camera
orientation rejects vertical railing-post directions independently of depth.
Those regions supply automatic positive-point prompts to SAM. A bounded candidate
covering at least 70% of the tread-region hull may be proposed with SAM confidence
at least 0.2. That lower confidence is explicitly preserved; it does not bypass
geometry/depth acceptance. Cached proposals bind RGB, original mask and depth
hashes and reject changed inputs.

This stage found a much tighter visible stair mask in ROOM10B. A single-view
structural proposal can be tested when at least five tread edges and a correlated
incline support it; its single-view status and inferred hidden geometry are
explicit in the confidence report. Legacy object proposal generation also no
longer extrudes a solid staircase from a combined `staircase railing` label.

The initial resulting staircase passed image/depth gates but its base was at
Y=-3278.8 while the existing architectural floor was at zero. Inspection therefore
did NOT approve it as a coherent room reconstruction. Its artifact remains in
`build/room3d-parametric-stairs-development`. The next code revision locks stair
base height to the chosen room floor and tries ray-scaled contact initializations.
The grounded ROOM10B trial failed the appearance gates. This exposes an unresolved
floor-level/depth alignment problem; neither accepting a floating flight nor
forcing every level onto Y=0 establishes the desired geometry.

A potentially stronger source was identified in existing game code:
`stairs_height_update` at USA 0x0041bf90 uses room action records to compute player
height. Before deriving new record semantics, AGENTS.md requires binary verification
through Ghidra. The MCP instance query found no running instance; the user has been
asked to open the USA project/server. No new binary interpretation has been added.
Image-based experiments and report validation can continue independently.

The main-hall evaluation (ROOM1000/ROOM1060) uses a frozen pre-contact-constraint
snapshot and is still completing its render reports. It currently has no accepted
detail candidates. Do not attribute the newer floor-contact changes to that run.

There are now 21 shape/search/segmentation/contact tests and 34 batch tests passing.
Ruff passes. The tread-guided segmentation phase is a separate eventual PR from
the initial seven-file fitting/report implementation: stair-edge module, object
proposal fix, fitter/batcher integration, focused tests and this document (six
files). No PR or publication has been made.


## Verified height records and coverage (2026-10-11)

The earlier Ghidra dependency is resolved: an isolated local headless Ghidra MCP
server imported the original USA `assets/ResidentEvil.exe`. No user action is
required. Decompiled functions are retained in `build/ghidra-0x*.txt`. The original
executable and game source were not modified by this investigation.

USA `stairs_height_update` (0x0041bf90) uses direction 0/1 for forward/reverse X,
2/3 for forward/reverse Z, an unsigned interval, and a signed height increment.
It computes `(int16(distance / interval) + 1) * increment`. This describes player
height inside an action zone; it does not prove visible tread dimensions or the
floor height of the whole room. Some such rules can represent constant levels.
`cmd_room_action_set` (0x00461130) and `cmd_room_action_reset` (0x00460fc0) verify
field provenance. Reset retains the old zone pointer. The extractor records a
reset only if exactly one prior zone is known, labels it conditional, and reports
missing or ambiguous zones without guessing. Branches are not executed.

Verified additional command widths allow complete bounded initialization parsing
for ROOM1000, ROOM1060, ROOM10B0 and ROOM6190. None declares the height handler in
its initialization stream. This does not establish absence from other scripts
or runtime state. ROOM10B still has a stair-transition door declaration, not a
verified player-height rule for the visible flight.

A local inventory in `build/room-height-anchor-survey.json` examined 320 nonempty
USA RDT files: 183 initialization streams completed, 137 stopped at unsupported
commands, and none raised a malformed-input error. Height rules were recovered
in ROOM40D0 and ROOM40D1, two variants of the same room: zone
`[17700, 2000, 7000, 3800]`, direction 1, interval 175, increment +36. This is
coverage evidence, not a reconstruction result or an independent-room benchmark.
Do not infer missing height rules for the 137 incomplete streams.

The bounded parser also handles the verified variable width of player-property
opcode 0x33 (USA 0x004314b0); unsupported subtypes stop extraction. Related width
verification: 0x06 at 0x00460760, 0x14 at 0x00461040, 0x20 at 0x00430f60, 0x23
at 0x00431280, 0x2B at 0x00431990, 0x31 at 0x004608d0, 0x37 at 0x00460a30,
and 0x3B at 0x00431ea0. Never resynchronize by searching arbitrary bytes.

All 38 batch tests pass, including signed height decoding, invalid direction or
zero interval, ambiguous/missing reset zones, variable-width commands and
truncated records. The parser/provenance stage is a separate three-file change
(scene extractor, batch tests, this document). No height-derived geometry is
accepted yet. The main-hall evaluation completed with zero accepted candidates
(ROOM1000: 0/2; ROOM1060: 0/6). The full five-part goal remains active; physical
contacts, detail fidelity and generalization remain unresolved.


## Internal tread evidence and visibility diagnosis

Tread refinement now serializes the detected line segments, not only their count
and hull. Parametric stair fitting compares the projected upper front edges of
steps with those internal image lines using symmetric pixel distance. Samples
are checked against the candidate and room depth buffers; at least five detected
segments are required. The fitting objective allocates 20% to this detail term
when evidence exists. Older caches without segments retain their prior behavior.
The final baseline mask/depth acceptance gates remain unchanged. This is an
experimental fitting signal, not a claim of correct step counts.

All 23 shape tests pass, including a synthetic nine-step target that penalizes a
four-step candidate, full foreground occlusion, and absent-evidence fallback.
Ruff passes for the changed modules. No image-derived segment is treated as
verified geometry.

The ROOM10B diagnostic in `build/room3d-stair-fit-internal-edges` regenerated the
same cached flight region (identical polygon) to recover five line segments.
It accepted zero replacements. All 80 optimization trials had zero visible
candidate pixels and saturated detail error. The selected grounded candidate
projects to 4,828 pixels, overlaps 3,763 pixels of the 4,662-pixel observed mask,
but the existing room geometry occludes every projected pixel. Estimated-depth
visibility alone retains 559 pixels. Thus correcting the internal-edge loss is
insufficient: identify the responsible room surface and verify architecture and
level alignment before another stair optimization. Do not simply disable
occlusion or declare the grounded candidate correct. Diagnostic measurements are
in `visibility-diagnostic.json`; this is development evidence, not generalization.

This iteration touches the stair evidence extractor, shape fitter, shape tests,
and this document (four-file eventual PR scope). No runtime mesh was replaced.


## Automatic occlusion attribution

When a fitted candidate has zero visible pixels in a training camera, the fitter
now reports projected area, target overlap, room visibility, predicted-depth
visibility and their intersection. It renders scene groups separately to name
the nearest occluders, retaining reliable-depth sample counts. Unknown predicted
depth cannot be used as a measured occluder distance. This diagnostic does not
remove surfaces or bypass any gate.

ROOM10B's grounded candidate is occluded by boundary_2 and boundary_4. The former
is a shared extruded wall plane; the latter remains a provisional collision box.
Their respective nearest coverage in the candidate/mask overlap is 1,856 and
1,907 pixels. The estimated image depth is between these two surfaces. This
supports investigating both the foreground extrusion and the candidate's floor
alignment, not deleting both surfaces. The first room remains incomplete.

The changes belong to the shape fitter, its tests, and this document. The runtime
and accepted reconstruction are unchanged.


## Elevated-support experiment (development only)

A twelve-candidate scene-height sweep evaluated boundary_2 and boundary_4 at
500, 1000, 1500, 2500, 3500 and 4500 game units against every original camera.
Lowering both occluders was not supported: lowering boundary_4 worsened predicted
depth. Raising it improved camera 2 depth error, while other camera scores stayed
unchanged. This suggests testing a local raised support rather than a global
floor, but monocular depth alone does not prove such a platform.

`build/probe_stair_platform.py` tests boundary_4 tops at -2500, -3500 and -4500,
keeping the room floor at zero and using each top as the stair seed base. The
first two pass the existing replacement gates; -4500 is rejected. These are
explicit development interventions, NOT automatically selected supports. No
production algorithm or runtime mesh was changed. The fitter diagnostic still
reports the recipe floor as its prior; for these monkeypatched probes that field
must not be read as the actual base. Use the object's serialized origin instead.

At -2500 the fitted stair has seven steps, width 1837.7, run 4098.4, rise 1144.2,
and base [3849.9, -2500, 18950.9]. Mask IoU reaches 0.446 and 2575 pixels are
visible, compared with zero visibility under the earlier global-floor constraint.
The tread-edge error remains saturated at 1.0. Visual review shows only part of
the staircase is convincing; it is not a fidelity-approved reconstruction.
A separate true-shape rendered-depth probe confirms zero tread-edge error under
self-visibility, so the real edge mismatch must not be dismissed as a universal
rasterization failure.

Comparison: `build/room3d-stair-platform-probe/index.html`, with original views
and before/after untextured geometry for all three heights. Height-sweep results:
`build/room3d-stair-wall-heights/results.json`. Next work must infer candidate
support surfaces and validate contact extents, optimize support and object jointly,
compare against the original scene (not only a manually modified baseline), and
check other rooms before claiming this improvement generalizes.


## Unsaturated internal-edge objective

The original tread objective clipped all sufficiently distant line sets to 1.0,
creating a plateau during search. It now maps mean symmetric pixel distance d to
`d / (d + 6)`, retaining ordering for every finite distance while missing visible
edges still cost 1. A regression test verifies that 30-pixel and 60-pixel offsets
remain distinguishable; all 24 shape tests pass and Ruff passes.

The same elevated-support development probes were rerun under
`build/room3d-stair-platform-smooth`. Both -2500 and -3500 variants still pass the
existing gates. The -2500 candidate now adjusts width/yaw and reaches mask IoU
0.4468, versus 0.4457 previously: this is a small change, not a major visual gain.
Its new edge score is 0.6432; the -3500 score is 0.6577. These scores use a new
normalization and must NOT be compared numerically with the earlier saturated
1.0 as if they were a measured fidelity improvement. Camera 2 was visually
inspected again; substantial occlusion and coarse geometry remain. Elevated
supports are still diagnostic interventions, not general automatic selection.


## Automatic joint support search (opt-in)

Set `joint_stair_supports: true` with parametric details to run the new
`room3d_supports.py` path. It proposes the global floor plus nearby non-wall
collision boxes overlapping the training point cloud. Box heights include the
existing height and 0.75/1/1.25 times the height inferred from the lowest observed
band (90th Y percentile), bounded to 300..6000 units. Skipped/shared wall bounds
are excluded. These are procedural hypotheses, not learned or verified floors.

The fitter optimizes each support/object combination, keeping the base at that
support's Y. Every solid-step base vertex must stay inside the proposed platform;
invalid contact is penalized and cannot be selected. Support selection uses only
training observations and their full-scene depth errors. Held-out object views
remain excluded from selection. Final acceptance compares the entire selected
scene, including its changed support, against the original unmodified baseline
in every camera. Rejected support trials are serialized separately. No per-room
IDs or manually supplied elevations are used by this path.

ROOM10B automatic development run: `build/room3d-stair-auto-support`. Five
supports were considered (floor 0, box tops -1500, -2476, -3301, -4126). The last
failed contact validation. The algorithm selected boundary_4 at -2476 with eight
steps; candidate mask IoU is 0.4454. Camera 2 full-scene depth error improves from
0.40708 to 0.38388; the other three camera scores are unchanged. All existing
replacement gates pass. Visual inspection still shows partial visibility and
coarse details; this is not a fidelity-approved room. Comparison HTML:
`build/room3d-stair-auto-support/comparison.html`.

All 26 shape tests and 38 batch tests pass. New tests cover nearby obstacle
selection, excluded walls/skipped bounds, nonmutation of the source recipe,
base height and full-footprint contact. Ruff passes. The four-file eventual PR
slice comprises support module, fitter integration, shape tests and this document.
No game runtime or distributed mesh has been replaced.

A frozen cross-room rerun completed at `build/room3d-auto-support-transfer`
for ROOM1060 and ROOM1000. These rooms were inspected previously, so it is a
regression/transfer check, not unseen-room validation. The full five-part goal
remains active; contact selection alone does not prove staircase detail fidelity
or generalization.

Transfer outcome: ROOM1060 rejected its sole staircase group (floor selected; mask improvement/support gates failed). ROOM1000 had zero eligible staircase groups. Absence of a candidate is not a successful reconstruction. Neither result establishes generalization of the new support search.


## Recovering a real staircase observation in ROOM106

Visual review found the earlier sole eligible ROOM106 camera-3 staircase was a
false-positive corridor region. Actual stairs in cameras 1 and 6 had original
SAM IoU predictions around 0.709 and 0.713, below the default 0.8 observation gate.
Do not interpret the earlier rejected corridor fit as a failed reconstruction of
the visible main flight, and do not remove its occluding walls to force a fit.

The tread extractor now searches 3-pixel and 1.5-pixel minimum edge spacing while
retaining its existing orientation, depth-slope and correlation checks. Identical
line sets are deduplicated across scales. This recovers five coherent tread edges
in camera 1 (correlation about 0.942); the camera 3/4 false-positive regions still
produce no flight regions. Camera 6 remains unresolved.

SAM point prompts on this recovered region fragmented the flight: three masks
covered only 5..20% of the tread hull. Those masks remain rejected. A new fallback
can retain the original mask with structural evidence when detector score >=0.3,
original SAM confidence >=0.65, at least five tread edges, correlation >=0.9,
and tread-region coverage of 15..90% of the original mask all hold. It preserves
the original confidence and explicitly records
`original_mask_with_tread_evidence`; it does not claim SAM corrected the mask.
Geometric and cross-view acceptance remain required. These procedural thresholds
need further cross-room evaluation and are not calibrated probabilities.

The new ROOM106 cache is `build/room3d-stair-supported-original/room1060`. Camera 1
is now eligible with its original 0.70877 confidence. A new frozen fitting run is
at `build/room3d-auto-support-original-transfer`; it includes this genuine flight
and preserves the prior corridor candidate for rejection/reporting. This room is
now development data, not an unseen-room evaluation. Regression tests cover
cross-scale deduplication, unchanged confidence, weak evidence rejection and
flat-floor rejection. This three-file slice changes the stair evidence module,
shape tests and this document.

The fitting run completed: both ROOM106 groups were rejected. The real camera-1
flight improved its local score but reached only 0.0064 mask IoU (141 visible
pixels), failing support and cross-camera depth gates. The corridor candidate
remained rejected. All 27 shape tests and 38 batch tests pass; no accepted room
geometry or runtime mesh was changed by this iteration.


## Camera-derived tread orientation

The staircase initializer can now derive its horizontal run direction from the
backprojection planes of at least five tread lines and the original camera
orientation. The perpendicular horizontal direction gives the flight run,
without using predicted depth scale. An eigenvalue agreement check rejects
inconsistent line directions. Run sign and extent still use observed depth and
remain uncertain; this is not full camera-only reconstruction.

When a structurally supported tread polygon exists, observations retain a
separate depth cloud inside that polygon. Initialization uses this local cloud
instead of the entire staircase mask, which may include railings or background.
The support anchoring transformation applies to the same local cloud; evidence
for association and final silhouette evaluation retains the original mask.
Axis-aligned depth-cloud initialization remains the fallback when line evidence
is unavailable. No room-specific orientation was supplied.

A synthetic diagonal staircase checks direction recovery from camera-projected
edges alone. Development reruns are `build/room3d-tread-oriented-transfer` for
ROOM106 and `build/room3d-stair-auto-oriented` for ROOM10B. These assess fit quality
and regression, not unseen-room generalization. This three-file slice changes
the fitter, shape tests and this document.


The orientation-only replacement regressed ROOM10B: restricting the cloud to
visible tread lines shrank the reconstructed flight, reducing mask IoU to 0.218
and failing support. ROOM106 remained rejected. Therefore the final code keeps
both the original whole-mask initializer and the camera/tread-region initializer
as independent hypotheses; neither displaces the other unconditionally. Search
family keys retain their distinct initialization provenance. A regression test
verifies that both starts remain available when tread evidence exists. The
orientation-only artifacts remain diagnostic failures, not approved output.

The dual-seed ROOM10B run is `build/room3d-stair-auto-dual-seeds`. All 28 shape
tests pass, including the diagonal orientation and retained-initializer checks.
Ruff passes. These tests validate mechanics only; real-room fidelity remains
incomplete and the goal is still active.


## Learned correspondence probe — negative result

Existing SIFT clouds had seven accepted points in ROOM106 and one in ROOM10B,
but no pair lay within staircase masks in both views. A local LoFTR indoor-new
probe was therefore run using Kornia 0.8.3 in the existing vision virtualenv.
No images were uploaded. Official weights: https://huggingface.co/kornia/loftr ,
revision `0fb3b456cf176746709c38fa9cf80695a1c82dac`; checkpoint SHA256 and library
version are recorded in `build/room3d-loftr-all-pairs/result.json`.

All 28 camera pairs in ROOM106 and six in ROOM10B were evaluated at original
320x240 resolution. Results: 439 proposed matches,
38 pass known-camera geometry before a
confidence filter, 1 also pass confidence >=0.5, and zero
pass geometry while lying in staircase masks in both views. A separate check
without the confidence threshold also found zero such staircase correspondences.
The geometric gates are the existing <=1.5-pixel reprojection, positive depth,
and 3..150-degree ray-angle tests. No threshold was relaxed to manufacture a
usable correspondence. This does not prove the views lack overlap, only that
these tested matchers did not recover the necessary evidence.

The probe is not integrated into the batcher. `summary.json` stores input archive
hashes; per-pair NPZ files retain images and raw matches for audit, and
`reproduce.py` records the exact local experiment. This dependency installation
changed only the ignored vision virtualenv (kornia and kornia-rs); no game source,
accepted mesh or runtime build changed. Further reconstruction must continue with
single-view parametric constraints and available shared architecture rather than
assuming this experiment supplied multiview detail evidence.


### Creative fallback implementation and first probe

The optional recipe flag `creative_fallback: true` now tries deterministic
category alternatives after evidence fitting rejects an object. Eligibility
requires an existing matched object, multiple observations, or the repeated-tread
structural prior. Uncertain isolated detections do not automatically become stairs.
Lamps can use curved candelabra arms with four or six candles; stairs preserve
flight dimensions and choose regular treads; railings preserve their path.

Selection checks projected location/coverage, all-camera depth regression,
door-panel AABB intersections and staircase support. These are heuristic limits,
not calibrated confidence or complete traversal checks. Lamp ceiling attachment
and railing support remain inherited and unverified. Existing gameplay collision
data is unchanged. The original rejected fit and its images are retained, and
reports separately count evidence fits and creative interpretations.

Local probe `build/probe_creative_candle.py` / `build/room3d-creative-candle/index.html`
selected the four-arm candelabra in ROOM108 camera 2: 165 visible pixels, centroid
displacement 6.65 pixels and 85.5% of its visible pixels inside the reference mask.
Depth scores were unchanged in all four cameras. This establishes a functioning
fallback, not complete visual quality: the preview remains coarse flat geometry,
and attachment, materials and generalization need further work.


### Creative appearance and cross-room regression

`bake_projected` now accepts interpreted groups. These use local material
continuation rather than projection of the original object's silhouette and
shadows onto redesigned geometry. Missing patches remain explicitly neutral.
The normal batch and leave-one-camera-out bake both pass this distinction.
A regression test verifies zero projected samples on interpreted geometry,
material transfer when available, and no material leakage from an excluded camera.
All 39 batch tests pass.

`build/room3d-creative-stair-transfer/room1060` tests the same fallback on
a previously inspected room (regression, not unseen validation). Neither of
the two groups was accepted: the genuine stair proposal had only 68 visible
pixels, 57.1px centroid displacement and failed depth consistency. The corridor
false positive was not eligible for creative fallback. This confirms that creative
freedom alone does not repair bad placement or occluding room geometry.


### ROOM106 occluder diagnosis

The creative stair candidate projects 5,104 pixels with 4,887 inside its mask,
but only 68 remain visible. Nearest occluders are boundary groups 18 (3,734
masked pixels), 10 (932), and 9 (220). This is measured rendering evidence,
not authority to remove collision surfaces. Fit diagnostics now run for mask
IoU below 0.1, including almost-hidden candidates previously missed by the
zero-visible-pixel condition. All 32 shape tests pass.

`build/probe_stair_occluder_heights.py` evaluates 13 diagnostic hypotheses.
Lowering all three provisional surfaces from 6,000 to 1,500 exposes 2,040
pixels with 94.4% inside the mask; individual changes are insufficient.
However, all-camera depth deltas versus the original room are [0.0, 0.0024, 0.0014, 0.0344, 0.2213, 0.0027, 0.0018, -0.0499].
The candidate still fails the original-room consistency requirement, so the
height changes have NOT been applied. The comparison HTML is in
`build/room3d-creative-stair-transfer/room1060/object0/interpretation/height-probe/`.
Next work must jointly fit stair placement and implicated structural surfaces,
retaining original-room all-camera validation; changing heights alone is insufficient.


### Joint extent diagnostic: first cross-camera-compatible ROOM106 candidates

`build/probe_joint_stair_extent.py` isolates structure-only depth from added
stair depth. Lowering the three diagnosed surfaces to 1,500 changes errors by
[0, +.0027, +.0023, 0, 0, +.0027, +.0019, -.0503]. The large earlier regression
was therefore caused by the added stair extent, rather than the lower walls.

A 72-hypothesis sweep varies run, width, rise and longitudinal position. Eight
candidates pass creative placement and <=.01 per-camera depth regression.
The lowest-loss result reduces run to 5,490 from 13,725, retains rise 6,009 and
has 4,158 visible pixels, 81.8% inside the mask. However its slope is roughly
48 degrees; lower-slope alternatives also pass. No result is accepted as a usable
stair: landings/connectivity remain unverified and all cameras were used in the
search, so this is development evidence, not held-out validation.

Artifacts and source hashes: `build/room3d-creative-stair-transfer/room1060/object0/interpretation/joint-extent/`.
This supports adding a bounded joint structural/extent optimizer, with explicit
landing constraints and no per-room group IDs, rather than continuing independent
object fitting behind fixed extrusions. The final room recipe is unchanged.


### Stair end-contact verifier

`room3d_connections.stair_connections` checks a configurable-height tolerance
against horizontal room triangles just beyond both flight ends, sampling three
positions across 80% of width. It excludes the staircase itself, works under
translation/yaw, and reports a 15–42 degree design inclination prior. This is
a geometric check, not a building standard or proof of game traversal. Creative
stair interpretation now includes this gate, with detailed sample coordinates.

Three tests cover absent upper platforms, narrow corner-only contact, and
pose invariance; JSON serialization is checked. Re-evaluating the 72 ROOM106
hypotheses finds two passing both the earlier visual/depth checks and this
geometric gate. Results: `joint-extent/connections.json` under the previous
probe output. Horizontal wall caps can satisfy contact, so semantic walkability,
headroom and gameplay connectivity remain unverified. Neither candidate is
promoted to the final room. Next integration must distinguish actual landings
from wall caps before treating these contacts as traversable endpoints.


### Contact provenance exposes a ceiling false positive

The two visually compatible end-contact candidates both meet `primitive_4` at
their top. The source recipe identifies this as the inferred ceiling box
(Y -6050 to -6000), not a landing. Prior geometric contact alone was insufficient.
`support_mesh` now retains OBJ group identity for every triangle; connection
reports record nearest/contact groups and separate geometric from explicitly
classified walkable contact. Creative stair checks exclude ceiling primitives.
Four tests cover geometric contact, semantic distinction, exclusions, pose
invariance and serialization. No ROOM106 stair is validated as connected yet.
This also identifies the missing reconstruction component: an actual upper-floor
landing with independent image/structural evidence, not another height-only fit.


### Local upper-level detector probe

`build/probe_floor_levels.py` measures semantic-floor depth heights across
ROOM106. Median Y varies from roughly -1046 to +1483 in confident masks; camera7
is especially dispersed. Existing depth is aligned to camera baselines but is
not sufficiently consistent to take a single masked median as an upper landing.

`build/probe_landing_detection.py` reruns cached local Grounding DINO + SAM2
offline on cameras 1,5,6,7 with landing/upper-floor/balcony prompts. All four
completed. Visual inspection of camera1 shows upper-floor selecting the ceiling,
landing selecting the whole staircase, and balcony mixing railing and wall.
Railing masks are useful candidates, not horizontal platforms. Their confident
height ranges (roughly -6300 to -7900 in cameras1/6) disagree with the stair masks
(roughly -600 to -2600), so these cannot be joined directly from DA3 medians.
No generated platform or final recipe change was made.

Reproducible evidence and comparison: `build/room3d-landing-detection/index.html`,
`height-evidence.json`, `probe-config.json`; floor statistics in
`build/room3d-floor-level-probe/room106.json`. Next hypothesis should use projected
railing base edges shared across cameras to constrain the upper level, with
explicit uncertainty, rather than treating semantic boxes as floor geometry.


### Calibrated railing-base line hypothesis

New `room3d_levels` extracts piecewise mask lower-envelope lines and intersects
their calibrated interpretation planes. It rejects weak plane separation,
non-horizontal directions, behind-camera endpoints and short/disjoint spans.
A separate unused-view projection check requires <=3px perpendicular error
and >=20px overlap. Synthetic tests recover a known level and reject same-view
degeneracy and displaced corroboration; mask-line extraction is tested.

Strict railing masks give five lines and no shared hypotheses. Including
confident balcony masks gives 13 lines, 21 pairwise hypotheses, and one with
third-camera support. Cameras5/6 estimate Y=-6385.6, 10.4 units vertical drift
over 2335 units; camera1 corroborates at 2.28–2.91px. Camera7 does not support
this segment. This is hypothesis filtering across reused development views,
not held-out generalization or confirmed object correspondence.

Output: `build/room3d-landing-detection/levels-balcony/results.json`; reproducible
probe `build/probe_balcony_levels.py`. No platform is generated yet. Next fit
can use this candidate level and finite base edge, estimate which side contains
the walkway, and validate an actual connected stair/landing assembly.


### Connected stair/landing assembly probe

`room3d_assemblies.stair_landings` proposes both sides of a shared edge, two
gradients and two landing depths (eight assemblies). The slab meets the stair
at the estimated level; width follows edge extent and all extents remain
explicitly creative. Two tests verify both sides under non-axis-aligned edges,
full-width endpoint contact and no invented flight for a ground-level edge.

ROOM106 evaluation with original structural geometry: four proposals pass
placement (centroid errors 7.2–8.3px; roughly 72% visible pixels inside mask),
but none passes all-camera depth consistency. No assembly enters the final
recipe. This probe does NOT apply the earlier lowered-wall hypothesis, so
the next comparison must evaluate connected assemblies jointly with implicated
provisional architecture, not assume earlier individual fit results carry over.

Results and HTML: `build/room3d-creative-stair-transfer/room1060/object0/interpretation/assemblies/`.
Source: `build/probe_stair_assemblies.py`; provenance hashes are saved alongside
results. Assemblies are geometrically connected proposals, not demonstrated
walkable upper-floor reconstruction.


### Width-adjusted connected assembly

Depth attribution in camera5 identifies the stair (4475 confident visible
pixels, median predicted depth7456 versus target9923) as the +.01454 error
source; the landing contributes no visible confident pixels there.
`stair_landings` now accepts a bounded width fraction (0.4–1), preserving the
same width and contact across both components. Three assembly tests pass.

`build/probe_assembly_width.py` evaluates narrower fractions .5/.65, two
gradients and two landing depths on the promising side. Two proposals
(width fraction .5, gradient .7) pass placement and all-camera depth, with
maximum regression .00858 under the existing .01 limit. Geometry/depth
composition caches the unchanged scene; full-scene parity is checked by
`build/render_assembly_width.py`.

These remain diagnostic candidates. Their level (-6385.6) is above the
provisional ceiling slab (-6050..-6000), revealing that the enclosing ceiling
also needs a joint upper-floor-aware fit. No final recipe is promoted.
Artifacts: `build/room3d-creative-stair-transfer/room1060/object0/interpretation/assembly-width/`.


### Upper enclosure joint fit evidence

`ceiling_hypotheses` proposes clearances above a measured level, preserving
thickness and modifying only explicitly estimated ceilings. Four assembly
tests pass, including authored-geometry preservation and non-mutation.
ROOM106 clearances1800/2500/3500 alone leave visible gaps over walls and fail
all-camera depth. Extending provisional walls to the lowest proposed ceiling
improves cameras0/1/2/6 substantially, but severely occludes camera7.

Measured camera7 regression is dominated by boundary14,9,8,18,10,5. Restoring
only these surfaces to their original heights gives depth deltas
[-.13705,-.09603,-.05067,+.01449,+.01867,+.00293,-.10298,+.00538].
The two corridor cameras3/4 still fail the .01 limit; no recipe is promoted.
This is development evidence for spatially varying ceiling levels, not a
reason to weaken the depth gate or remove original collision data.

Probes: `build/probe_assembly_ceiling.py`, `build/probe_connected_enclosure.py`,
`build/diagnose_enclosure_regression.py`, `build/probe_selective_enclosure.py`.
Comparison and reports: `assembly-width/ceiling/index.html` under the ROOM106
creative stair probe. All processes completed. Next hypothesis should distinguish
upper hall ceiling from corridor enclosure while preserving stair headroom.


### Zoned ceiling: depth-compatible connected proposal, visual quality still weak

`zoned_ceiling` splits an inferred slab on X/Z, preserves slab thickness and
adds a fascia joining the two heights. Tests cover coverage, thickness,
non-mutation and invalid cuts; five assembly tests pass. ROOM106 boundary-derived
Z cuts (18000,18500,20500,22000,22500,23000,24500,25000) all pass original-room
all-camera depth limits with the selective wall extension and connected stairs.
The lowest mean error is at18000; deltas approximately
[-.1371,-.1087,-.0693,0,0,-.0183,-.1175,+.0054].

`build/bake_connected_room106.py` generated geometry, appearance and all eight
comparisons. Creative stair and landing use inferred local materials rather
than projecting the original object image; landing maps to floor material.
Visual inspection of cameras1/5 shows substantial remaining defects: planar
wall occlusion, stretched/repeated materials and poor overall resemblance.
Passing depth is explicitly NOT equivalent to good reconstruction. This
proposal is diagnostic only and is not applied to final game content.

Artifact: `assembly-width/ceiling/zoned/appearance/index.html` under the
ROOM106 creative stair probe, with recipe and report. All probe processes
completed. The process still depends on development scripts; a general joint
optimizer and validation in new rooms remain necessary, along with visual
quality improvements beyond the current depth/placement gates.


### Projection conditioning and inferred material UV repair

The appearance baker now rejects projection samples with view/normal cosine
below .25 instead of merely down-weighting them. The old behavior could still
select grazing views when they were the only source. The threshold is explicit
and reported; a synthetic grazing-plane test verifies rejection. ROOM106
rejects6683 such samples, reducing sampled vertices11630→8841 and increasing
inferred vertices3246→4620. Visual impact is limited; geometry defects remain.
ROOM108 also completes this bake, with6405 inferred vertices.

A separate bug was found in inferred-material UVs: per-vertex modulo can collapse
a triangle when its dimensions are multiples of500, and both the diagnostic
renderer and DX11 sampler clamp UVs. `material_chart` now maps a continuous
source-face chart across subdivisions, eliminating that folding/collapse.
This uses a local patch rather than claiming seamless periodic material recovery.
A regression test covers a1000-unit face and its midpoint. All41 batch tests
pass; appearance is regenerated for ROOM106 and ROOM108 under
`appearance-chart-fixed` beside the prior comparison folders.


### Creative support materials and pipeline level inference

Creative staircase vertical faces now receive an explicit neutral stone
material; tread faces retain inferred local appearance. This artistic material
is never presented as recovered albedo and does not affect evidence-fitted
objects.42 batch tests pass. The ROOM106 support-material comparison removes
the repeated carpet-like stripes from the staircase side, but clearly exposes
its large side wall and continuing pose/occlusion mismatch. This is not an
overall reconstruction-quality success.

`room3d_levels.infer_levels` now collects and corroborates base-edge hypotheses
from normal segmentation inputs. With `creative_fallback` and parametric details,
the normal batch writes `level-hypotheses.json` and includes it in its report.
No inferred floor is silently added. Using ROOM106 original masks produces two
related hypotheses around Y=-6549/-6562 from cameras0/1/6. These are overlapping
correspondence hypotheses, not independent confirmations of two floors. The
targeted masks produce none under the now-explicit detector>=.3/SAM>=.8 rules
(the older experimental script did not require the detector-score gate).

Current assembly placement anchors the flight to a finite detected railing
edge; the visual result suggests that edge may not be the stair's actual
desembarkation. Next fitting should use the corroborated HEIGHT as a prior
while resolving flight position/orientation from staircase observations, rather
than equating a shared railing segment with the staircase connection point.
Comparison: `appearance-support-material/index.html` beside prior ROOM106
outputs; level reports in `build/room3d-landing-detection/`.


### Height-only pose fitting and direct floor-ray initialization

`landing_for_stairs` now attaches a slab to any flight pose, independently of
which observation supplied the level. Existing five assembly tests still pass.
`build/probe_level_constrained_stairs.py` uses the original-mask corroborated
height (~-6556), explores72 cardinal orientation/position/gradient combinations
around the previous cloud and ranks with silhouette plus tread-edge error.
No candidate passes the depth and IoU gates; best IoU is only .0313.

New `room3d_levels.floor_anchors` intersects low-mask-band camera rays with a
supplied floor, without monocular depth. Three ROOM106camera1 anchors are near
X14330..15002,Z16706..18454, substantially displaced from the previous depth
seed near X19454. Three level tests pass, including reprojection consistency.
`build/probe_floor_ray_stairs.py` evaluates24 anchored pose/gradient combinations.
Best IoU rises to .0767 with tread-edge error .6013, but none passes all gates.
No new geometry is promoted. These are diagnostic initializations, not proof
that low mask pixels always touch the floor (occlusions can violate that prior).

Artifacts: `build/room3d-creative-stair-transfer/room1060/object0/level-constrained/`
and `floor-ray-constrained/`. Both searches completed. They retain the current
provisional structural walls; the continuing low visible overlap calls for
joint structural classification/occlusion fitting rather than more unrestricted
search around an unreliable depth seed.


### Joint shorter visual obstacles and slope-preserving fit

`lower_visual_occluders` proposes reduced heights for explicitly named existing
generated boundary groups, updating both extrusion heights and shared surfaces.
Source collision records and other primitives are unchanged. A regression test
covers consistency, non-mutation and rejection of unknown groups.

ROOM106 floor-ray diagnostic attributes remaining masked occlusion to groups
18/10/9. Re-running24 poses with these visual surfaces at1500 increases best
IoU .0767→.1292;14 candidates pass depth but none reaches .2IoU.
`fit_group` now accepts supplied initial candidates and fixed shape parameters.
Continuous fitting with only height/step count fixed reaches .3826IoU by
collapsing the flight into an excessively steep shape, so that result is rejected.

Fixing rise, step count, run and landing preserves the ~35-degree flight,
optimizes pose/width, and reaches .20564IoU (3771visible pixels, tread-edge error
.52649). All-camera depth deltas are [-.13705,-.10975,-.06811,0,0,-.03208,-.11432,+.00694].
Width grows to3057.6, run remains9365.2, rise6555.7. Every recorded trial was
checked for fixed-parameter preservation.33 shape tests pass, including a
synthetic seeded/fixed-radius fit with a reserved camera.

This remains weak visual agreement, not validated reconstruction or traversal.
The supplied enclosure was tuned in earlier development probes, so this is not
a generalization result. Artifacts: `floor-ray-short-occluders/`,
`floor-ray-refined/` (rejected steep result), and `floor-ray-refined-slope/`
under the ROOM106 object0 probe. Appearance comparison is regenerated in the
last folder. The next integration should make floor-ray/level priors and
measured occluder hypotheses part of the bounded joint batch search.

### Integrated level and occluder proposals (2026-10-11)

The ordinary creative stair fitter now considers floor-ray starts at levels
corroborated between cameras. Level evidence involving a held-out object camera
is excluded. Rise, run, landing and step count remain fixed for these starts,
so silhouette optimization cannot collapse them into steep flights.

The first integrated ROOM106 run (`build/room3d-integrated-level-priors`)
accepted neither of its two groups. The real-stair training IoU was .1177;
the corridor false positive reached .1936. Both failed depth and support gates.
This establishes that level priors alone do not reproduce the tuned probe.

Visibility attribution now covers fits below .35 IoU, instead of .1. In creative
joint-support mode, the original floor trial can propose at most two visual
occluder corrections (1500/3000 height), affecting at most three existing
boundary groups. Each group needs at least eight reliable training-view pixels
and a foreground depth discrepancy beyond the existing 180/12% tolerance.
Reserved views cannot propose corrections; source collision data is untouched.
The ordinary all-camera acceptance checks still decide whether a result is kept.
These are bounded hypotheses, not a claim that lowering walls is always correct.

A fresh integration probe runs in `build/room3d-integrated-occlusion-priors`;
its visual outcome and acceptance remain unverified at this entry. No room
content has been promoted to the game. The source test checks proposal bounds,
nonmutation, held-out exclusion, and rejection without contradictory depth.

The depth-only proposal branch finds no ROOM106 corrections: boundary18's
observed median is nearer than its rendered surface (12331 vs13297), while
boundaries10/9 have no reliable depth pixels. The rendered camera1 candidate
still shows largely featureless occluding surfaces, not a recognizable stair.
Thus depth-only evidence prevents exploring the useful alternative when DA3
itself is uncertain or inconsistent.

The branch now also permits proposals from a training observation with at least
five repeated tread edges and correlation >=.9, with at least eight overlapping
pixels blocked by that boundary. Reports distinguish depth discrepancy from
repeated-tread evidence. This does not relax final acceptance, does not apply to
an unconfirmed corridor detection, and does not use held-out observations.
`build/room3d-integrated-structural-occlusion` is the new comparison run; its
result is pending. Tests cover this alternative evidence and camera exclusion.

### Structural occlusion result and visual audit

ROOM106 object0 now passes the unchanged improving/consistent/depth/support
acceptance gates automatically. It selects visual boundary18/10/9 heights1500;
training mask IoU rises .1177→.5977, tread-edge error .5805→.4299. All-camera
normalized depth changes are [0,.00147,.00098,.00120,0,.00213,.00134,-.00832].
The alternative at height3000 scores worse. This is development-room evidence,
not held-out-room generalization and not a percentage of fidelity.

The selected shape has width5635, run7407, rise6122, 15steps and landing2745,
with a corrected flight slope52.71degrees (the earlier39.57 incorrectly included
the horizontal landing). Step rise is about408 game units: far coarser than the
180-unit procedural prior. Its appearance preview shows recognizable treads
on the left, but a broad plain side and substantial ceiling obstruction. These
visual defects survive the numerical gates. The result is retained for diagnosis,
not promoted as finished game content. Next work must incorporate regular-step
proportions and joint upper-landing/headroom into selection, then evaluate rooms
not used for tuning; simply reporting the passing gates would overstate quality.

Comparison: `build/room3d-integrated-structural-occlusion/room1060/object0/support1/appearance/index.html`.
The bake uses the actual automatic combined mesh, with an explicitly inferred
stair material. No tuned ceiling or manually chosen stair pose is substituted.


### Regular-step proportions probe

Added reusable `stair_proportions` and `regular_stair_variants` to the assembly
module. Measurements exclude the horizontal landing. Proposed variants keep
pose, width, total run and rise, and try regular160/180/220-unit step heights;
landing can be redistributed into the flight. Plausibility requires120–240
rise, at least200tread and15–42degree flight. These are declared artistic scale
priors, not recovered dimensions or proof of traversability. Six assembly tests
pass, including the steep-with-landing regression and pose/extent preservation.

The fixed-pose comparison is running in
`build/room3d-integrated-structural-occlusion/room1060/regular-steps`. Its original
candidate confirms52.71degrees and408-unit risers despite passing image/depth
gates. This exposes a missing shape-quality constraint in ordinary acceptance.
Variants are not integrated/promoted until their image and depth impact is known.

The fixed-pose run completed:34steps/rise180 and28steps/rise219 both pass
all-camera depth checks, with IoU.532/.536 versus.598original. Both have39.57
degree flights. Tread-edge loss worsens .430→.573/.572, so geometric plausibility
and image-detail objectives conflict. The28-step material comparison is being
rendered in `regular-steps/2/appearance`; no acceptance policy changed yet.

### Regular-step integration

The material bake completed and was inspected: regular treads are recognizable,
but ceiling obstruction and broad untextured faces remain. The creative fitter
now tries regular variants after fitting, using training observations only,
requiring IoU>=.35 and loss regression<=.1 per observation plus support contact.
It preserves explicit frozen step/landing parameters and multi-view level priors.
A selected alternative is labeled `regular_step_prior` and counted as procedural
interpretation, not recovered geometry. Ordinary all-camera acceptance still
runs; creative stairs also require plausible flight/step proportions. No full
headroom or traversal claim is implied by this gate.

Seven assembly tests pass, including rejection for excessive image regression,
weak silhouette and lost contact. The integration comparison is running in
`build/room3d-integrated-regular-stairs`; no results are claimed yet.

### Integrated regular stairs and measured headroom

`build/room3d-integrated-regular-stairs` completed: one interpreted stair accepted,
zero reconstructed objects; the corridor false positive remains rejected. The
selected28-step candidate has219rise/265tread and39.57degree flight and passes
all existing image/depth gates plus the new proportion gate. This confirms the
fixed-pose variant is now selected by the normal fitting path, not only a probe.

Added `stair_headroom`: three samples across each tread, vertical intersections
against explicit ceiling groups with provenance. Missing overhead is reported
unknown, not clear. This is sampled clearance, not continuous collision or
navigation proof. Normal stair reports now include it (the completed integration
above predates this reporting addition). Five connection tests pass, including
low/high/missing overhead cases; lint and diff checks pass.

`build/room3d-integrated-structural-occlusion/room1060/regular-headroom` completed.
The existing ceiling leaves only97units minimum clearance and obstructs24samples.
Raising the full inferred ceiling to1800/2500/3500 clearance resolves these
samples but ALL three fail all-camera depth preservation. At1800, cameras3/4/5
regress .0174/.0198/.0285; larger clearances regress further. Therefore moving
the whole slab is not a usable correction. Next bounded proposals should alter
only the overhead region around the stair/upper landing, retaining the rest of
the enclosure, and evaluate the seam/wall continuity as well as all cameras.
No headroom-safe room or gameplay-ready content is claimed.

### Local ceiling patch proposals

Added `raised_ceiling_patch` and `stair_ceiling_patches`: split only unsplit,
explicitly inferred ceiling boxes, retain the outer slab, raise a rectangular
stair-footprint region, and close all four patch edges with vertical fascias.
Four bounded proposals use250/750lateral margins and1800/2500clearance; no room
IDs or image-picked coordinates. Eight assembly tests pass, including original
slab/nonmutation, cap height, edge closure and out-of-bounds rejection. Lint and
diff checks pass.

`build/room3d-integrated-structural-occlusion/room1060/local-headroom` completed:
all four candidates clear the sampled stair headroom and pass unchanged
all-camera depth tolerance. Candidate1 (250margin,1800clearance) gives minimum
clearance1800 with zero obstructed samples. Its depth deltas are
[0,.00184,.00056,.00120,-.000024,.00162,.00132,-.04552]. This is materially better
than raising the entire ceiling, which failed three cameras even at1800.
The actual proposed mesh is being baked in `local-headroom/1/appearance`.
Visual seam quality, landing connectivity and generalization remain unverified;
these candidates have not yet been integrated into normal joint selection.

### Automatic overhead selection

The local-patch material preview was inspected in cameras1/6. Treads remain
recognizable; the foreground ceiling still dominates the upper image. The local
patch solves sampled clearance, not the entire room's visual ceiling mismatch.

`room3d_supports.fit` now calls bounded `fit_headroom` after selecting the stair
and support in creative mode. Only plausible stairs with measured obstruction
branch. It evaluates four local ceiling candidates against training-view object
scores (maximum .03 regression), training scene depth (.005 tolerance), and
sampled overhead clearance. It selects on training loss; ordinary final checks
still evaluate all cameras. It does not use reserved views to select a proposal.
Every proposal and overhead sample is retained in `headroom/results.json`.

The isolated integration (`room1060/automatic-headroom`) selected candidate2:
750margin/1800clearance, already passing all-camera checks in the prior probe.
The complete pipeline is running in `build/room3d-integrated-headroom`; its final
acceptance is pending. No additional game assets were promoted, and room-wide
appearance/generalization/landing connectivity remain incomplete.

### Frozen cross-room evaluation cohort

The complete integrated ROOM106 run finished: one interpreted stair accepted,
with candidate2 overhead selected and sampled headroom clear; the false corridor
candidate is rejected. Regression ROOM10B preserves its accepted stair (147rise,
514tread,15.99degree flight), with clear sampled headroom. ROOM100 supplies zero
stair observations, so its zero accepted objects are not positive evidence.

A new cohort was fixed before image/detail results were inspected: ROOM2010,
ROOM2050 and ROOM2060, with8/4/7cameras respectively. Only inventory/basic mesh
artifacts were found for these rooms; no detail evaluation in the plan/docs.
`build/room3d-frozen-stage2-evaluation` contains recipes, source hashes and actual
copied Python sources. The process executes the copied batcher, with local/offline
DINO/SAM/DA3/reviewer, parametric details, stair refinement, creative fallback and
joint supports enabled identically for all three rooms. No room-specific edits
will be used during this run. Full output is in run.log; process is active at this
entry. Failure, absent categories and held-out-camera availability must all be
reported. Cohort success cannot be inferred from development-room results.

### Evaluation coverage reporting

While the frozen cohort runs, the separate evaluation reporter now lists missing
fit categories explicitly (absence is not success), accepted multi-camera
objects, per-object/held-out evidence, and accepted stairs with unknown or failed
headroom. Repeated observations from one camera do not count as shared geometry.
Accepted creative fallback replacements cannot inherit numerical/headroom scores
from their rejected evidence fit. Two focused evaluation tests and lint pass.
The frozen reconstruction sources are unchanged; only the later reporting code
has changed. The cohort is still processing its first room, not yet a result.

Frozen cohort reference inspection: ROOM201 camera1 contains an actual descending
staircase with railings and wall lamps; camera4 contains a wall-mounted shaded
lamp beside a doorway. These observations are saved with image hashes in
`reference-inspection.json` and do not change model prompts, recipes or code.
This supplies real target-category coverage independent of detector acceptance.
The staircase is partly shadowed and lamp attachment differs from hanging lamps,
so passing development fixtures alone would not establish success here.

ROOM201 preflight using the frozen code and completed DA3/SAM artifacts finds
10 eligible lamp observations, all in single-camera association groups. No stair
or railing enters this pre-refinement set despite the visible reference stairs.
This is saved in `room2010-association-preflight.json`; structural refinement has
not yet been included, so it is an upstream coverage diagnostic, not the final
batch outcome. It already shows why category absence and multi-view coverage
must be reported separately from fitting quality. Do not tune on this cohort
before the unchanged run finishes.

ROOM201 structural refinement completed with zero proposals/attempts and no
accepted stair refinement. Stair detection score.2804 and mask-IoU estimate.681;
railing score.2959 and mask-IoU estimate.730, below existing .3/.8 eligibility.
Visual inspection of the stair mask shows it largely covers the foreground
railing/side rather than isolating the tread flight. This supports improving
region proposals, not blindly lowering thresholds. The first lamp fit is rejected
by its support/evidence gate despite image/depth improvements. These diagnostics
are retained in the reference inspection record; the cohort is still active.

### Landing clearance coverage (outside frozen cohort)

Independent audit found stair_headroom sampled treads but omitted the horizontal
landing within a stair primitive. It now samples three longitudinal positions
across three width positions on that landing too. A synthetic scene with clear
treads and low ceiling exclusively above the landing is correctly rejected
(nine obstructed landing samples). Six connection tests and lint pass. This does
not alter the running frozen cohort; its reports retain the older sampling
scope and must be interpreted accordingly. The ROOM106 regular variant has
landing0, so its earlier clearance result is unaffected by this particular gap.

ROOM201 first accepted object audit: object3 is a four-arm creative lamp in
camera3. Re-rendering its actual accepted group gives279projected pixels and
173strictly visible pixels, identical to the tolerance-based count. The diagnostic
original-image overlay is `details/object3/visibility-audit.png`; the candidate
lies on the original lamp region near the image top. This rules out tolerance-only
visibility as the explanation for acceptance. Its initial existing lamp anchor
has confident_fraction0.0 in `before-details-recipe.json`, so the acceptance does
not establish reliable 3D wall attachment. No numerical acceptance is promoted
to visual/structural success on this evidence alone.

### Lamp attachment geometry audit

Added reusable finite-triangle point distance and lamp stem-endpoint proximity
measurement to room3d_connections. Seven connection tests pass, including plane
projection outside the finite triangle, degenerate triangles, separated ceiling,
and excluding the lamp's own mesh. This is a diagnostic, not mechanical support
or orientation proof. It does not change the frozen cohort's acceptance rules.

ROOM201 object3 actual accepted mesh: stem endpoint is729.76units from nearest
vertical boundary surface (boundary14), outside100-unit diagnostic tolerance.
Explicit ceiling surfaces and vertical boundary triangles were considered;
the lamp's own triangles were excluded. Saved as attachment-audit.json beside
the object report. Its screen overlap is therefore insufficient evidence of an
attached fixture. The next fitting revision needs an attachment-aware pose/shape
constraint, with observed-view preservation; simply changing acceptance wording
would not fix the geometry. Cohort continues unchanged.

### ROOM201 completed detail result and appearance audit

The detail stage completed:10lamp groups,1creative acceptance,9rejections,
no staircase/railing fit, and no shared-camera groups. Frozen source hashes still
match. Actual `assisted.obj` views were rendered to `room2010/visual-audit`.
Camera1 shows a broad block instead of reconstructed treads/railings; camera3
shows major wall/appearance distortions around the small lamp. This is a failed
visual transfer on a new room, not satisfactory generalization. The pipeline
continues its texture-excluded-view bakes/reviews before moving to the next room.

Audit clarification: the main batcher's excluded-view images withhold a view
from texture projection only; room geometry has already used the available views.
The current source labels this explicitly and adds an HTML explanation. Frozen
cohort outputs keep their original label, so interpret them with this caveat.
Per-object reserved-camera fitting evidence is separate and exists only for
sufficient multi-observation groups. No ROOM201 object provides that evidence.

### Separate wall-fixture generator

The frozen run remains in local visual review. Added an optional `wall_sconce`
lamp profile outside its snapshot: one projecting arm, a wall plate and shade,
with local+Z away from the wall and the first beam endpoint as attachment anchor.
It requires one arm and does not automatically replace existing lamp fits yet.
Eight connection tests pass, including finite attachment under three yaws and
translation, shade remaining in front of the wall, and invalid multi-arm input.
This adds reusable category geometry; automatic wall selection/anchoring and
multi-view fitting remain to be implemented and verified separately.

ROOM201 visual evidence is assembled at `room2010/visual-audit/index.html` with
original/actual-baked cameras1/3 and explicit failed-detail notes. The unchanged
cohort continues; this partial review is not a final three-room conclusion.

### Finite wall-anchor proposals (outside frozen cohort)

Added `room3d_fixtures.wall_lamp_proposals`: bounded nearest finite-wall proposals
for the wall_sconce family. Anchors attach to real triangle points (not infinite
planes), ignore near-horizontal faces, cap displacement/candidate count, and
orient local+Z toward the reference fixture or observing cameras independently
of triangle winding. Provenance includes group, anchor, normal and displacement.
Ambiguous orientation is omitted. Plate footprint contact remains unverified.
The existing point-distance helper now exposes closest_triangle_point for reuse.

Nine connection tests pass, including wall attachment, remote-face/ceiling
rejection and winding invariance;34shape tests also pass. Lint/diff checks pass.
These are candidate generators, not an integrated accepted correction. Fitting
must preserve attachment and check every observing image before retaining them.
The Stage2 evaluation continues to execute its unchanged snapshot.

### Attachment-preserving render fitting

Creative lamp fits with wall-related labels now add bounded wall-sconce starts
from training observations and existing boundary faces. Each candidate keeps its
surface anchor and outward yaw while size parameters optimize by rendering;
position/yaw perturbations cannot detach it. Shape families retain independent
starts. Accepted anchored alternatives are labeled procedural interpretations.
No single-view evidence or final all-camera gates were relaxed.

A synthetic render-fit test verifies every recorded trial retains the wall
anchor and orientation while improving or preserving training loss. The first
test fixture omitted baseline floor boundaries and failed setup; adding a valid
floor extent fixed it, and the focused test passes. Full shape tests are running.
Lint/diff checks pass. Real-room attachment fit and plate footprint clearance
remain unverified; frozen Stage2 reconstruction continues unchanged.

### Plate support and development regressions

The35shape-test rerun passed after fixing the synthetic fixture. ROOM108
regression completed and preserves its one creative lamp; its observation label
is chandelier, so this is a non-regression check rather than wall-anchor evidence.
ROOM10B lamp-only development run is active to exercise actual wall-labeled starts.

Added finite plate contact checks for four corners and center against the cited
wall group. Anchored starts/variants receive invalid attachment loss if the plate
is unsupported, and ordinary acceptance includes this attachment condition.
The synthetic render-fit now includes an actual wall boundary and asserts valid
plate support; a tiny patch touching only the center is rejected. Nine connection
tests and the focused render test pass. Lint/diff checks pass. The two development
runs started before this extra plate check; do not attribute it to their results.
Frozen Stage2 continues unchanged. Real-room plate-aware validation is pending.

### Wall-fixture regression results and fallback preservation

ROOM10B lamp development completed: one existing shade lamp accepted; four
wall-sconce starts were evaluated but did not win. This demonstrates preserving
the previous choice, not successful wall reconstruction. ROOM108 wall-labeled
observations are now fitting in `build/room3d-wall-plate-108` with finite plate
contact enabled. This is a development regression, not unseen-room evidence.

Creative fallback for an already anchored wall lamp now preserves its wall
family and attachment while varying reach; it cannot silently replace it with
an unattached candelabra. The fallback also checks finite plate contact.
Initialization/final reports retain attachment provenance for audit. Nine
connection tests, including attached creative alternatives, pass; lint/diff
checks pass. This does not fix unanchored hanging candidates or prove a positive
real-room wall fixture yet. Frozen Stage2 remains unmodified.

### Anchored search redundancy and independent visual review

The local ROOM201 camera1 reviewer independently describes the original stairs
and the reconstructed foreground as a long flat platform. This corroborates the
manual image inspection, without making reviewer text ground truth.

Anchored optimization now skips perturbations that collapse to identical geometry
after restoring the wall anchor. Plate support checks are reused for identical
plate vertices/group in the immutable scene, including across observing cameras;
radius/shade changes do not require repeating the same finite-wall query.
The focused render-fit still passes (observed0.51s after caching, not a controlled
benchmark). Lint/diff checks pass. The currently running ROOM108 development fit
predates this optimization and is left running rather than restarted.

### Multiview anchored fixture test and first real failure

The anchored render-fit test now uses three cameras: two for fitting and one
reserved. It preserves anchor/yaw in every trial, retains finite plate support,
and exceeds .6maskIoU in the reserved camera. The focused test and lint pass.
This synthetic test supplies the shared object group and wall anchor; it does
not validate real-world observation association or anchor discovery.

ROOM108 first two wall groups completed and were rejected. Object0 generates
four wall candidates with valid plate support but zero maskIoU, only6–8visible
pixels; the unanchored winner also has zeroIoU. Object1 has no nearby wall starts
and zero visibility. Thus finite attachment alone does not solve placement:
nearest-wall projection from the depth-seeded lamp can move to the wrong screen
region. The next proposal path should intersect rays from the observed mask
with finite wall faces, then validate multi-view agreement, instead of relying
solely on nearest depth-space surfaces. Remaining ROOM108 groups are still live.

### Image-ray wall attachment proposals

Added bounded finite-wall ray proposals from training-mask upper bands. They
intersect calibrated camera rays with wall triangles, reject behind/near-camera
and off-triangle hits, estimate fixture dimensions from projected mask extent,
and retain camera/pixel/surface provenance. They do not require the DA3-seeded
fixture to already be near the correct wall. At most two distinct wall hits per
ray and eight starts total enter the existing family-ranked search. Only training
observations supply rays; final fit/plate/image/depth gates remain in place.

Ten connection tests pass, including exact source-pixel reprojection despite an
intentionally wrong reference depth and rejection of behind/off-ray walls. Lint
passes. A targeted development comparison for ROOM108 camera0 mask03 now runs in
`build/room3d-ray-wall-108`; the earlier nearest-wall-only run remains live and
unchanged. No positive real-room result is claimed until that comparison ends.

### Ray candidate accepted, then shade penetration found

ROOM108 camera0 ray fit completed with one anchored procedural acceptance:
maskIoU.531,210visible pixels, finite plate contact and all-camera depth gates
pass. The initial ray candidate already had.212IoU versus zero for nearest-wall
starts. Attachment uses boundary1 and pixel(231,116), not a tuned 3D position.

Geometry audit invalidates the accepted shape as finished content: radius105
and shade457 place part of the shade behind its wall plane. Added wall-sconce
shape validity requiring radius >= (shade+thickness)/2; initial proposals respect
it, fitting rejects invalid variants, and creative reach variants skip violations.
Ten connection tests pass including this penetration case; the focused anchored
render test also passes. This is a local-plane constraint, not collision proof
against every room surface. A fresh clearance-constrained fit runs in
`build/room3d-ray-wall-clearance-108`; the previous accepted artifact remains
retained and explicitly unsuitable for promotion.

### Clearance-constrained lamp: geometry passes, visual support remains wrong

The fresh ROOM108 camera0 fit completed with one procedural acceptance. Its
wall plate passes finite-surface contact; the nearest shade vertex stays27.816
units in front of the attachment plane. MaskIoU drops to.3333 (148visible pixels),
from.5311 for the invalid penetrating version. The valid constraint must remain;
the higher old score is not a reason to restore penetration.

Actual material baking and all four camera renders are available at
`build/room3d-ray-wall-clearance-108/appearance/index.html`. Inspection of the
original and rendered camera0 exposes a semantic problem: the original lamp
appears supported by furniture, despite its detector label `wall lamp`. The
room's furniture is also severely deformed. This is not a visually successful
room or confirmed original fixture placement. `visual-audit.json` records the
finite contact check, measured shade clearance and artifact hashes. One observed
camera supplies no independent multiview verification.

Next fixture work should compare furniture-supported and wall-supported proposals
using scene geometry and image evidence instead of treating the detector label
as sufficient support evidence. The frozen Stage2 batch remains unmodified and
has advanced to ROOM205; ROOM201's completed visual failure remains recorded.

### Competing fixture support families

Added a table-lamp generator with a finite base, upright stem and shade. Training
mask lower-band rays can propose contact on horizontal furniture/platform tops;
wall proposals no longer require the detector's wall label. Candidate furniture
surfaces come from explicit furniture primitives or non-wall collision blocks.
Only the highest horizontal face per support group is eligible. Base corners
and center must fit the finite support. Resizing preserves the chosen anchor and
family, including creative alternatives; contact checks use the whole base in
their cache key. Original collision data is unchanged.

Eleven connection tests and35shape tests pass. New coverage verifies support
reprojection, exclusion of the underside/floor, base containment, no geometry
below the base plane, and preservation under resizing and creative alternatives.
Lint passes for the five changed Python files.

ROOM108 preflight produces **zero table candidates** on its current scene mesh.
This is evidence of missing/inaccurate support geometry, not evidence that the
lamp belongs on a wall. The existing recipe skips several collision boundaries
and uses approximate obstacle heights. The full family comparison is running at
`build/room3d-support-family-108`; it cannot demonstrate table-vs-wall selection
when one family has no admissible support. The next step is to diagnose support
height/footprint coverage and evaluate joint furniture-and-fixture hypotheses,
retaining full-scene depth and visibility checks. No real-room table-lamp success
or generalized support solution is claimed.

### Joint furniture-height diagnosis and bounded proposal search

ROOM108 camera0 lower-mask rays intersect the footprint of active boundary14,
but require a top height around1800–1900 rather than the default1500. They also
cross skipped boundaries11/15; these remain skipped. The numeric diagnostic is
`build/room3d-support-family-108/support-coverage.json`.

`lamp_support_hypotheses` now generates at most eight training-ray/footprint
height hypotheses, preserving footprint, skipped boundaries, shared surfaces and
source collision data. Only non-wall blocks are eligible, with bounded heights.
`fit_fixture_supports` integrates those recipes with table-lamp fitting under
creative fallback. Candidate selection uses training views and rejects whole-scene
depth regression beyond.005; final all-camera gates still run in fit_objects.
Rejected support trials remain in the report. Twelve connection tests pass,
including immutable input, excluded supports and empty training evidence.

The two-round development probe at `build/room3d-joint-fixture-support-108`
finds finite table support with maskIoU up to.7005, versus.3333 for the wall
interpretation. All three proposed top heights worsen depth in cameras0/1 beyond
the existing threshold (approximately+.008/+.007 at the lowest height). Thus
lamp-only image improvement does not justify promotion. This points to excessive
or inaccurate support footprint/shape, not merely a missing lamp family.
The full integrated run is active at `build/room3d-integrated-fixture-support-108`;
its outcome is not yet verified. The frozen Stage2 evaluation remains separate:
ROOM205 detail fitting accepted one single-camera chandelier, whose visual result
still requires inspection; no shared-object validation has been established.

### Integrated support rejection and ROOM205 visual audit

The full integrated ROOM108 run completed. All three table-support trials pass
contact but fail the training depth gate; none is selected. It retains the
previous procedural wall interpretation, whose semantic support remains doubtful.
This verifies rejection/reporting integration, not a successful furniture repair.

ROOM205 actual textured mesh was rendered from all four cameras into
`build/room3d-frozen-stage2-evaluation/room2050/visual-audit/index.html`.
Inspection of cameras0/1 finds a missing tile-floor appearance, obscured door/
center geometry in camera0, and stretched hanging surfaces in camera1. Its
single-camera chandelier maskIoU.818 does not validate the room. Findings are
recorded in `visual-audit/findings.json`; this is another visual transfer failure.

The bake samples3372of27120vertices (12.43%). This counts vertices, not textured
screen area; geometry occlusion, view angle and semantic rejection still need
separate diagnosis. The evaluation report now exposes vertex-sampling fraction,
inferred vertex count and grazing rejections, returning unknown when bake data
is missing. Its two tests and lint pass. Frozen generation sources remain intact.
Before further fixture refinements, investigate structural occlusion and appearance
coverage: both development and new-room evidence show these limit the whole room.

### Structural door occlusion isolated from depth-based appearance rejection

ROOM205 group rendering identifies boundary1 in front of the scripted door panel:
the panel had zero visible pixels in camera0. Its wall plane is x16920, while the
panel spans x16933..16957. `door_openings` now subtracts the panel rectangle from
the nearest designated axis-aligned structural wall, only for script-owned door
panels within100units. Other wall faces, collision data and the closed panel remain.
It preserves face winding and handles repeated application without further cuts.
Two opening tests and42batch tests pass; lint passes. Generated camera0 geometry
now exposes2697door pixels. This fixes geometric visibility, not its missing texture.

A diagnostic bake with observed-depth rejection disabled restores much of the
floor texture, while retaining mesh visibility and semantic checks. The normal
bake with the door opening still renders a gray door/floor region. Thus depth
consistency is a separate appearance bottleneck. Do not disable it globally:
foreground texture contamination still needs prevention. Inspect scale/offset and
confidence of DA3 versus the known floor and script door before changing acceptance.
Artifacts are under `build/room3d-surface-diagnosis-205/{no-observed-depth,door-openings}`;
`door-openings/door-visibility.json` records all four views. Frozen outputs are
unchanged. ROOM205 has now informed development and cannot serve as a fresh
unseen validation room for these changes.

### Floor-supported appearance depth calibration

The structural semantic reader accepted only the exact label `floor`, missing
`wooden floor` despite SAM scores above.85. It now recognizes the floor token
while giving fixture labels (including floor lamps) priority. ROOM205 floor rays
show nearly constant geometric/DA3 depth ratios per camera:1.249,1.189,1.268,1.357.
Checkerboard validation90th-percentile log errors are.008,.031,.010,.034. The
previous foreground veto therefore rejected legitimate floor appearance.

Appearance baking now calibrates each source observation from confident masked
floor pixels visible in the mesh. Alternate16pixel tiles fit/validate a bounded
scale; fewer than300samples, poor validation or excluded source cameras leave the
original depth untouched. Geometry optimization inputs remain unchanged. Mesh
occlusion, semantic filtering, discontinuity checks and depth veto remain active.
The report records per-camera scale and validation; this is not a held-out-camera
geometry test. Synthetic tests cover mislabeled floor, floor-lamp exclusion,
scale recovery, inconsistent validation, occluded floor and absent evidence.
All43batch tests and lint pass.

Actual bake/render comparison at
`build/room3d-surface-diagnosis-205/floor-calibrated/index.html` visibly recovers the
tile floor, but the door remains gray and distorted wall/ceiling surfaces remain.
Sampled vertices rise from3372to5767of27120; this is not an area/fidelity score.
ROOM108 regression is at `build/room3d-ray-wall-clearance-108/floor-calibrated`:
three camera calibrations validate, camera3has no floor samples and stays unchanged.
Inspected camera0retains its previous appearance and unresolved furniture shape.
Next diagnose door semantic texture rejection and enclosure shape separately;
do not describe these rooms as reconstructed successfully.

### Door-category ambiguity and third-room visual audit

ROOM205 camera0door has2697visible panel pixels. A crate detection(.518score,
.981SAM) overlaps94.4%; a door detection(.303score,.952SAM) overlaps94.0%.
The previous.4detector threshold discarded the door interpretation and permitted
only151panel pixels as texture sources. For an expected door_panel, a door mask
with score>=.3and SAM>=.9can now provide compatible evidence. Foreground fixture
masks still veto it; the exception does not apply to generic walls or weak masks.
Tests cover label order, weak evidence and foreground exclusion. All44batch tests
and lint pass.

Actual appearance at `build/room3d-surface-diagnosis-205/door-semantic-prior`
recovers texture on the door panel and retains the floor improvement. A gray
border remains because the estimated panel is smaller than the original door.
This is partial appearance recovery; panel extent fitting and distorted enclosure
surfaces remain unresolved. The visual audit is saved beside the comparison.

Frozen ROOM206 was rendered into `room2060/visual-audit/index.html`. Camera0
retains wallpaper/floor but misses furniture; camera3misses the central table and
distorts the doorway. Details accept one lamp and reject another lamp and a
railing, all single-camera observations. Findings are recorded in that directory.
There is still no demonstrated shared-camera object success in the frozen cohort.

### Depth-scale association probe: overlap alone gives a false candidate

`build/room3d-floor-association/results.json` compares raw and floor-calibrated
observation grouping on the frozen geometry. ROOM201 changes from10single-view
groups to9groups with one two-camera pair; ROOM205 remains1single-view group and
ROOM206 remains3single-view groups. This experiment does not integrate calibrated
depth into geometry fitting or overwrite frozen results.

The ROOM201 proposed pair is camera5mask03 and camera6mask00. Its median pixels
are(203,30)and(232,65), and calibrated-depth boxes overlap. However,
`triangulate_centers` returnsNone: the camera rays cannot establish a consistent
finite center. The targeted fitting probe exits before fitting, with evidence in
`build/room3d-shared-lamp-201/association.json`. This is not a shared-object success.

Lamp association now requires the existing bounded ray-consistency check whenever
camera models are available, in addition to box overlap. A synthetic regression
keeps truly converging views together and separates incompatible rays despite
identical uncertain depth boxes. All36shape tests and lint pass. Calibrating
depth alone is therefore insufficient: future grouping needs better correspondences
and detection coverage, not just larger overlap tolerances. No room-specific match
or manually asserted object identity was added.

### Association coverage and a depth-independent pair fitting experiment

`build/room3d-floor-association/pair-coverage.json` enumerates all same-category,
different-camera pairs without requiring depth-box overlap. ROOM201 has10usable
observations,42pairs and one ray-consistent proposal (camera3mask02/camera4mask00).
ROOM205 has1observation and no pair; ROOM206 has3observations,1pair and no consistent
proposal. The HTML in that directory records this coverage. Detector coverage is
therefore a separate limitation from association thresholds.

The ROOM201 pair was jointly fitted without automatically changing association.
`build/room3d-ray-pair-lamp-201/result.json` retains the shared candidate and search.
Its two maskIoUs are.197/.656: the mean-loss optimum favors one view and leaves the
other poor. It is an unanchored six-arm shade approximation, not a validated wall
fixture or shared-object success. Both views train the fit; no third view exists.
Post-hoc trial inspection finds a worst-view-loss candidate at.475/.274IoU and a
max-min-IoU candidate at.298/.358. Neither proves good reconstruction. The next
experiment should balance view losses during search and preserve support priors;
simply admitting ray-based associations would promote an inadequate shared object.

### Balanced-view objective experiment

Added opt-in `balanced_view_fit` for shape-search ranking and coordinate updates:
half mean training loss plus half worst training loss. It leaves single-view
behavior unchanged and does not include held-out views. Existing per-view
regression/contact gates remain. The report names the training objective; default
remains mean until broader evidence supports a change. All37shape tests and lint
pass, including the ranking tradeoff and single-view invariance.

The same three-round ROOM201 pair trial now producesIoU.269/.495, versus.197/.656
for mean loss. The worst loss improves.758to.698, but mean loss worsens and the
selected shape is still an unsupported five-arm shade. This is insufficient for
promotion or confirmed association. `build/room3d-balanced-pair-lamp-201` retains
the result and `comparison.html` overlays each candidate in both original views.
The experiment suggests that loss balance alone cannot replace an appropriate
fixture family, reliable attachment and better observation correspondence.

### Support-plane anchor refinement experiment

Added opt-in `refine_fixture_anchor`: inferred anchors may slide tangentially on
their existing support plane during coordinate fitting. Normal displacement is
removed, cumulative motion is bounded1000units from the recorded seed, and the
existing finite-contact gate still rejects moves outside the support. Parameter
resizing preserves the anchor. Wall orientation remains determined by its support;
horizontal table supports preserve height. Thirteen connection tests and lint
pass, covering plane constraints, resize behavior and cumulative-motion bounds.

The balanced ROOM201 pair experiment at `build/room3d-sliding-pair-lamp-201`
completed with the same winning unanchored shape andIoU.269/.495. The best
worst-view wall trial reaches.310/.218with contact valid in both views. Sliding
alone therefore does not solve this pair and remains experimental. Existing wall
templates have a vertical shade, while the source fixtures appear to project
outward from their mounting brackets; orientation/topology deserves a bounded
template comparison rather than relaxing attachment or association gates.

### Orientable wall-shade experiment: no improvement on the pair

Wall-sconce geometry now supports bounded shade tilt with explicit vertex clearance
against its wall plane. Existing zero-tilt geometry retains its support. An opt-in
`orientable_wall_shades` search proposes both horizontal orientations in addition
to the existing shade; their initial arm reach clears the wall and optimization
continues to enforce clearance/contact. Fourteen connection tests and lint pass.

`build/room3d-oriented-pair-lamp-201/result.json` records54oriented optimization
trials. The best worst-view oriented trial hasIoU.243/.203; the overall winner is
still the previous unanchored five-arm shade at.269/.495. This is a negative
experiment, not a new visual improvement. The option stays disabled by default.
Changing tilt alone does not explain this pair; correspondence and the reconstructed
support plane remain uncertain. Do not promote this association or weaken contact
checks to make an attached candidate win.

### Frozen cohort complete and full asset inventory

The frozen Stage2 process completed successfully. The consolidated report is
`build/room3d-frozen-stage2-report/index.html`; source snapshot hashes verify.
It records14object proposals across ROOM201/205/206,3numerically accepted lamps,
zero accepted shared-camera objects and zero objects with reserved-camera geometry
evaluation. Stairs have no fit observations in all three rooms; this includes a
known missed staircase in ROOM201, so missing observations cannot count as success.
Qualitative visual failures/limitations and rendered comparison links are now
included in the reusable evaluator. Missing visual audits explicitly remain
`not_inspected`. Both evaluation tests and lint pass.

A fresh full `--survey-only` run at `build/room3d-current-coverage` read348RDTfiles:
320room entries and28empty slots, with1684camera entries and no missing background
files. These include scenario variants, not320unique physical rooms. This confirms
asset-reader coverage only; no all-room reconstruction or manual-work savings
measurement was performed. `coverage.json` records the inventory summary.

Current completion evidence therefore remains insufficient for the five-part
goal: generators and fitting exist, but reliable shared objects and transfer of
stairs/railings to new rooms are not demonstrated. The next stage needs detector/
correspondence coverage and a new frozen cohort for the recent appearance fixes;
repeatedly tuning ROOM201/205cannot establish generalization.

### New Stage3 cohort and inset-background compatibility

Selected ROOM3060/ROOM3110 before inspecting their images, from inventory camera
counts(4/3). Prior artifacts were inventory/basic meshes. Current tools and recipes
are snapshotted in `build/room3d-frozen-stage3-evaluation`; experimental balanced
loss/anchor slide/shade orientation options remain off. DINO/SAM/DA3 run locally
offline. Direct rendered inspection replaces the slow Qwen reviewer for this cohort.

Initial execution terminated before inference on a316x236TIM background. The
reader only accepted320x240. Existing runtime source `Rendering.cpp` AddBackgroundQuad
maps full textureUV0..1to the logical viewport, including inset textures. The reader
now normalizes316x236to320x240with bilinear sampling, preserving source assets and
rejecting other unexpected sizes. This matches the current runtime display mapping;
it does not establish the historical executable's inset/shake semantics. All45batch
tests and lint pass. The frozen manifest records the startup error, previous hash,
compatibility correction and renewed batcher hash. The terminated process was then
restarted; inference is now active. No visual parameter tuning occurred.

Planned old/new comparison will reuse identical model artifacts. The old snapshot
also requires the same inset-image compatibility normalization to run these rooms;
that prerequisite must be recorded separately from appearance/geometry improvements.
Do not claim successful Stage3 reconstruction until fitting and visual checks finish.

### Stage3 completed: new coverage failures, paired baseline running

The new frozen run completed both rooms. ROOM306 has0detail observations; ROOM311
has1rejected observation. Actual textured camera renders and findings are saved
in each `visual-audit` directory and consolidated at `build/room3d-stage3-report`.
ROOM306 camera2recovers parts of stone walls but has major gray/distorted surfaces.
Its other source images are dominated by a repeating surface; interpreting those
as complete room references is unverified. A stride diagnostic confirmed camera0
is declared320x240and decodes to that exact payload size; it does not establish
corruption or a decoder stride bug.

ROOM311 camera0renders black against an outdoor courtyard reference. The bake has
zero sampled vertices and the survey only one collision boundary. Missing scene
coverage is a prerequisite failure, not a lamp or texture-style issue. These are
negative transfer results and should not be described as successfully reconstructed.

`build/room3d-stage3-baseline` snapshots the older Stage2 tools, copying only the
inset-background normalization as a documented shared compatibility prerequisite.
Its active process reuses the exact new-run detector/SAM/DA3 caches (hash-verified
by existing loaders) without new inference. Compare after completion; do not infer
relative improvement from the new images alone. Architectural coverage from sparse
collision records now needs investigation before more small-fixture tuning.

### Sparse enclosure prototype and completed paired baseline

ROOM311's sole collision rectangle spans x0..36,z0..18, while its cameras lie
thousands of units away. The generated floor and ceiling inherited that tiny
extent. This explains the missing scene coverage; the collision rectangle cannot
serve as the visual room perimeter.

`build/probe_sparse_enclosure_311.py` prototypes an image-driven alternative:
floor/wall SAM masks and valid DA3 samples are backprojected through original
cameras, horizontal wall normals are fitted by PCA with residual limits, and a
floor is inferred from pooled floor observations. It preserves source collision
records; the experimental visual recipe replaces tiny enclosing surfaces and
removes the unsupported ceiling. No hand-authored room coordinates are used.
At `build/room3d-sparse-enclosure-311/index.html`, camera0now shows courtyard walls
and ground instead of black. Camera1still has a wall-floor gap and texture seams;
the sky/exterior is absent. Per-view wall planes are not yet fused into a shared
enclosure. This remains a development prototype, not production integration or
validated geometry. The next step is shared-plane reconciliation and coverage/
occlusion checks before enabling a sparse-collision fallback.

The baseline process was confirmed terminal with exit143after finishing ROOM306.
It was resumed for the pending ROOM311 only and completed. The paired report at
`build/room3d-stage3-paired-report` verifies both frozen source sets and reuses the
same model outputs. Both versions yield zero accepted details and zero sampled
vertices in ROOM311. ROOM306 has1365vs1316sampled vertices (old/new) out of17160;
this is not an image-quality measure or evidence of improvement. The new-room
experiment does not yet demonstrate transfer of the recent appearance fixes.

### Shared-plane and floor-junction ablation (ROOM311, development)

`build/probe_enclosure_junctions.py` compares the existing per-view enclosure,
a shared-wall variant, and that variant with a finite floor extension. Report,
recipes, all three camera renders and comparison HTML live at
`build/room3d-enclosure-junctions-311/`. These are development proposals, not
production defaults or held-out validation.

The two wall patches differ by 1.437 degrees, with maximum endpoint distance
450.497 world units from the larger plane. The smaller patch is contained along
the larger plane's tangent. The prototype absorbs it using provisional limits
of 3 degrees and 500 units; these absolute limits still need a noise/scale-aware
replacement before reuse. It does not establish that all wall observations
belong to a single planar surface. The floor extension follows the inferred
wall only over the existing floor's tangent interval, bounded to a quarter of
the floor's perpendicular span. Source game collision records are unchanged.

Camera1 semantic floor/wall coverage is 62938/67357 pixels for per-view planes,
62412/67357 for shared walls alone, and 67124/67357 after the junction. Total
coverage increases from 68222 to 75207 pixels for the last step, but comparable
valid-depth pixels increase by only 319 (45740 to 46059). Much of the new area
therefore lacks depth evidence. Foreground-excess pixels under the diagnostic
12%/180-unit rule remain 499; this is not proof of correct added geometry.
Cameras0/2 have unchanged coverage and foreground-excess counts. No camera was
reserved from fitting.

All three camera rows were visually inspected in `comparison.png`: shared walls
reduce the duplicate-plane texture seam; the floor junction closes most of the
black strip in camera1, with smeared/shadow-like texture near the wall. Camera0
still lacks sky, trees, railing and lamp geometry; camera2 retains black borders.
The result supports investigating shared architectural support before further
fixture fitting, but does not justify enabling this fallback globally. Next:
replace absolute association tolerance with depth uncertainty, validate plane
support per camera, and test the enclosure fallback in another sparse room.

### Residual-gated wall association and cross-room stress test

`tools/room3d_enclosure.py` now holds an experimental reusable vertical-plane
fitter and shared-wall proposal generator. It fits equal sample counts per view,
requires overlapping finite patches and compatible normals, and checks every
original point cloud after each merge. The candidate's 90th-percentile residual
must remain within the original residual plus 1% of that view's median depth.
This replaces the earlier fixed 500-world-unit association allowance, but the
1% allowance remains a provisional heuristic, not calibrated uncertainty. All
source cameras and per-view residual checks are retained in the decision log.
There is no production batcher integration yet.

Four tests in `tests/test_room3d_enclosure.py` passed: equal-view weighting and
scale invariance; separated/disjoint patches; accumulated drift across a chain
of cameras; invalid/degenerate inputs. Ruff passed for the module and tests.

The identical algorithm was run using existing verified caches in ROOM311,
ROOM205 and ROOM306 (`build/probe_evidence_enclosure_{311,205,306}.py` and
`build/room3d-evidence-enclosure-{311,205,306}/`). ROOM311 accepts the two-view
proposal: source p90 residuals 420.62/256.07 become 511.17/283.71, below limits
651.35/358.29. This is estimated-depth agreement, not recovered architectural
truth. Its three renders still show missing exterior, floor gaps and texture
seams; this experiment intentionally omits the previous floor-extension ablation.

ROOM205 supplies only one eligible wall cloud, so no multi-view association is
possible. Its inferred enclosure loses large portions of the walls and misprojects
furniture into surfaces. ROOM306's only wall proposal has residual/span 0.162,
exceeding the prototype's 0.15 gate, leaving no inferred wall. Its room cannot be
reconstructed by replacing existing structural surfaces with these proposals.
All three ROOM311 views, all four ROOM205 views and ROOM306 camera2 were visually
inspected in `build/room3d-evidence-enclosure-311/cross-room.png`. These are known
development rooms, not a new held-out evaluation.

Conclusion: association is reusable but a blanket image-only enclosure replacement
is rejected. Next work must extract multiple planes from individual wall masks
and preserve supported existing surfaces when evidence is incomplete. The floor
height also needs camera-consistent calibration before it can replace an existing
floor. No claim of enclosure generalization or full goal completion follows.

### Multiple wall planes and bounded observed support

`room3d_enclosure.extract_wall_planes` now produces deterministic vertical-plane
RANSAC proposals from a wall point cloud. It refits each supported plane, removes
assigned samples, retains unassigned clutter, and records source indices and a
provisional depth-relative tolerance. `supported_wall_tiles` bounds each plane to
occupied cells instead of filling its whole rectangular extent. Empty cells are
unknown, not declared free space. Neither API is enabled in production.

Seven enclosure tests pass, including perpendicular walls at multiple scales,
determinism, unique sample ownership, clean single-plane behavior, small inputs,
and preservation of an unobserved gap. Ruff passes after import formatting.

Two additive experiments preserve existing structural surfaces, floor and game
collision records. `build/probe_multiplane_enclosure.py` produces rectangular
proposals; `build/probe_tiled_enclosure.py` limits them to supported cells. Both
were run on known development ROOM306 and ROOM205 with the same cached inputs.
Artifacts live in `build/room3d-{multiplane,tiled}-enclosure-{306,205}/`.
ROOM306 camera2 supplies four planes (the former single-plane proposal failed).
Rectangle additions introduce 8367 foreground-excess pixels among 38860 valid
wall-depth comparisons; supported cells introduce zero by the same 12%/180-unit
diagnostic. Original wall coverage was already complete, so this is not a coverage
gain. ROOM205 yields two proposals; foreground-excess remains 1719/10119 for both
variants and the existing scene, with no improvement established.

Visual inspection of ROOM306 camera2 (`comparison.png`) shows an important
tradeoff: rectangular patches reproduce more of the image but disagree with some
estimated depth; supported cells reduce that conflict while producing fragmented
edges and retaining gray existing geometry. ROOM205 camera0 remains visibly
incomplete. A passing depth comparison cannot establish visual fidelity, and DA3
is not ground truth. Neither variant is accepted as a finished enclosure solution.
Remaining work: use render/silhouette agreement and cross-view visibility to
resolve patch boundaries, preserve openings, and assess geometry outside the
source view before integrating or declaring transfer to new rooms.

### Per-view evidence distinguishes support, occlusion and missing labels

`wall_view_evidence` in `tools/room3d_enclosure.py` classifies projected points as
unknown, wall-supported, occluded, free-space conflict, or depth-matching without
a wall label. All five depth neighbors must be valid, finite, positive and smooth;
invalid neighbors/discontinuities produce unknown rather than confident votes.
Source and other-view evidence are reported separately. Eight enclosure tests
pass, including occlusion versus free-space conflict, absent semantic labels,
invalid neighbors and discontinuities. Ruff passes.

`build/audit_enclosure_views.py` sampled 400 regular points per rectangular
proposal in the preceding ROOM205/306 experiment. Its JSON/HTML evidence lives
in `build/room3d-enclosure-view-audit/`. This is an audit of known development
rooms against estimated depth, not a held-out fidelity score.

ROOM205's two camera0 proposals have 199/400 and 159/400 source-view wall-supported
samples, but camera1 contradicts 122/400 and 128/400 respectively. Neither has
other-camera semantic wall support. This contradiction was missed by evaluating
only wall-labeled pixels in the original additive rendering report: camera1 has
no accepted wall mask, yet its depth can still identify free-space conflicts.
A missing category mask must not suppress geometric contradiction checks.

ROOM306's four camera2 proposals have 293,338,141,377 source wall-supported samples.
Other views provide no semantic wall support; they are mostly unknown, occluded,
or depth-matching without a wall label. Other-view conflicts affect 6,0,2,7 samples
respectively. The third proposal additionally contradicts source depth at39/400
samples, exposing its unsupported rectangular extension. Zero other-view support
here must not be misrepresented as either independent validation or total failure:
depth-only matches and occlusions are explicitly retained.

No proposals are promoted by this audit. Next diagnosis should test whether
camera-dependent depth scale accounts for the ROOM205 contradictions before
carving away geometry. Existing floor-based scale calibration is restricted to
appearance; any use for geometric proposals needs a separate measured experiment.

### Geometric scale ablation: disappearing contradictions are not validation

`build/probe_calibrated_enclosure.py 205` applies the existing visible-floor
scale estimator to a copied depth set before wall backprojection, leaving raw
caches and production geometry fitting unchanged. Camera scales are
1.2493/1.1893/1.2682/1.3567. This is explicitly an experiment extending an
appearance calibration hypothesis into geometry, not a supported default.
Artifacts are in `build/room3d-calibrated-enclosure-205/`.

`build/audit_calibrated_enclosure.py` performs a 2x2 comparison of raw/calibrated
geometry against raw/calibrated evaluation depth. Scaling only the evaluation
depth preserves the camera1 contradictions (122 and128 samples). Scaling geometry
moves those samples to unknown in camera1, yielding zero contradictory samples
but also zero supporting samples. Source-view support is unchanged when geometry
and depth are scaled together (199 and159): projective self-consistency cannot
prove correctness. Other cameras likewise provide no positive wall evidence.
The camera0 render was inspected: geometry remains visibly broken, including the
door region. The experiment does not establish improved reconstruction.

`compare_view_evidence` now records corresponding-sample transitions separately:
conflict to wall support, unknown, occlusion, or unlabeled depth match; remaining
and new conflicts; and lost wall support. It deliberately makes no acceptance
claim. Nine enclosure tests pass, including the regression in which122conflicts
all become unknown and none counts as resolved with positive wall evidence.
The real experiment's per-sample transitions are saved in `factorial-audit.json`.
Ruff passes for the module and tests.

Conclusion: floor scale plausibly changes placement, but these rooms lack the
independent wall evidence needed to select the geometry. Do not tune a scale to
minimize conflict counts or promote this proposal. Next, use original-camera
image correspondences/architectural edges as additional constraints, and maintain
unknown-vs-supported distinctions in any candidate selection.

### Known-camera correspondences provide an independent scale diagnostic

The existing `feature_cloud` implementation already uses mutual SIFT matching,
known-camera triangulation, reprojection error and ray-angle gates. Its collision
bounds filter is now optional (default unchanged), and reports distinguish ray
acceptance from collision-prior rejection. This is necessary for sparse-collision
rooms, but disabling the prior must never imply that every surviving match is
correct. An actual textured-plane two-camera test recovers the known2000-unit
depth with the prior disabled and demonstrates rejection by a tiny collision
footprint. The new test and45existing batch tests pass; Ruff passes.

`build/probe_known_camera_matches.py` evaluated original backgrounds in the same
three development rooms. ROOM205 yields15matches both with and without bounds:
14between cameras1/2 and1between0/1. ROOM306 andROOM311 yieldzero under both modes;
collision filtering is therefore not the reason those particular rooms lack
matches. Artifacts and overlays: `build/room3d-known-camera-matches/`. Both ROOM205
pair overlays were inspected. The1/2 cluster appears around the chimney/mantel;
the isolated0/1 point is insufficient and may reflect repeated floor texture.
No correspondence is treated as ground truth or used to fit this experiment.

`build/check_match_depths.py` compares triangulated depths for pair1/2 with raw
and previously floor-calibrated DA3. Valid-depth support exists for10points in
camera1 and9in camera2. Median triangulated/raw depth ratios are1.2574 and1.2115.
Median relative depth discrepancy decreases from20.47%to5.42% in camera1 and
17.46%to4.68% in camera2 after the existing floor calibration. These matches were
not used to fit the floor scales, so they offer additional localized evidence
for scale correction. They do not validate the camera0 wall proposals, room-wide
geometry, object identity, or held-out-room generalization. Sparse matches can
still be wrong on repeated patterns. Full records: `depth-check.json`.

Next: connect only spatially supported correspondence clusters to geometry
hypotheses, retaining source pixels, reprojection diagnostics and ambiguity;
absence of correspondences must remain unknown rather than acceptance.

### Sparse correspondence support exposes collision-derived wall occluders

`unique_correspondences` now removes pair-local observations whose pixel locations
are within2px in both views, retaining the smallest reprojection error. This
prevents multiple SIFT orientations at the same location from inflating support.
`feature_cloud` reports removed duplicates and per-pair unique counts while
retaining raw accepted counts for diagnostics. Two correspondence tests and45
batch tests pass; Ruff passes. Existing stored clouds are not rewritten.

`build/diagnose_match_occluders.py` deduplicates the saved ROOM205 camera1/2 cloud:
14observations become12unique matches, producing24camera-to-point rays. It checks
analytic finite-quad intersections against the current inferred structural
surfaces, excluding intersections within2%of the target distance. `boundary_2`
intersects10rays and `boundary_3` intersects15rays (rays may hit both). These are
visual surfaces inferred from collision bounds, not evidence that source game
collisions are incorrect. This explains why adding a correctly placed surface
behind them cannot restore its visibility.

A deterministic sparse planar proposal from10of12unique points has approximately
x1817, p90 residual96.2world units, and4138.6units of observed horizontal span.
The two remaining points are unassigned; no room-wide plane or semantic identity
is inferred. The records retain every source ray and its obstructing group in
`build/room3d-match-occluders-205/report.json`. This localizes candidate corrections
to visual wall surfaces, instead of continuing to tune a global depth scale.
No scene is modified in this diagnostic, and correspondence errors remain possible.

Next experiment: revise only the portions of inferred wall surfaces that obscure
the supported planar region, preserve original game collisions and unsupported
regions, and compare both source-camera renders and untouched cameras. Sparse
point support alone cannot justify deleting an entire wall or flattening the
chimney geometry.

### Local visual cutout experiment: visibility improves, appearance rejected

`build/probe_local_wall_correction.py` constructs a reversible recipe from the
same10inlier triangulated matches. For each existing structural quad it intersects
the source rays and proposes a rectangular local cutout only when at least3unique
points and2cameras support an obstruction. The provisional25-world-unit margin
is not calibrated image uncertainty. Outside rectangles retain their original
geometry/material/group; a bounded inferred back-plane patch is added. Original
survey/collision records and production recipes are unchanged.

Two quads qualify: boundary_2 surface4 (4points,2views) and boundary_3 surface7
(6points,2views). Artifacts, recipes, comparison HTML and all camera renders live
at `build/room3d-local-wall-correction-205/`. Of10fitting feature rays in each of
cameras1/2,0were unobstructed before and7after;3remain obstructed in each view.
This is fitting-set visibility, not independent validation. Coverage stays76800
pixels in all four views and therefore provides no success evidence.

All four comparison rows were visually inspected. Cameras1/2 acquire visible
rectangular gray cuts around the chimney; existing door, armor and floor defects
remain. The result is rejected for visual regression despite its improved ray
metric. Per-camera changed-pixel counts are retained in report.json, together
with the audit. No generic cutout operation or default pipeline behavior is
promoted from this experiment.

This verifies a causal obstruction but also shows why sparse point rectangles
cannot define a continuous room surface. Further work should fit a coherent wall
and attached chimney jointly, with image edges and visibility constraints, instead
of accumulating holes around individual feature clusters. The original five-part
object reconstruction/evaluation goal remains incomplete.

### Continuous wall plus attached-volume hypothesis

`build/probe_continuous_wall.py` replaces the preceding cutout experiment with
a continuous backing plane over the observed lateral interval, extending it to
the existing visual floor/ceiling heights (unobserved vertical extent is estimated).
An attached box reuses the obstruction footprint while searching heights1000,
1500and2000. The experiment explicitly selects boundary_2 as backing geometry
and boundary_3 as its attachment from the preceding ray diagnosis; this selection
is not yet an automatic general batcher rule. Game collision records remain
untouched. Artifacts: `build/room3d-continuous-wall-205/`.

All three variants expose9/10fitting feature rays in each of cameras1/2, versus
0/10baseline and7/10local-cutout. Cameras0/3 have identical image scores; neither
contains the feature observations. Four rows were visually inspected in
comparison.png: the continuous wall removes cutout-like gray rectangles, while
the attached box remains an approximation and texture seams persist. Height2000
visibly distorts the hearth more than the lower alternatives. Existing armor,
door and floor defects remain substantial.

`build/score_continuous_wall.py` records source-image RGB MAE and reference-edge
distance without presenting them as 3D fidelity. Height1500 is the development
compromise: camera1 RGB MAE13.346→12.800 and edge distance2.900→2.863; camera2 RGB
MAE11.491→11.545 (slight regression) and edge distance6.380→6.259. Height2000 wins
camera1 RGB alone but worsens camera2 RGB to12.134 and edge distance to6.407.
All textures reuse original images; all correspondence cameras participated in
geometry, so none of these measurements is held-out validation.

The causal experiment supports a continuous backing wall with an attached volume
over sparse cutouts. Next work is a reusable proposal rule deriving the backing
surface and attachment from ray/footprint relations, with explicit ambiguity,
rather than hardcoded boundary IDs. It must then be tested outside this room
before enabling any default behavior. No production integration or full objective
completion is claimed.

### Automatic backing/attachment identification

`identify_wall_attachment` in `tools/room3d_enclosure.py` replaces the ROOM205
boundary-ID choice with an explicit topology proposal. It requires at least3
unique occluded targets and2views per boundary, a near-axis-aligned supporting
plane, a backing footprint spanning that plane's lateral evidence, and a narrower
intersecting footprint protruding from exactly one side. Extent comparisons scale
with observed plane span. Multiple plausible pairs return `ambiguous`; no pair
returns `no_hypothesis`. It does not infer category, height or correctness.

Three new tests in `tests/test_room3d_attachment.py` cover renamed IDs, axis swap,
three scene scales, input preservation, missing/duplicate evidence, and ambiguous
attachments. These plus9enclosure tests pass; Ruff passes. ROOM205 independently
returns backing2/attachment3 from the saved obstruction evidence.
`build/check_automatic_attachment_recipe.py` verifies that this selection produces
exactly the three already-rendered development recipes (1000/1500/2000). Its
recipe emitter explicitly supports only positive-X attachments and rejects other
orientations; the identifier itself handles both footprint axes. This limitation
must be removed before general integration. Verification artifact:
`build/room3d-continuous-wall-205/automatic-selection.json`.

`build/probe_attachment_transfer.py` stress-tests the same identification chain on
known development ROOM201 andROOM206 without fitting room-specific parameters.
ROOM201 supplies13unique matches and1sparse planar proposal, but no eligible
attachment; ROOM206 supplies11matches and no qualifying plane. JSON retains all
pair counts/proposals/ray evidence in `build/room3d-attachment-transfer/`. These
negative results prevent treating the ROOM205 success as generalized reconstruction.
No geometry changes or production defaults were applied to those rooms.

Next: generalized recipe emission with consistent local coordinates and an
explicit candidate-selection audit. The original stairs/railings/lamps goals and
held-out-room evaluation remain incomplete; attachment identification is supporting
architecture work, not a replacement completion criterion.

### General local-coordinate attachment recipe generator

`tools/room3d_wall_attachment.py` now emits the continuous-wall/attachment proposal
from the identified pair, supporting both footprint axes and both protrusion
sides. It trims only overlapping front backing segments, preserves segments
outside the observed lateral interval and unrelated recipe entries, and follows
the fitted plane at the attachment's rear edge. The existing configured floor_y
is respected. Vertical wall continuation and attachment height remain estimates.
The complete attachment must lie within the observed lateral interval; partial
coverage is rejected instead of silently erasing unsupported geometry.

The mesh tiler assumes parallelograms. A tilted backing plane therefore produces
a parallel front edge (constant attachment depth) so every emitted quad satisfies
that constraint; an axis-aligned front paired with a tilted back would produce
trapezoids and incorrect tiling. The collision footprint's front position is
matched at its midpoint, with the small endpoint displacement explicitly part of
the approximate shape. Original game collisions are not modified.

Five attachment tests pass (three identifier tests and two emitter tests), covering
four orientations, nonzero floor, outside-region preservation, hidden-group data,
input immutability, height bounds, tilted-plane parallelograms and rejected partial
extent. Ruff passes for the new module and tests.

`build/probe_general_attachment_v2.py` uses the automatic identifier and this new
generator on ROOM205 at height1500. Artifact:
`build/room3d-general-attachment-v2-205/`. It retains9/10unobstructed fitting rays
in cameras1/2; camera2 was visually checked and remains a rough textured block,
with the same substantial surrounding room defects. This establishes that the
reusable generator reproduces the local visibility benefit, not that it solves
the room or generalizes visually. The earlier v1artifact predates the tiler-shape
fix and should not be used as the current recipe.

The generator is not yet wired into default batch processing. Candidate selection,
held-out validation, broader room reconstruction and the original object-specific
goal remain outstanding.

### Batcher integration: audited attachment alternatives, default off

`propose_attachments` in `tools/room3d_wall_attachment.py` now composes unique
feature observations, sparse plane extraction, finite ray-obstruction checks,
topology identification and generalized recipe emission. It retains source rays,
plane membership, ambiguous/no-hypothesis outcomes and explicit emitter rejection
reasons. Heights are2/3,1and4/3of obstacle_height (default1500). Candidates remain
unselected; generating them is not an acceptance decision.

The image-assisted batcher recognizes recipe option `wall_attachment_proposals:
true` after structural consolidation. It computes a separate unbounded feature
cloud for this diagnostic and writes `wall-attachments/feature-cloud.json`,
`report.json`, and per-alternative recipe JSON/OBJ files. It does not mutate the
active fitted recipe or replace assisted.obj. The normal collision-bounded cloud
and existing reconstruction flow retain their defaults. The option is off when
absent. There is not yet an automatic texture/render-based selector for these
alternatives; the artifacts are preparation for that required evaluation.

`build/run_attachment_batch_probe.py` ran the actual `process` entry point to
completion on ROOM205 using the frozen Stage2 local detector/segmentation/depth
caches, with parametric details disabled for this architecture integration probe.
Manifest and all outputs: `build/room3d-attachment-batch-probe/`. It found13unique
feature matches, identified backing2/attachment3 from ray evidence, and emitted
three alternatives at1000/1500/2000. No room-specific boundary IDs occur in the
pipeline composition. This plane differs slightly from the earlier hand-isolated
camera-pair experiment because the automatic chain uses all eligible room matches;
its quality must therefore be evaluated anew, not inherited from prior renders.

Five attachment tests and45batch tests pass; Ruff passes for the changed source
files. The next required step is evaluating the emitted alternatives through the
same cameras with appearance and per-view regression evidence, including explicit
insufficient-evidence outcomes. No default promotion, held-out generalization,
or completion of the original five-part goal is claimed.

### Common-region render evaluation and provisional ranking

`tools/room3d_attachment_evaluation.py` bakes and renders baseline/alternatives
with the same appearance settings, then fixes one common region per camera: the
union of depth/coverage changes across all alternatives, dilated by one pixel.
Every candidate is scored on exactly that region. Metrics are normalized RGB
MAE, reference-edge distance and newly uncovered pixels. Per-view regression
limits prevent an average gain from hiding a bad camera; fewer than64affected
pixels or no material image improvement yields no recommendation. Nonfinite
metrics are rejected. Thresholds are explicit diagnostic heuristics, not learned
confidence or fidelity percentages.

The batcher option `wall_attachment_evaluation: true`, alongside
`wall_attachment_proposals: true`, runs this comparison and writes an HTML gallery,
region overlays, candidate renders, bake reports and evaluation.json under
`wall-attachments/evaluation/`. It reports a provisional recommendation but does
not apply it to the active recipe. Optional absent depth input is handled. Four
ranking tests pass (single-camera regression despite average gain, empty evidence,
new holes, NaN rejection); the previously run5attachment and45batch tests pass.
Ruff passes for changed modules/tests.

`build/evaluate_batch_attachments.py` evaluated all three actual ROOM205 batcher
proposals with the frozen model caches. It recommends plane0_height1000 under
these source-image diagnostics. The fixed camera1region has5809pixels: baseline
RGB MAE.07888→.05074 and edge distance1.385→.890. Camera2has14784pixels: RGB
.03512→.03146 and edge distance1.791→1.193. Cameras0/3 have no affected pixels,
so they supply no evidence for selecting a height. All three heights meet these
bounds;1000has the largest mean RGB gain. The prior hand-isolated-camera experiment
favored1500, demonstrating why its result could not be inherited by the automatic
all-match proposal. Camera2's recommended render was visually inspected: the
local wall is more continuous, but the hearth remains a textured approximation,
with substantial armor and other room errors.

The score reuses original images for both texturing and comparison; it is not
held-out-camera or novel-view validation. No automatic scene promotion or complete
goal claim follows. Next work must evaluate support/visibility outside the fitting
views and test this full proposal/evaluation chain on other rooms, while returning
to the original stairs/railings/lamps requirements rather than treating this
architectural subproblem as the whole goal.

### Return to object goal: correspondence coverage and missing stair diagnosis

`build/audit_object_correspondences.py` intersects all deduplicated known-camera
SIFT points with the saved semantic masks in ROOM201/205/206. The rooms have
13/13/11matches respectively, but zero points fall within reliable staircase,
railing or lamp masks in both views. ROOM201's matches lie on window/painting
labels; ROOM205's on wall/chest-like labels or unlabeled pixels; ROOM206's on
wooden floor. Artifacts: `build/room3d-object-correspondence-audit/`. Architectural
correspondences must not be claimed as evidence for shared lamps or stairs.

`build/probe_stair_support_201.py` tests the missed camera1 staircase using its
original mask, an11x11dilation, and the mask bounding rectangle. The unchanged
flight-region algorithm yields zero candidates in all three. The original image
was visually inspected: the flight is dark and partly occluded by railing.
A separate temporary three-edge diagnostic (`build/probe_stair_support_three.py`,
`build/room3d_stair_edges_three_probe.py`) changes only the minimum run length and
adds gate logging; production thresholds remain untouched. It also yields zero
candidates. `build/room3d-stair-support-three-201/gate-audit.json` identifies why:
three/four-line groups have horizontal spans1360–1527world units but only40–78units
of reconstructed vertical change. They fail the existing300-unit minimum rise.
Thus lowered detector confidence or broader masks alone cannot recover this
stair: DA3 is effectively flattening the observed tread sequence.

Next aligned object experiment is an explicitly approximate regular stair prior
fit to the observed tread projection, with competing floor/railing hypotheses and
camera/occlusion checks. Its rise must not be forced to the flattened monocular
estimate. Image-only tread evidence remains ambiguous; this finding is a diagnosis,
not acceptance of a stair or justification to lower all production gates.

### Image-space regular stair prior versus flat surface

Temporary diagnostic copies of the stair-edge extractor test image evidence
without enforcing DA3's flattened vertical change. Visual inspection of the
original-threshold result showed four lines were actually paired edges of only
two treads; they cannot be counted as four independent steps. A lower-contrast
probe (Canny10/30, Hough8, minimum3px line separation) finds a plausible sequence
of three distinct tread edges in the original ROOM201 camera1 mask. These changes
remain only in build/probe scripts, not production thresholds.

`build/fit_stair_image_prior.py` fits regular parallel tread segments through the
original camera for flat and signed-rise hypotheses. It optimizes position,
width, yaw and run, treating observed short segments as possibly occluded rather
than forcing full width to them. The first sequence yields1.944pxRMS for a flat
model,1.553for rise200/run296, and1.434for rise250/run229. Negative rises fit worse.
A second line sequence provides little flat/stair separation (1.917versus1.839).
Artifacts and red-observation/yellow-projection overlays:
`build/room3d-stair-image-fit-201/`.

The best overlay was inspected and aligns the three tread edges reasonably, but
positive-rise fits hit the lower X-position search bound. Their location changes
by several thousand world units relative to the flat fit. A half-pixel image gain
therefore does not establish a usable3Dflight or its placement. The report labels
this underconstrained and unaccepted; neither scene geometry nor production gates
were changed. This is a diagnostic of how a regular prior can escape flattened
monocular depth, not evidence that the missing staircase is solved.

Next: constrain the flight origin with independent landing/floor or gameplay
height evidence, and retain competing regular/floor hypotheses until camera and
support constraints separate them. Do not merely widen search bounds until an
image-only fit reports success.

### Stair entrance anchor hypothesis from existing extracted interaction data

The existing ROOM201 `scene-anchors.json` has four transition zones, including
zone[11345,10300,2605,2500], transition_type24, source_offset268808 near the pictured
flight. It contains no `stair_height` declaration. Inspection used existing source
and previously decoded data only; no new binary reverse engineering or claimed
decoding of type24 was performed. Its use as a stair entrance is a development
hypothesis, not a verified height/landing rule.

`build/fit_stair_anchored_prior.py` constrains the first observed tread's XZ center
to that zone and assumes existing visual floor_y0 with first tread at floor+rise.
It repeats the image fit for rise0/150/200/250. Artifact:
`build/room3d-stair-anchored-fit-201/`. RMS errors are1.944/1.792/1.738/1.683px;
runs969/789/729/669units. The positive-rise XZ solutions stay inside the zone,
unlike the earlier unbounded fit. Height250places the first center around
[12732.7,250,11297.8], width1936.4. Its overlay was visually inspected and remains
consistent with the selected three image edges, but does not identify true rise,
width, world height or object extent. Anchor provenance and assumptions are saved
explicitly in report.json. No production change or accepted staircase follows.

A separate integration limitation was confirmed in current `generate_mesh`: the
room floor is emitted as one continuous rectangle, with no floor-opening recipe
support. A descending flight below floor_y0would therefore be hidden even if its
shape fit improves. Next aligned implementation is explicit bounded visual floor
openings associated with stair hypotheses, preserving gameplay collisions and
surrounding floor. Opening placement must remain coupled to the candidate flight;
it is not justified by a generic interaction rectangle alone.

### Visual floor openings for descending stair proposals

`tools/room3d_floor.py` subtracts explicit convex XZ openings from a tiled room
floor using half-plane clipping. It supports rotated openings, overlap/union,
repeated and out-of-bounds openings, preserves floor winding, and rejects malformed,
concave or self-intersecting polygons. Tiling bounds texture triangles without
assuming clipped polygons are parallelograms. Current Room3D.cpp OBJ loading was
checked: polygon faces are triangulated as a fan, compatible with these convex
patches. No runtime code or gameplay collision records change.

`generate_mesh` consumes optional `floor_openings` entries with a `polygon` field;
recipes without entries retain the prior floor path. The experimental stair
recipe records `source_object: stair_prior` and derives its opening directly from
the three fitted tread edges/advance, not from the entire transition trigger.
This link is provenance, not automatic validation of the stair hypothesis.
Three floor tests plus45batch tests pass. Floor tests cover area conservation,
rotated geometry, winding, input preservation, duplicate/overlapping cuts, exterior
openings and invalid polygons; Ruff passes.

`build/probe_descending_floor.py` generates closed/opened versions of the bounded
ROOM201 three-tread hypothesis without projected textures. It renders both the
isolated floor/stair and full scene. With the floor closed, all three tread-center
rays are occluded; with the opening, the first two match their expected tread-plane
depth and the third remains hidden by the floor edge. Expected depth is evaluated
at the rasterizer's pixel center to avoid mistaking sampling offsets for occlusion.
Artifact: `build/room3d-descending-floor-201/`. Both opened isolated and full-scene
renders were inspected: the floor cut works, but the full room still has large
incorrect collision-derived visual obstacles. This is functional floor-opening
validation, not successful reconstruction or a playable stair.

The third tread's visibility requires a better-supported opening/landing extent,
not arbitrary enlargement. Original height/position assumptions remain unresolved;
no default stair acceptance or goal completion follows.

### Descending shaft: blocking proxies and explicit creative completion

`build/audit_stair_occluders.py` renders individual visual groups at the three
provisional tread-center pixels. boundary_6andboundary_7 both obscure all three,
by roughly3340–3681depth units. They are non-wall collision-derived solid proxies
under existing generator rules; original gameplay collision records are not
changed or reinterpreted as proven room geometry. The floor also hides tread3.
Records: `build/room3d-descending-floor-201/occluders.json`.

`build/probe_stair_shaft.py` derives removable proxy IDs from that audit, compares
removal alone and an opening bounded by the smallest such collision footprint
containing the flight. Both boundaries6/7contain it; the smaller7is used. Removing
proxies exposes2/3sampled treads; the larger shaft opening exposes3/3. All eight
camera rows were visually inspected in `build/room3d-stair-shaft-201/comparison.png`.
The opening introduces missing coverage in cameras0/1/2 (72067/75065/75078pixels
covered, versus76800baseline), while others remain full. This is not an acceptable
finished scene and motivates explicit estimated lower geometry rather than black
voids or restoring false solid blocks.

`build/complete_stair_shaft.py` extends the same regular flight direction until
the containing footprint boundary, yielding6steps at250rise (bottom_y1500), and
adds estimated vertical shaft lining plus a lower landing. Only3tread image edges
were observed; additional steps and lower surfaces are creative hypotheses, not
recovered detail. Artifacts: `build/room3d-stair-shaft-completion-201/`. All eight
renders regain full coverage; camera1geometry was visually inspected. Uniform
materials do not establish appearance or fine geometry, and full coverage is not
fidelity. Side railings remain solid proxies and the upper-floor/entrance assumptions
remain unverified. No production recipe/default is changed by these scripts.

Next useful validation is texture projection and cross-camera comparison of this
continuous descending-flight hypothesis, then replacing only supported railing
proxies with reusable railing geometry. This remains work toward the original
stair/railing objective, not its completion.

### Full-room operational pass (2026-10-11)

User prioritizes approximate/creative results for every room, followed by separate
manual retouching, rather than further tuning ROOM201 before processing others.
The previous turn produced new evidence: all348RDT entries exported in mesh-only
mode (320nonempty entries,28empty slots,0errors). These include scenario variants;
this is not an all-room image-assisted success claim.

`batch_room3d.py --resume` now uses content-checked per-room checkpoints. Verified
on the complete mesh batch: a repeat reused all348entries;45existing batch tests
and3checkpoint tests passed. Failed/partial rooms are not cached, output changes
invalidate reuse, and recipe changes invalidate their respective room. Shared
code/assets invalidate all checkpoints. Model weights/environments are outside
the fingerprint; changes there require running without resume.

Started the full local-model pass with `build/room3d-all-assisted-recipes.json`
(parametric_details, creative_fallback, stair_refinement, joint_stair_supports),
output `build/room3d-all-assisted`, log `build/room3d-all-assisted.log`. Network
model access is disabled via HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE. At this entry
it is processing ROOM1000; completion and per-room quality remain unverified.
Inspect the live process and log before deciding whether to resume it.

Still needed: finish the all-room pass and inspect failures; provide a simple
per-room retouch workflow that preserves automatic results and user corrections;
verify representative retouches and newly processed rooms. Original shared-object,
geometry/appearance and generalization requirements remain open where the earlier
evidence was negative. No claim that every room will need only small changes.

### Independent manual retouch exports (2026-10-11)

Added `room3d_retouch.py`: lists generated OBJ groups and applies per-group
translation, positive scale, yaw, custom pivot or hiding without model inference.
Each new variant preserves textures and camera cutaways, saves edits/provenance
and renders original/automatic/retouched comparisons. Source outputs remain
unchanged. Shared vertices are split so moving one group cannot move another.
This is a JSON/CLI workflow, not yet a graphical editor; it cannot repair missing
surfaces or update gameplay collision. Moved projected appearance includes baked
lighting. Multiple groups belonging to one object require explicit joint edits.

Three tests cover group isolation with shared vertices, UV/color preservation,
rotation/scaling, invalid edits, hiding and local material dependencies. Ruff
passes. Real ROOM205 identity run: maximum pixel delta0 on camera1 and hashes of
all source files unchanged (`build/room3d-retouch-identity-205/verification.json`).
A translated/scaled door test changes2396pixels in camera0 and0in cameras1–3;
camera0 and3 comparison sheets visually inspected. This demonstrates applying
retouches, not a recommended visual improvement or in-game validation.
Artifacts: `build/room3d-retouch-probe-205/index.html`.

Full-room batch session39534 remains live (PID661673 checked directly). ROOM1000
completed DINO/SAM/DA3 and detail fitting and is performing the local4B visual
review. No full room checkpoint was complete at this inspection. Keep monitoring
the existing process; do not restart because individual inference stages are slow.

### Offline transform editor (2026-10-11)

`room3d_editor.py` generates a standalone HTML from a completed room; the
`room3d_editor.html` template provides numeric position/scale/pivot/yaw controls,
group/camera selection, visibility, reset and JSON import/export. Selected-group
wireframes project over original images; they deliberately do not claim texture
or occlusion preview. Textured comparisons still use `room3d_retouch.py`. No
server, network or input mutation is needed. Example generated artifact:
`build/room3d-retouch-editor-205.html` (10groups,4cameras).

Node checks cover transform convention, near clipping and input validation.
The Chrome test found HTML step validation rejected valid initial pivots/scales;
changed inputs to allow arbitrary finite decimals and verified the controls in
Chrome again. The reproducible test also passes decimal translation312.5,
camera switching, hiding and reset. Screenshot inspected at
`build/room3d-editor-ui.png`; controls and selected door wireframe render.
This closes a basic visual editing path, not direct manipulation, topology editing,
automatic shared-object grouping or in-game collision validation.

Full batch remains live. ROOM1000 camera0 local4B review took107.8seconds CPU;
camera1 also finished and camera2 review is running at this inspection. This is
an observed runtime bottleneck, not a hung process. No all-room completion claim.

### Deferred review pass (2026-10-11)

The first Qwen4B review measured107.8seconds CPU per camera. Added
`--defer-review` to keep model-based reconstruction and render fitting enabled
while writing hash-bound per-camera review queues. `room3d_review_queue.py`
processes those separately, reuses exact existing reviews, retries interrupted
review files and refuses stale images. Reports live outside room exports so
review completion does not invalidate geometry checkpoints. Queued review is
explicitly pending, not successful validation.

Validation:45batch tests and2queue tests passed; Ruff passed. Actual ROOM1000
with cached DINO/SAM/DA3 completed image_assisted,6cameras,6queued jobs and a
checkpoint in `build/room3d-deferred-smoke`. Separate reviewer reused all4completed
synchronous reviews by exact hashes and is processing cameras4/5. It has not
completed yet. This preserves review work while freeing geometry generation.

Deliberately terminated the synchronous batch PID661673 (session39534 confirmed
exit143), retaining `build/room3d-all-assisted` and its inputs/reviews. This was a
workflow replacement after successful validation, not a timeout-triggered retry.
New full generation: session21374 / PID672218, output `build/room3d-all-generated`,
log `build/room3d-all-generated.log`, same all-room recipes and all local models,
with --defer-review --resume. Separate smoke review: session77587 / PID672272,
output `build/room3d-deferred-reviews`, log `build/room3d-deferred-reviews.log`.
Both processes confirmed live at this entry. Continue monitoring these handles;
full-room completion, remaining review jobs and visual quality are still open.

### Room catalog and ROOM1000 inspection (2026-10-11)

Inspected ROOM1000 camera2 comparison and detail evidence. A wall molding was
recognized as railing and converted to a creative rail. Its evidence fit was
rejected, but the creative placement/depth gates accepted the fallback. This is
a semantic false positive, not proof of faithful railing reconstruction. Another
rail proposal in camera5 was rejected. Both objects are single-view with no
reserved evaluation camera. General shared-object/fidelity requirements remain
unproven; no acceptance gate was weakened to hide this finding.

Added `room3d_catalog.py` to aggregate completion and review signals and generate
per-room offline editors outside batch outputs. It distinguishes baseline-only,
pending/partial, changed/incomplete and image-assisted exports; lists creative,
single-view and rejected details; can link separately completed visual reviews.
Export status checks matching RDT/checkpoint and key artifact hashes, not visual
quality. Two tests pass for stale/missing exports and detail summaries; Ruff passes.
Smoke catalog contains1exported room,319pending,28empty slots.

Live catalog watcher: session81549, output `build/room3d-catalog`,
log `build/room3d-catalog.log`, refresh30seconds. Full-generation session21374/
PID672218 remains live, still completing ROOM1000 at this entry. Separate review
session77587/PID672272 has completed camera4 and is working on camera5. Old
synchronous session39534 is terminal; do not confuse it with the replacement.

### First completed production room and retouch verification (2026-10-11)

The new full batch has completed ROOM1000 (image_assisted checkpoint,6cameras)
and is processing ROOM1001. Its automatically generated catalog editor passed
real Chrome controls tests. The separate smoke review finished all6cameras;
all6reports were reused by exact image hashes into `build/room3d-all-reviews`,
confirming equality of reviewed inputs between smoke and production outputs.
Session77587 is terminal exit0; session21374 and catalog session81549 remain live.

Applied a deliberately manual correction to the production room: hide
`primitive_3`, the false-positive creative rail. Outputs in
`build/room1000-retouch-no-false-railing`. Camera2 comparison visually inspected:
rail removed, but gray region exposed. Camera0changes519pixels, camera2changes1902,
others0. This demonstrates editing a shared mesh across cameras without inference,
not automatic semantic repair.

Added optional `room3d_retouch.py --reproject` with existing room SAM/DA3 roots.
It rebakes appearance after transforms, retains camera cutaways and emits bake
metadata. Real run completed in `build/room1000-retouch-reprojected`; camera2
inspected. Reprojection did NOT substantially restore this gray wall: full-image
RGBMAE5.8433->5.8405/255, a negligible same-camera appearance change, not fidelity.
Underlying geometry/semantic support still needs repair. All original room files
match checkpoint hashes after both retouches (`verification.json`). Existing
3retouch tests and Ruff pass. Keep the failed visual expectation explicit.

### Preview cost measurement and exact-output optimization (2026-10-11)

Profiled3ROOM1000 renders:3.261s under cProfile, including0.953s OBJ/material
loading and0.705s near-plane clipping (23952calls). Replaced per-face np.roll
allocation with indexed edges and fast all-inside/all-outside clipping; reused
the default material array rather than allocating it for every triangle.

The clipping regression checks400polygons including near-plane boundary points,
all attributes, vertex order and input immutability. It and45existing batch tests
pass; Ruff passes. Compared10real views (ROOM1000 andROOM205) against a frozen
pre-change renderer: RGB, coverage AND depth arrays are exactly equal in every
view. Alternating measured total7.794s->6.587s (~15.5%less renderer time), while
other batch work ran concurrently. This is not a15.5%whole-pipeline speedup.
Evidence: `build/room3d-render-optimization.json`, profile text/prof adjacent.

The existing full batch retains its already-imported renderer; it was not
restarted for this optimization. PID672218/session21374 remains live with2completed
entries and ROOM1010 processing at this inspection. The change benefits future
processes and separately invoked previews without lowering quality gates.

### Complete background decoding audit (2026-10-11)

Validated every referenced background, independent of the ongoing expensive model
pass. First matched all RDT content hashes against the complete survey inventory,
then resolved actual case-sensitive asset paths and decoded each distinct PAK via
the production `background()` function. Result:620distinct files,1684camera uses,
620successful normalized320x240decodes,0errors,68.16seconds. Saved source/RGB hashes
and room-camera references in `build/room3d-background-validation.json`;
reproducer `build/validate_all_room_backgrounds.py`. Session87470 is terminal0.
This rules out missing/unsupported backgrounds for these inputs, not geometry,
model or runtime errors. Shared backgrounds explain why camera-reference counts
exceed file counts; no scenario or room reconstruction was skipped.

Full batch PID672218/session21374 remains live,2exports complete, ROOM1010
processing, no recorded batch exception at inspection. Catalog watcher remains
live. This turn produced a complete input validation result while preserving the
running reconstruction job; goal completion is still unproven.

### Exact image-pair review reuse across variants (2026-10-11)

Compared ROOM1000 and ROOM1001 queues: all6ordered reference/render image pairs
have identical hashes. Previously the reviewer cache required the same room folder
and would unnecessarily rerun inference for the other scenario. Added an index
across completed review outputs keyed by both image hashes, requiring the pinned
4B model/revision and exact prompt before indexing and verifying again on reuse.
It also incorporates reviews completed earlier in the same queue run.

Three queue tests pass, covering stale input rejection, interrupted-record retry,
exact cross-room/camera reuse and changed-prompt rejection; Ruff passes. Actual
ROOM1001 review pass completed6/6 with `reused:true`, no model inference. This
reuses only review evidence; both rooms were independently reconstructed. Reports
are in `build/room3d-all-reviews/room1001` and catalog refresh shows their counts.

Full-generation PID672218/session21374 is still live; ROOM1010 is in parametric
detail fitting and has not emitted a final details report. Its output is19MB,
whole running batch81MB, no exception recorded. Do not treat slower fitting as
terminal or restart it solely because observation time elapsed.

### Verified ROOM1010 fitting progress (2026-10-11)

Recomputed observation association from the existing hashed SAM/DA3 inputs:
8groups (6lamps,1railing,1staircase). Seven object result reports are written;
the eighth staircase group is actively rewriting candidate artifacts. CPU time
and output modification times advanced across direct PID/session checks and a
30second wait on session21374. No terminal state or exception was observed.
Snapshot: `build/room1010-fitting-progress.json`. Its per-object acceptance values
come from nested report fields; completion of an optimization report is not room
acceptance. No restart or reduction of fitting gates was made merely because
this room takes longer than ROOM1000. Full batch still incomplete.

The last inspection above then observed `details/results.json` newly present:
all8object reports are now finished. This is forward progress into final export,
not completion of the full-room batch.

### ROOM1010 shared fixture transfer evidence (2026-10-11)

ROOM1010 is now a completed production export; catalog shows3exported entries
and ROOM1011 processing. Of8detail groups, only1was accepted: a wall lamp shared
by observations in cameras4and5. It uses a single final mesh group `primitive_4`,
fixture_attachment_prior, wall_sconce profile,1arm. Silhouette IoU after fitting
is0.7545/0.6010 on those training views; projected coverage190/929pixels. All listed
placement/depth/attachment gates passed, but the report correctly marks procedural
interpretation rather than recovered original design. No third held-out view
exists, so this is not independent fidelity validation. The staircase and other
6proposals were rejected.

Visually inspected original/before/after and a colored isolated-group overlay:
`build/room1010-shared-lamp-review.png` and
`build/room1010-shared-lamp-textured.png`. The same simple shade occupies the lamp
region in both views, but texture distortion and large architectural gaps remain
in the complete scene. This supports limited shared-object transfer, not overall
room quality. The overlay is isolated projected silhouette, not an occlusion map.
The observed result improves evidence for the original shared-object objective
without redefining a mask-fit score as fidelity.

Started independent6camera Qwen review for ROOM1010: session57170, log
`build/room3d-room1010-reviews.log`, output `build/room3d-all-reviews/room1010`.
Full reconstruction session21374 and catalog watcher81549 remain live. Goal stays
active; all-room completion and broader shared-object generalization are unproven.

### Independent room asset packaging (2026-10-11)

Read the existing Room3D.cpp loader: it resolves MTL and map_Kd relative to the
OBJ directory and expects StageN/roomNNNN.obj plus .cameras. Batcher exports reuse
filenames such as assisted.mtl and source-camera0.png, so flattening multiple room
outputs into one stage directory would overwrite dependencies.

Added `package_room3d.py` to stage completed exports into a new directory, with
per-room material/texture paths, unchanged mesh groups/camera cutaways, optional
manual-variant overrides and explicit current loader size/count limits. It does
not install or alter the game. Two tests pass for conflicting source filenames,
independent red/blue textures, exact rendered pixels, preserved camera sidecars,
missing dependencies and invalid room names. Ruff passes.

Actual3room bundle: `build/room3d-package-probe`. Selected cameras from ROOM1000,
ROOM1001 and ROOM1010 have exactly identical RGB, coverage and depth after path
rewriting (`verification.json`). No game-runtime validation was performed and no
claim of playable completeness is made. Full generation and ROOM1010 deferred
review remain active; this closes the asset-name collision risk for packaging.

### Two-worker full-batch continuation (2026-10-11)

Measured available resources:20logical CPUs,32GBRAM,~21GBavailable at inspection.
Added --jobs to batch_room3d using spawned independent room processes. Parent owns
the aggregate manifest/index; each worker retains per-room error isolation,
model directories and checkpoints. Algorithms/gates are unchanged. Runtime GUI
probes require1worker. Worker count is excluded from checkpoint fingerprints.

Validation: the spawned-process test covers ordered aggregation, switching2->1
workers with resume, and one invalid room alongside a successful empty slot;
45existing batch tests also pass and Ruff passes. A real2-worker reconstruction
of ROOM1000/1001 using cached model inputs finished successfully. Fitted/assisted
OBJ, camera sidecars and all12assisted PNGs exactly match serial exports byte for
byte (`build/room3d-parallel-smoke/verification.json`). No whole-batch speedup
percentage claimed from this small comparison.

After ROOM1011 completed, deliberately stopped serial PID672218/session21374;
exit143 verified, child694884 terminated, outputs retained. Verified all artifacts
of4completed rooms against their original checkpoints and RDT hashes. Kept their
original code keys unchanged. One-off continuation plan:
`build/room3d-parallel-plan.json`,4preserved exports +344pending RDT entries
(316nonempty rooms and28empty slots). The driver merges preserved results back
into the final manifest/index when the remaining batch finishes.

Current full generation: session85941, driverPID695424,
batchPID695450, workers695479/695482 (695478is resource tracker), log
`build/room3d-all-parallel.log`, same `build/room3d-all-generated` output.
Driver: `build/run_room3d_parallel_remaining.py`. Both first pending rooms
ROOM1020/1021 are starting. Old21374 is terminal; do not poll it as the current job.
Catalog watcher81549 remains live and needs no restart. ROOM1010 visual review
session57170 remains active (five camera reports complete at last inspection).

### Parallel worker and review follow-up (2026-10-11)

Revalidated session85941 and worker processes: ROOM1020/1021 both completed
DINO/SAM/DA3 and stair-edge preprocessing and entered detail fitting, with one
object directory started each. No batch exception recorded. A30second wait on
the confirmed live generation handle did not indicate termination; no restart.

ROOM1010 reviewer session57170 finished exit0 (all6reports). ROOM1011's6queued
image pairs are exact matches; ran its review queue and all6reports reused the
existing results without inference. Thus the4completed exports have24per-camera
review records, still unverified model proposals rather than approval. Current
model memory/resources are available; no model subprocess remained active at
inspection because generation workers had moved into geometry fitting.
Active generation remains85941 (driver695424/batch695450/workers695479,695482),
catalog81549. Reviewer57170 is now terminal. Full-room goal remains incomplete.

### ROOM1020/1021 export milestone (2026-10-11)

Both initial parallel tasks completed image_assisted; catalog now6exported entries,
2in progress,312pending nonempty entries plus28empty slots. Each ROOM102 variant
has4cameras,13968vertices,4656triangles,21textures and23materials; both pass the
packager's existing runtime format/count/size/dependency checks. There was1detail
proposal per room and0accepted, so export success is not a detail-reconstruction
success claim. Audit: `build/room102-export-audit.json`.

Their4ordered review image pairs match exactly. Started one sequential queue
for room1020 and room1021, which can reuse completed identical pair reviews:
session56458, log `build/room3d-room102-reviews.log`, output root
`build/room3d-all-reviews`. It is pending, not completed. Full-generation85941
and catalog81549 remain active; no new generation exceptions observed.

### Isolated GPU reviewer feasibility (2026-10-11)

Full generation remains live (driver 695424, batch 695450); six exports
completed at inspection. New isolated `build/room3d-review-gpu-venv` uses
CUDA torch 2.6.0+cu124 with transformers 4.57.1, bitsandbytes 0.50.2 and
accelerate 1.15.0. Existing running environments were not modified.
`build/probe_room3d_review_gpu.py` tests pinned Qwen3-VL-4B locally with
NF4 weights; outputs stay separate from CPU review caches. Identical-image
control returned NO_CLEAR_DIFFERENCE in 1.98 seconds of generation, with
3,222,738,944 peak allocated CUDA bytes. This excludes startup/model loading
and does not establish general review reliability. Actual ROOM1020 camera0
comparison is running as session1148, log `build/room3d-review-gpu-real.log`.
No production integration or quality equivalence is claimed.

GPU actual-pair probe completed exit0: 1.46s generation; peak CUDA 3221821440 bytes. Result: `build/room3d-review-gpu-probe/room1020-camera0.json`. CPU same-pair generation took172.07s; single-pair timings exclude load and are not a throughput benchmark. Output still needs visual validation before integration.

Important: actual-pair GPU response was NO_CLEAR_DIFFERENCE, despite CPU
reporting multiple differences and different input pixels. This is a potential
false negative, so the quantized reviewer must NOT replace the production
CPU reviewer based on speed. Inspect saved comparison visually and expand
controls before deciding whether NF4 is usable.

### ROOM1030 and GPU reviewer rejection (2026-10-11)

ROOM1030 exported and passed `package_room3d.py` staging/format checks in
`build/room3d-package-room103-probe`; this is not an in-game acceptance.
ROOM1020 four CPU reviews completed and ROOM1021 reused all four exact image
pairs. CPU review queue now started for ROOM1030, log
`build/room3d-room103-reviews.log`. Main full-generation process remains live.

Inspected GPU comparison sheet: reconstructed ceiling is blue, and major
geometry/texture differences are plainly visible. NF4 reviewer incorrectly
returned NO_CLEAR_DIFFERENCE. Preserving visual and lm_head in BF16 also
returned the same false negative (0.94s generation, 3,800,043,520 CUDA bytes).
Probe `build/probe_room3d_review_gpu_visual.py`, output
`build/room3d-review-gpu-probe/visual-bf16-room1020-camera0.json`.
Neither GPU configuration is accepted for production; keep current CPU
reviewer. These controls reject these configurations, not all GPU inference.

### Full batch detail triage (2026-10-11)

Eight entries exported. `build/room3d-full-batch-detail-audit.json` records
per-category accept/reject counts and failed gates, with source hashes.
ROOM1030 and1031 each reject11 proposed details (8lamps,1railing,2stairs).
All11 fail support evidence; stairs also fail plausible proportions, railing
also fails improvement. All observations are single-view and no heldout
cameras exist. Creative fallback is not attempted for these unsupported
single-view hypotheses (`matches`, two views, or structural_prior required).
This identifies placement/association evidence as a bottleneck; relaxing
acceptance alone would not establish correct placement. No thresholds were
changed. Exported geometry remains available for retouching.

### Single-view wall fixture candidate verification (2026-10-11)

Opt-in `single_view_wall_fixtures` implemented; 38 shape tests and Ruff pass.
Rule requires creative mode, verified wall attachment, >=16 visible pixels,
>=0.6 silhouette IoU, and existing improvement/consistency/depth gates.
Post-hoc audit identifies one candidate each in ROOM101 and ROOM103 plus
scenario duplicates. Original and visible-geometry overlays were inspected:
both align with actual lamps. They remain estimates, not heldout evidence.
Visual artifact: `build/room3d-single-view-fixture-visual/index.html`.
Integration probe now refits those selected candidates against complete scenes
and bakes textures using full semantic masks and DA3 depths, without changing
source exports. Script `build/probe_single_view_fixture_fit.py`; session65318;
log `build/room3d-single-view-fixture-fit.log`; output
`build/room3d-single-view-fixture-fit`. Still pending; do not enable globally
until results are inspected. Full batch driver695424 remains active.

Single-view integration probe ROOM1010 finished and all acceptance gates
passed. Inspected camera3 candidate against original and previous export: the
new lamp is positioned correctly but appears dark, whereas the source fixture
is luminous. Full room still has major pre-existing architectural gaps. Thus
geometry acceptance is not visual-quality acceptance; keep flag opt-in.
Textured comparison: `build/room3d-single-view-fixture-fit/room1010/index.html`.
ROOM1030 probe remains in the same live session65318; do not restart it.

### Fixture appearance probe (2026-10-11)

The room-wide lamp patch came from camera1, unrelated to the candidate in
camera3. Restricting material_patches to each fixture observation yields no
patch: both masks are too small for the existing square-interior crop rule.
Mask-only median RGB is available (ROOM101:139,197,222; ROOM103:164,189,164).
Prototype `build/probe_fixture_tint.py` assigns the ROOM101 observed median to
the newly added group only, leaving geometry and existing groups unchanged.
Inspected `build/room3d-single-view-fixture-fit/room1010/camera3-object-tint.png`:
fixture is visibly brighter and no longer the dark patch of the shared material.
This is uniform baked appearance, not recovered albedo or dynamic emission.
Production integration and ROOM103 validation remain pending.

### Object-local fixture appearance integration (2026-10-11)

Implemented opt-in local_fixture_appearance through production bake,
excluded-camera bake and retouch reprojection. Records per-object exact mask
sources, detector score and SAM confidence; uniform median baked color only.
Two isolation/exclusion tests,45batchtests,3retouchtests and Ruff pass.
Actual integrated rebake finished both ROOM1010 and ROOM1030; visually inspected
results in `build/room3d-single-view-fixture-fit/index.html`. ROOM101 dark fixture
becomes light blue; ROOM103 white fixture becomes pale green, matching its own
source tint more closely but still approximate. No architecture fix claimed.
Correction to prior diagnosis: detector scores .3983/.3607 are below generic
material_patches .4 threshold, independently preventing those samples; earlier
claim that mask size alone caused missing patches was not established. New
path is only for geometrically accepted interpreted fixtures, with SAM>=.7
and>=16pixels; detector uncertainty remains explicitly recorded.
Both flags stay opt-in; full batch remains on existing recipes.

### Further-room fixture guard audit (2026-10-11)

ROOM1040 detail generation is available while full export comparisons continue.
The unchanged single-view fixture rule accepts no additional proposals: rejected
lamp IoUs are .3171,.2449,.5835. Existing accepted chandelier is a different
prior-supported path (IoU .3925); no claim of new-rule positive generalization.
Staircase additionally fails depth safety/proportions. Audit with source hashes:
`build/room104-detail-evaluation.json`. No threshold changed for this case.
ROOM1031 passes packaging in `build/room3d-package-room1031-probe`.
Full driver695424/batch695450 are still live; no restart requested or performed.

### Bake projection reuse and catalog recovery (2026-10-11)

Catalog watcher80189 was authoritatively terminal(exit143), while batch driver
695424 remained live. Restarted only catalog, session50980; now10exports,
2in progress(ROOM105 variants),308pending plus28empty. ROOM1040/1041 pass
packaging in `build/room3d-package-room104-probe`.

Profiled ROOM1040 excluded-camera bake: 15,621 camera_basis calls; each camera
rotation was needlessly recalculated per triangle. Reuse a per-bake list of
rotations in project() without changing operations/order on point arrays.
45batchtests and Ruff pass. All46 OBJ/MTL/PNG/cameras/report artifacts match
byte-for-byte between old/new bakes. Evidence:
`build/room104-bake-cache-verification.json`; profile scripts/stats under
`build/room104-bake-profile` and `build/room104-bake-cached`. Observed profile
times28.07s/10.65s are under concurrent changing load, not a reliable speedup
percentage. Existing running workers retain previously loaded code; no restart
to apply this optimization to in-flight rooms. Depth-cache optimization remains
unimplemented; this change only reuses camera bases.

### Depth cache camera-exclusion transition (2026-10-11)

Added per-room BakeDepthCache keyed by mesh bytes, visibility sidecar bytes,
camera values and requested group set. Arrays are read-only; geometry changes
clear prior views. Calibration/color/segmentation remain outside the cache.
One invalidation test plus45batchtests and Ruff pass. Prior46artifact bytewise
comparison passed with0warm depth render calls. New real transition test warms
cache while excludingcamera0, then excludescamera1 and compares against a
fresh independent camera1 bake. Evidence:
`build/room104-cache-camera-switch/verification.json`; script
`build/verify_room104_cache_switch.py`; completed session51102 exit0.
At inspection batch695450, catalog715501 and ROOM1030 reviewer706220 live;
ROOM105 variants generating new artifacts. No batch restart.

### ROOM105 lamp observation (2026-10-11)

While ROOM1050 export comparisons are ongoing, object0 lamp passed existing
all gates with maskIoU .8134 in camera0. Inspected original/assisted camera0:
lamp is visible at expected left-wall position. There is only one observation,
no heldout view; cannot establish independent cross-camera consistency.
Table is still a gray proxy and upper architecture has gaps/distortion.
Artifact `build/room105-lamp-evaluation.json` records scope and source hash.
This is the existing running pipeline, not the new optional single-view rule.

### ROOM105 exports completed (2026-10-11)

Both ROOM105 variants exported; full catalog now12exported,2in progress
(ROOM106 variants),306pending nonempty entries,28empty. Both passed
package_room3d staging/format checks in `build/room3d-package-room105-probe`.
Their 7 ordered review input pairs are identical; reuse only
exact hashes once first-variant reports exist. ROOM1030 reviewer session37986
was polled and remained live; no replacement reviewer was started.

### Review queue advanced through ROOM103 (2026-10-11)

ROOM1030 seven reviews completed, session37986 exit0. Started queue for
ROOM1031,1040,1041,1050,1051: session59543, log
`build/room3d-room1031-105-reviews.log`. ROOM1031 reused all seven exact
model/prompt/image-pair matches, checked status records. Other queued rooms
remain pending. These model reports are discrepancy proposals, not quality
approval. Main batch695450 and catalog715501 remain active; ROOM106 variants
are still fitting details.

### Stair constraints and optimization handoff (2026-10-11)

Added opt-in `constrain_stair_proportions` during candidate scoring, using
existing stair_proportions limits and the same invalid loss as support/contact
violations. A synthetic render regression proves an exact silhouette with
oversized steps loses to a plausible alternative when enabled; targeted test
passed. Full shape suite session74987 remains running at this entry. Ruff passes.
The flag remains off in the general recipes pending real-room evaluation.
ROOM1061 isolated probe uses the exact before-details recipe, cached validated
refinements/depth and camera1-original-mask00 observation; session55780,
script `build/probe_room106_stair_constraint.py`, output
`build/room106-stair-constraint-probe`. No fidelity result claimed yet.

Both ROOM106 exports completed. Verified all14 completed checkpoint inventories
and original RDT hashes, then deliberately stopped generation group695424
(session85941 terminal143) to activate previously verified camera-basis and
per-room depth caches. Continuation session16779 preserves14 exports and
processes334 remaining entries, including28 empty slots. Checkpoint bytes and
original provenance keys were verified unchanged against
`build/room3d-optimization-handoff.json`. Partial ROOM107 work may repeat;
completed exports are preserved. Catalog/reviewer processes were left running.

Full shape regression completed: session74987 exit0, 39 tests in132.317s;
Ruff passed. ROOM106 probe session55780 was polled for50s and remains live;
no final object result yet. General continuation remains independently active.

### ROOM106 constraint result (2026-10-11)

Probe session55780 completed exit0. Both previous and constrained candidates
rejected. Constrained candidate passes proportions (212.85rise/254.47tread,
39.91degrees) and all-camera depth gate, which previous candidate failed.
MaskIoU falls from .36874 to .34684, below .35 support threshold. Creative
fallback also rejects the unsupported upper endpoint. No acceptance threshold
was lowered. Evidence `build/room106-stair-constraint-probe/comparison.json`
includes report hashes and both sets of gates; inspected full-scene comparison
`provisional-comparison.png` still shows architecture occluding the stair.
The restriction prevents implausible fitting but does not establish room
fidelity or generalized reconstruction success. Next work must jointly address
architecture/upper landing support; stair treads alone are insufficient.

### Connected upper-edge proposals (2026-10-11)

Added reusable corroborated_stair_assemblies generator: upper edge position,
orientation and height constrain the stair/landing joint, not only stair rise.
Requires at least3 contributing cameras, excludes reserved cameras, suppresses
reversed duplicate edges while retaining crossing edges, bounds2 edges/32
proposals, retains estimated-support qualification. Nine assembly tests and
Ruff pass. Not integrated into automatic acceptance.

ROOM1061 probe `build/probe_room106_connected_assemblies.py` session59987
completed exit0.16 assemblies evaluated against original scene and previously
training-proposed shorter occluders,32 trials total. Best shortened-scene
training silhouetteIoU .36951 vs original-scene .1584, but max all-camera depth
regression .03037 exceeds .01. No trial passes both IoU>=.35 and maxdelta<=.01.
Inspected shortened-scene camera1: architectural proxies still dominate and
hide stair detail. This is negative evidence, not a quality improvement.
Results and comparison: `build/room106-connected-assemblies/{results.json,index.html}`.
No source collision records or completed exports changed. Next architecture
work needs to distinguish colonnade/open regions from solid extrusions, rather
than treating an added landing as proof of structural support.

### Architectural prompt ablation (2026-10-11)

Current default detector has no column/arch categories. Isolated local probe
session76233 completed exit0 with stone column,pillar,archway,balcony,wall,
staircase,railing on ROOM106 cameras0,1,6. Camera1 proposes2 columns and2 arches;
camera6 proposes1 column and4 arches. Camera1 mask sheet inspected: columns and
arch-shaped regions correspond to visible structures. Detector confidences
roughly .28-.36 remain low; SAM confidence is not geometry/identity truth.
No shape reconstruction or wall removal was performed from these masks.

Added optional --labels to room3d_vision with prompt_labels in each record and
pre-model validation of empty/nonstring/separator-containing labels. Default
categories unchanged. Repeated three-image probe through production CLI,
session45714 exit0: detections and image hashes identical to prior probe, prompt
metadata recorded; `build/room106-architecture-detection/api-verification.json`.
Ruff and invalid-input checks pass. Full probe HTML in same directory index.html.
SAM outputs remain bound to their original detection file hashes; API repeat
is separate and has not silently replaced those inputs. Future work must
associate architecture proposals across views and check geometric evidence
before altering visual walls. General generation remains independent.

### Column correspondence pilot (2026-10-11)

Completed remaining ROOM106 views2,3,4,5,7, local detector/SAM session49172
exit0; probe now covers all8 cameras. Added column_foot_correspondences:
floor-plane base hypothesis, reciprocal3px projection check, reserved-camera
exclusion, truncated-mask rejection, ambiguity and failed-pair diagnostics.
Three synthetic tests pass for same/different columns, ambiguity, excluded view
and truncation; Ruff passes. This module does not edit meshes.

Real data yields9 usable bases and0 accepted pairs. Closest pair is camera1
mask05/camera2 mask02 with errors9.95/4.60px. Inspected camera2 segmentation;
these masks do not yet establish matching physical foot points. No threshold
was relaxed. Failure can reflect different pillars or incomplete bases/assumed
floor; cannot distinguish from current evidence. Updated full8-view HTML and
column-correspondences.json under build/room106-architecture-detection.
This is an association diagnostic, not completed shared-geometry recovery.
General catalog reached15 exports during inspection; no finished room changed.

### Vertical-axis column association (2026-10-11)

Added column_axis_hypotheses as an alternative to matching visible foot pixels.
Middle mask rows estimate vertical interpretation planes, whose intersection
supplies XZ independently of foot location. Axis agreement alone is explicitly
insufficient: gates require front-facing projection, <=3px centerline error,
compatible inferred heights and floor extent. Excluded views cannot contribute;
ambiguous alternatives are reported. Five architecture tests and Ruff pass,
including partial foot visibility, different columns, incompatible heights,
ambiguity and reserved camera exclusion.

Real full8-view ROOM106 pilot:1 of34 pairs passes preliminary gates, camera1
mask05/camera2 mask02. Axis errors .723/1.779px, inferred heights4022/3955units
(relative difference .01691), XZ30343/10409. This is provisional, not identity
or recovered-shape approval. Existing foot-pixel method rejected the same pair
at9.95/4.60px; surface-foot pixels need not represent the shared axis foot.
The cause is consistent with foot geometry/mask coverage but not independently
proven. Updated column-axis-hypotheses.json and HTML under architecture probe.
Next: fit one shared column shape to both masks, then seek independent-view
support before changing architectural proxies. No completed exports modified.

### Shared column shape pilot (2026-10-11)

Added procedural column family: twelve-sided shaft represented by six closed
quadrilateral sectors plus square plinth/capital. Validates positive shaft and
part dimensions; generator regression passes. Added fit_column_axis: one shared
axis/shape, 36 bounded dimensional candidates, training selection by worst-view
silhouette/edge loss, no observed-depth selection and no automatic room edit.
Six architecture tests pass, including rendered two-view synthetic recovery
and rejecting ambiguous axes; Ruff passes after import formatting. Full shape
suite launched session32304, not yet claimed complete at this entry.

ROOM106 shared-column probe session7264 completed exit0. Selected height3988.65,
shaftwidth791.12,baseheight478.64,capitalheight199.43 at prior shared axis.
Isolated IoU .74011/.75711 in cameras1/2; inserted-scene scores .74011/.75513,
visible293/1347pixels. Max global observed-depth delta over8 cameras is0.
Both overlay images inspected: projected column corresponds to visible pillar.
These two views trained the shape, so this is not independent-view validation
or room-wide fidelity. Existing architecture does not materially occlude this
particular column; do not generalize stair occlusion diagnosis to all props.
Artifacts `build/room106-shared-column/{index.html,fit.json,scene-checks.json}`.
No finished room, collision record or main recipe was modified. Further
independent visibility audit session14134 examines all cameras without changing
fit; results other-view-visibility.json.

Independent visibility audit14134 completed exit0. Cameras0/7 have projected
pixels246/443, but no matching column mask (camera7 bestdetected-column IoU
.04708); camera3 projection is completely occluded by scene geometry. Rendered
and inspected unused-view overlays0/7. Camera7 projection falls over visible
upper railing/carpet rather than a verified matching ground-floor column.
Therefore the two-view candidate is rejected for room adoption. This contradicts
any broad claim of shared identity or correct cross-camera placement; apparent
training agreement alone is insufficient. Added independent_visual_audit to
fit.json and unused-view rejection panel to HTML. Next investigation must
resolve correspondence/vertical placement across additional views before
architectural replacement. No acceptance threshold relaxed.

Full shape suite32304 completed exit0:40 tests in58.568s. Architecture suite
six tests completed successfully separately; Ruff passes for all four source/
test files changed in this step. Goal remains incomplete; positive training
scores were not substituted for unused-view quality requirements.

### Correction: unused-view overlay is not an occlusion-aware rejection (2026-10-12)

Investigated projection/parser first: survey reads RDT_Camera fields at offsets
matching existing Types.h; camera_basis follows existing Room3D.cpp basis.
This code comparison is not independent runtime camera validation. No basis
change was made without evidence.

Crucial new evidence: of443 projected camera7 pixels,105 have reliable observed
depth and all105 indicate closer foreground (median observed15107 vs column
21479). Remaining338 unknown. Camera0 all246 pixels unknown; camera3 has255
observed-foreground and304 inferred-scene-occluded pixels. Zero model-depth
conflicts in any camera. Training views have48/1091 matching-mask depth-agreement
pixels. Thus earlier camera7 overlay rejection was unsupported: overlay ignored
occlusion. Corrected fit.json audit (prior entry retained in audit_history) and
HTML; no new identity/heldout approval claimed and no scene adopted.

Added projected_object_evidence with mutually exclusive per-pixel categories
and shape validation. Observed foreground, inferred scene occlusion and unknown
data are distinguished from closer-candidate depth conflict; matching-mask
agreement is separately counted. Regressions cover foreground occlusion,
nonfinite/unknown data, depth conflict vs positive evidence, inferred-scene
occlusion vs reliable matching observations. Pilot script
build/audit_room106_column_evidence.py session30046 exit0; detailed artifact
build/room106-shared-column/other-view-evidence.json includes selected OBJ hash.
Next work may retain this as an estimated two-view shape, but must not call
unused views either independent identity support or contradiction on current
evidence. No completed exports or source collision data changed.

### Frozen controls outside ROOM106 (2026-10-12)

Added tools/probe_room3d_architecture.py: local frozen-category detector/SAM,
axis proposals, shared column fit and occlusion-aware per-view evidence; new
output only, verifies full original checkpoint inventories before/after,
records all tool source hashes at start and detects source changes at end.
No automatic room edits. Two integrity tests and Ruff passed. User-facing CLI
usage documented in ROOM3D_BATCH.md.

Initial attempt44741 was deliberately terminated143 to correct logger and
initial provenance recording before validation; partial outputs preserved in
build/room3d-architecture-controls-interrupted-before-provenance-fix. Not an
observation-timeout restart. Corrected source passed checks before finalrun.
Final session25301 completed exit0, source_code_unchanged=true; all11 cameras
of ROOM1070/ROOM1080 processed with same category list/thresholds as ROOM106.
ROOM107:11 column-category detections,3 geometric trials,0 eligible pairs.
Inspected camera0: model labels statue+pedestal and pedestal alone as column.
ROOM108:3 detections,0 geometric trials,0 eligible pairs. Inspected camera0
column mask is wallpaper reaching top edge, excluded by truncation check.
No shared columns generated in either control. This supports false-proposal
rejection for these controls only, not positive cross-room shape generalization.
Artifacts build/room3d-architecture-controls/{manifest.json,evaluation.json,index.html}.

Previous review queue59543 confirmed exit0, all queued rooms1031–1051 completed.
Started next queue session49649, logbuild/room3d-room106-109-reviews.log for
ROOM106 variants through ROOM1090; reviewer live750097 at inspection. Main
continuation16779 still active; catalog22exported,2in progress,296pending,
28empty. Operational generation progress is not full goal completion.

### Frozen fixture rule trial in ROOM109 (2026-10-12)

Returned to goal's lamp/stair categories. ROOM109 camera3 contains a visible
wall bulb; three rejected anchored lamp candidates pass other gates but lack
multi-view support, IoU .67143/.60328/.60849. Launched isolated two-mode trial
build/probe_room109_frozen_fixtures.py session44955 (live754217 at lastcheck),
outbuild/room109-frozen-fixture-evaluation. Uses exact validated source
checkpoint, before-details recipe, cached masks/depth, fixed selected lamp
observations. Only geometry option changed is single_view_wall_fixtures false/
true. Shared appearance vs object-local color evaluated separately on the same
variant shape. Records code hashes and revalidates source at end.
Control finished fitting/baking: all3 rejected, IoUs reproduce original reports
exactly. Variant remains in progress; no improvement claimed yet. Source tools
must remain unchanged during this frozen run; doc/build-script writes excluded.

All24 completed entries ROOM100 variants through ROOM10B variants passed
package_room3d format/material/texture limits. Session90883 exit0; staged in
build/room3d-package-through-10b-probe, package.json contains24 packaged results.
This is staging/format validation only, not gameplay or fidelity validation.
No live game installation and no original exports modified.

### Fixture appearance separation and stair constraint results (2026-10-12)

Frozen ROOM109 fixture trial finished with unchanged tool hashes and original
export. Control rejected all three selected lamps; optional single-view rule
accepted all three. Their silhouette scores were unchanged (.67143/.60328/.60849).
Visual crops exposed a separate regression: whole-mask median tint darkened the
bright bulbs. Acceptance gates did not prove appearance quality.

Optional local appearance now retains observed contrast with separate support
and shade materials for parametric wall sconces/table lamps. Sampling remains
restricted to exact object masks, with reserved-camera samples excluded. Dark
quartile and brightest 15% medians are assigned only with luminance separation
>=.15; flat masks keep their uniform tint. This is procedural assignment of baked
colors, not recovered albedo/emission. Other profiles retain uniform tint.
Default recipes and primitive group identities are unchanged.

Appearance-only pilot build/room109-component-palette/index.html uses the same
accepted shape fit, regenerating material tags only. Crops confirm all three
bulbs are bright again with dark supports. They also reveal incorrect shade
placement/silhouette, especially cameras4/7; appearance correction does not solve
geometry. Source export and code hashes validated unchanged during the pilot.
Four palette/bake tests pass, including object isolation, flat masks, part
materials and reserved-camera exclusion; all45 existing batch tests and Ruff
pass. Option remains disabled in the full generation recipes. Further rooms
must be checked before any default change.

Frozen ROOM10B constrained-stair probe finished with unchanged source and exact
original trial baseline. Proportions became plausible (154 rise,479 tread,
17.85-degree slope), global depth remained safe, but IoU .34537 misses the .35
support gate. No staircase accepted; thresholds unchanged. Artifacts and
comparison: build/room10b-stair-constraint-probe/index.html. This still requires
better structural placement/occlusion, not merely a proportions constraint.

### Wall arm direction and anchor pilot (2026-10-12)

Added bounded arm_rise [-1,1] to the reusable wall_sconce generator. Negative
Y is up; positive rise raises the shade relative to the plate. Absent parameter
preserves the exact old formula. Plate geometry/anchor remains unchanged when
only direction changes. Synthetic invariant test verifies these properties and
rejects nonfinite/out-of-range input. Shape suite41 passed; Ruff passed.

Frozen ROOM109 exploratory search compares the brightest15% of each exact lamp
mask to its isolated shade render, separate from whole-object silhouette.
Changing arm direction alone improves bright-region agreement but loses whole
silhouette support. Searching vertical plate positions on the same wall also
produces candidates retaining whole IoU >=.35. Results (old -> exploratory):
primitive6 whole .6714 -> .5634, shade .1026 -> .5882;
primitive7 whole .6033 -> .5594, shade 0 -> .3846;
primitive8 whole .6085 -> .4429, shade 0 -> .2400.
This is a tradeoff, not universal shape improvement or a fidelity percentage.
All three finite-wall contact checks pass; all8 camera depth checks remain
within .005 of the previously accepted approximate scene. It does not validate
object identity, unseen parts or traversal. Visual crops still show placement
and silhouette errors. No batch default or accepted source export changed.

Artifacts: build/room109-wall-arm-probe-v4 report.json, scene-gates.json,
provenance.json; rendered comparison build/room109-wall-arm-render/index.html
and comparison-crops.png. Earlier exploratory v1 accidentally included the
floor in isolated masks and is invalid; v2 fixes primitive-only rendering and
reproduces original whole-object IoUs. V3 tests direction alone, V4 adds wall
anchor height search. Preserve these distinct experiments rather than treating
them as independent new-room validation. Next needed: source-integrated
part-aware objective with conservative bright-region qualification and tests
in rooms beyond ROOM109 before adopting automatic anchor search.

### Source-integrated part objective and new-room controls (2026-10-12)

room3d_fixture_parts now qualifies contained, dominant bright components from
exact training masks and performs a bounded arm/vertical-anchor search. It
abstains on flat, clipped or fragmented highlights. Candidates must retain
finite-wall contact, whole IoU >=.35, lose <=.15 whole IoU in every training view,
and improve each used bright-region IoU >=.05. Last view is reserved when >=3
observe the object. fit_objects receives original images and calls this stage
only for fixture_part_fit:true; the main batch passes images explicitly. Existing
final all-camera scene acceptance remains authoritative. Missing images with
the option enabled fail explicitly; normal recipes remain unchanged.

Four tests cover region containment/background isolation, contrast/fragmentation/
clipping abstention, reserved-image exclusion and invalid dimensions. Existing
45 batch and41 shape tests pass; Ruff passes. No model weights or external image
service added.

Frozen helper evaluation build/fixture-parts-generalization completed with
unchanged source hashes and original checkpoints. ROOM1010: shared lamp4
abstained; single-view lamp5 bright IoU .1889 -> .2674, whole .6000 -> .5714.
ROOM1030 lamp5 unchanged (.1681 bright IoU). ROOM1090: lamp6 .1026 -> .3333;
lamp7 abstained (clipped highlight), lamp8 unchanged. All three rooms pass depth
comparison against their previous approximate scene. These helper outputs do
not prove full-pipeline acceptance of the new-room candidate or visual fidelity.

Full source-integrated ROOM109 trial build/room109-integrated-part-fit finished:
all three lamps accepted through existing final gates; only lamp6 part fit
selected. Its final whole IoU .6351 and bright IoU .3333, wall contact and all
camera depth gates pass. Others preserve previous geometry. Rendered crops
show a modest first-lamp improvement, substantial remaining geometry errors in
cameras4/7. Source hashes/checkpoints unchanged. Default batch recipe not enabled.
Started full-pipeline isolated ROOM101 single-lamp control in
build/room101-integrated-part-fit, session recorded in execution context;
its outcome is still pending. This will test whether helper improvement survives
existing single-view acceptance rather than silently assuming it does.

ROOM101 full-pipeline control completed with unchanged code/source checkpoint:
part refinement selected the .5714 whole-IoU candidate, but final single-view
support requires .60, so the lamp was rejected despite all other gates passing.
This invalidates treating the earlier helper selection as a usable new-room
improvement. Corrected candidate eligibility to preserve the existing .60
single-view threshold when single_view_wall_fixtures is enabled; other searches
retain .35 and the per-view .15 maximum drop. No acceptance threshold relaxed.
The second ROOM101 trial in build/room101-integrated-part-fit-support-preserved
now retains the original lamp (part status unchanged), and all final gates pass.
Render/bake completed; source-code hashes and original checkpoint are unchanged.
Five part tests and Ruff pass after the correction. Current positive integrated
part-fit evidence remains ROOM109 only; ROOM101/103 are abstention controls,
not evidence of general shape improvement. Further positive new-room validation
is still required before enabling this option in the main recipes.

### Raised stair support and visual occlusion (2026-10-12)

ROOM108 full integrated lamp control completed with unchanged code/checkpoint.
Its lamp remains accepted; part fit status unchanged. This adds another abstention
control, not a positive part-fit generalization result.

Rejected stair inventory across completed distinct rooms:10 plausible-proportion
failures,9 support failures,3 global-depth failures,2 improvement failures
(categories overlap). ROOM10B constrained stair had plausible proportions and
safe global depth but whole IoU .34537. Its visibility diagnostic identifies
boundary_2:1734 nearest occluding pixels,1733 reliable-depth pixels, generated
surface5416 vs observed7651. Existing occlusion proposals are evaluated only
from the floor-support branch; the selected raised support is never searched.

Frozen fixed-geometry trial build/room10b-supported-occlusion-v2 reproduces the
original after.obj byte-for-byte after reconstructing the selected support.
Uses cached stair-refined mask, reproducing baseline .34537 exactly. Existing
1500/3000 visual height proposals both increase whole IoU to .68872 with no
stair geometry change. All4 camera depth changes stay within .005. Visual render
shows more of the staircase, while remaining room architecture is poor. Original
collision data untouched. V1 used the raw mask instead of the cached refined
mask; it is not a comparable reproduction of the original fitting scores.

Added optional raised_support_occlusion stage after choosing a non-floor stair
support, before existing headroom handling. Reuses existing depth/tread-qualified
visual-height proposals. No recursive branching, fixed fitted staircase, at most
two alternatives. Selection uses training loss/depth only; final fit_objects
scene gates remain mandatory. Two tests verify that a reserved-view depth failure
is reported rather than treated as approval, and that real observed foreground
prevents a false visibility improvement. Three existing support/occlusion tests
and Ruff pass. Default recipes remain unchanged.

Launched full source-integrated trials with constrain_stair_proportions plus
raised_support_occlusion on ROOM10B and previously untested ROOM10C staircase.
Outputs build/room10b-raised-support-integrated and
build/room10c-raised-support-integrated. Both are pending at this note; no
full-pipeline acceptance or new-room improvement claimed yet.

Both integrated stair trials finished with unchanged tool hashes, exact original
baseline mesh and unchanged source checkpoint. ROOM10B: accepted:true,
raised_support_occlusion selected:true, final whole IoU .688715 (old constrained
trial .345373), all final gates pass. Support boundary4 atY-2501, rise154.16,
tread478.78, slope17.85deg. Both height proposals eligible; final recipe stores
the selected visual geometry. No runtime traversal proof or reserved observing
stair camera (single observation).

ROOM10C: accepted:false, constrained fitting changed selected support to floor;
raised-support stage therefore did not run. Final IoU .10754, only supported gate
fails; proportions/depth now pass. This is not positive generalization. The
remaining search limitation is that correcting only the winning raised support
cannot reveal an initially worse raised-support hypothesis hidden by provisional
architecture when the floor wins. Next investigate bounded corrections on
promising raised alternatives before final support selection, retaining training
selection and all-camera final gates.

Started normal appearance bake for before/accepted ROOM10B scenes in
build/room10b-raised-support-appearance, using raw semantic masks like the main
pipeline. Its visual outcome remains pending at this note. Existing untextured
renders confirm increased stair visibility but substantial architecture errors
remain. No defaults changed, source collision records or existing exports edited.

ROOM10B appearance comparison completed, source/export provenance unchanged.
Inspected after/camera2.png: textured treads are visible after correction, but
large gray architectural patches and distorted chandelier projection remain.
This is a local staircase/occlusion improvement, not a completed room or overall
fidelity percentage. Comparison HTML is available under the appearance output.

### Support ranking and semantic target audit (2026-10-12)

Raised-support correction now considers the best two eligible raised solutions
before the final floor/raised choice, adding at most four visual-height trials.
Original candidates remain available; corrected reports are independent copies.
Selection loss includes the same tread-edge term as initial fit, avoiding an
unfair comparison with uncorrected alternatives. Five tests cover initial floor
winner replacement, bounded non-floor search, original-report preservation,
training/reserved separation, observed foreground, and consistent tread objective.
Ruff and three existing support/occlusion tests pass.

First ROOM10B ranking trial preserves accepted stair IoU .688715; its two raised
alternatives were examined. ROOM10C still rejects: no raised candidate passes
contact with constrained proportions. Visual inspection then revealed the
important correction: ROOM10C camera3 is a fountain/pedestal room, not a staircase.
The mask actually selects a narrow floor/border fragment. Do not force a plausible
stair into this false detection. Previous notes treating ROOM10C as a positive
stair generalization case are superseded; it is a useful negative control.

build/stair-category-audit.png collects14 staircase-labelled observations from
completed distinct-room exports. Several are shelves, rugs, furniture or floor
strips. Background presence of a stair is insufficient: mask audit
build/stair-mask-audit.png shows ROOM101 target selecting the banister panel,
ROOM109 target only bottom steps, and ROOM10B raw mask encompassing chandelier/
ceiling until its cached tread refinement isolates the real flight.

New full-pipeline probes use actual stair-containing ROOM101/109 plus ROOM10B
regression after objective consistency correction. ROOM109 remains rejected
(zero visible candidate pixels, invalid proportions, no support). ROOM10B remains
accepted at .688715 with all final gates passing. ROOM101 still active at this
note; no result assumed. Existing exports/checkpoints are preserved.

Read-only gameplay metadata audit reuses existing cmd_room_action_set,
stairs_height_update and door filename table implementations plus the existing
init-command decoder. build/stair-gameplay-evidence.json covers348 RDT files:
28 room variants have kai*.dor transition records; only ROOM40D0/1 expose handler
0x11 height zones before any decode stop.181 entries are partial/invalid/short
(including28 short slots), so absence is not a semantic veto. Static conditional
records are not runtime state. Transition arrival Y is destination-local, not
source stair rise. ROOM101 has kai02 transition, ROOM10B kai03; ROOM109 parsing
stops at unknown opcode. These inputs can add measured hints in supported cases,
not replace category/shape inference everywhere. No new opcode interpretation
or runtime change was made.

ROOM101 trial completed with unchanged source hashes/checkpoint and exact original
baseline. Proportions and two-view silhouette support pass (IoUs .42491/.44683),
but all-camera depth fails, so the staircase is rejected. ROOM109 and ROOM10B
provenance also verified. No new-room positive staircase result claimed. Target
part semantics (banister versus tread volume; partial bottom steps) is now the
next priority, rather than weakening the depth/support gates or forcing stairs
into false detector categories. Original full goal remains active.

### Part prompts, door-panel conflict guard and local category pilot (2026-10-12)

Fixed six-label local DINO/SAM pilot completed on six views from ROOM101 (two),
ROOM109, ROOM10B, ROOM10C and ROOM102. Cached pinned models, offline, no uploads;
original exports/code hashes preserved. build/stair-part-prompt-probe/index.html.
Visual inspection: ROOM101 new 'stair treads staircase' mask selects floor/wall
trim; ROOM102 shelf is still labelled as treads. Mixed/truncated labels and a
false fountain detection persist. Explicit part prompts alone do not solve
semantic separation; they are not adopted as defaults or normalized blindly.

A zoom of ROOM109 camera3 conclusively shows a panelled door, not the previously
hypothesized partial bottom stair. Prior notes calling ROOM109 a positive stair
case are superseded. Its existing door detection (.6536, SAM .958) contains98.64%
of the weaker staircase mask (.3913, SAM .843), providing a cross-category
abstention cue without loading another model.

Added optional semantic_stair_guard in fit_objects before observations/association.
room3d_semantics preserves inputs and other categories, removes only qualifying
contained staircase conflicts and emits explicit provenance/reasons. Four tests
cover exact door-panel conflict, weak/ambiguous alternatives, partial overlap,
railing preservation, nonmutation and pipeline reporting without fitting the
false stair. Existing45 batch tests and Ruff pass.

Frozen source/export-preserving audit on22 distinct completed room variants
ending0 removed one candidate: ROOM109 camera3. All stair candidates in ROOM101,
ROOM106 and ROOM10B remain. This proves the tested abstention only, not a complete
category classifier. build/stair-semantic-guard-evaluation/index.html/report.json.
No default full-batch recipes changed.

Started separate four-case Qwen3-VL-4B local category proposal pilot, pinned
existing cached model and fixed single-letter prompt, at
build/stair-category-review. Uses cyan mask contour with whole-frame context to
classify visible treads, railing, door panels, shelf, floor/ornament, other or
uncertain. No geometry acceptance uses its answers. Cases ROOM101 banister,
ROOM109 door panels, ROOM102 shelf and refined ROOM10B treads. Responses still
pending at this note; existing model comparison reviews continue independently.

Added nonfinite-confidence abstention to the semantic guard (NaN/Inf cannot veto
a category); unit tests cover both cases. Repeated distinct-room audit in
build/stair-semantic-guard-evaluation-v2 reproduces the same one rejection across
22 rooms with unchanged source/export provenance. Four guard tests and Ruff pass.
Local Qwen category pilot remains active, with first inference pending at this
note; no classifier accuracy or adoption claimed.

## Completed local part-category pilot and exact-input integration

The four-case pilot is now terminal. ROOM101 camera3 banister was classified B
(railing), ROOM102 camera0 shelf D (shelf): both correct on visual inspection.
ROOM109 camera3 door panels were classified A (visible treads), and ROOM10B
camera2 refined visible stair flight B (railing): both incorrect. These two
successes and two failures are a small diagnostic sample, not an accuracy
estimate. Category proposals must not route generators or approve geometry.

`room3d_part_review.py` records pinned model/revision/prompt and exact RGB/mask
hashes. Optional `stair_part_reviews` annotation in the fitter preserves original
detector labels, reports unverified proposals, skips absent reviews and rejects
stale records. Five tests cover nonmutation, cache identity, invalid mappings,
door-guard independence and CLI reuse without constructing the model, including
changed-image rejection after a valid segmentation cache update.

Real-data validation in `build/stair-part-review-integration-audit.json` loaded
all four records against the current original/refined masks without changing
the caches. The ROOM109 door guard rejects the false stair despite Qwen's A;
other three cases are retained for independent shape/depth checks. ROOM102 CLI
cache reuse was also executed with model construction forbidden and succeeded.
Its official inference completed separately; three other records reuse pilot
responses only after exact annotated-image, mask, model, revision and prompt
verification, recorded in `build/stair-part-review-import-audit.json`.

This integration adds traceable evidence, not a new geometric improvement or
proof of complete rooms. All experimental recipe options remain off in the
running full batch. Next geometric work must use independent part structure
and shared spatial constraints rather than trusting this category model.

An unchanged-threshold, raw-mask structural audit on six rooms is saved in
`build/stair-independent-edge-audit.json`, with source RGB/mask hashes. Tread
regions occur in ROOM106 camera1 (five edges, correlation .942) and ROOM10B
camera2 (eight proposals; strongest eight edges, correlation .946). None occur
in ROOM109's door, ROOM102's shelf, or ROOM101's banister masks. ROOM10C camera2
also yields a five-edge region (correlation .794) around the fountain/border
scene; visual inspection does not establish a complete stair flight there.
This is a structural false-positive/ambiguity to retain in evaluation, not a
reason to approve a staircase or tune a threshold solely to these examples.
ROOM106 camera1 is the next independent positive for spatial/part fitting.

## ROOM106 feasible initialization experiment

The frozen `build/room106-raised-support-trial` used ROOM1060 camera1, constrained
proportions and raised-support visual correction. It reproduced the previous
.346842 mask IoU, below the .35 supported threshold; no staircase was accepted.
Original exports and the source snapshot remained unchanged during that trial.

An initialization issue was isolated: the observation-derived mask seed has a
13527-unit run and 3552-unit rise, making it too shallow for the hard proportion
gate before local optimization. Added optional `feasible_stair_starts`, enabled
only with creative fallback and constrained proportions. It appends up to two
estimated alternatives without removing originals or modifying independent
level priors. Width/base pose/rise/landing are preserved; run and step count may
change. Infeasible generator step limits cause abstention. Initialization
provenance survives subsequent coordinate optimization.

`build/room106-feasible-starts-trial` achieved .474356 mask IoU and passed the
existing evidence-fit silhouette/depth/proportion gates. The explicit additional
`end-connection-audit.json` nevertheless finds bottom contact valid but upper
contact absent. This is an improved visual candidate, not a complete traversable
assembly. Existing evidence-fit gates do not prove stair-end connections.
The isolated uniform-material image obscures internal edges; an alternating
per-step diagnostic render, `diagnostic-step-colors-camera1.png`, confirms visible
regular steps. Those colors are diagnostic only, not an appearance modification.

Twelve assembly tests now include shallow/steep seed recovery, immutability,
preserved placement dimensions, independently constrained-level abstention,
generator-limit abstention and a rendered opt-in integration test. Forty-one
existing shape tests pass. The second frozen run completed after the provenance
preservation fix: `build/room106-feasible-starts-trial-v2` reproduces the exact
accepted OBJ bytes and .474356 IoU, now explicitly reported as procedural
interpretation. Source snapshot and original exports are unchanged. Its
`comparison.html` links geometry views, alternating-step diagnostics and the
explicit failed upper-contact audit; `comparison-audit.json` records these
checks. Forty-five batch regression tests and Ruff pass. Full-batch defaults
remain unchanged; upper-floor assembly/support is still required.

## Upper-floor connection and missing ROOM106 camera7 observation

Explored upper-floor slabs joining the fitted flight to a three-camera
corroborated railing/base edge. `build/room106-upper-floor-bridge` is an invalid
assembly test: its box-only `landing` primitive was not emitted by the mesh
exporter. Its scores must not be cited as evidence about added-floor geometry.
Version2 emits explicit prism parts and asserts that the landing group exists,
but its conservative opening leaves one upper-contact sample unsupported and
the inherited low ceiling patch cannot clear a taller flight. No acceptance.

Version3 constructs the floor opening from the stair footprint, restores the
original inferred roof slab from the untouched pre-detail recipe, and raises
a patch over the combined stair/floor footprint. It also checks the new landing
as a possible overhead obstruction. Six candidates cover slopes30/36/40 degrees,
with and without the fitted landing. All have geometric contact at both ends;
the variants without a landing pass headroom. None passes the all-camera depth
gate. Candidate4 reaches .549519 mask IoU, with maximum scene-depth degradation
.062073 in camera7. These are rejected procedural assemblies, not completed
rooms or gameplay traversal proof. Diagnostic HTML and all rejected trials are
under `build/room106-upper-floor-bridge-v3`.

Inspection of camera7 reveals a second clear view of the same stair flight,
missing from detector observations: only a weak wooden-floor mask was cached.
Unchanged structural edge extraction finds27 edges with .912 correlation inside
that mask. The resulting region overlaps visible stairs and adjoining landing;
it is not yet a verified staircase segmentation. Full-image extraction instead
selects mostly the downstairs carpet/landing, demonstrating why unbounded
structural proposals cannot automatically relabel objects. This missing view
must be recovered/checked before joint fitting can claim multiview consistency.
A separate fixed-prompt local DINO/SAM diagnostic completed for camera7;
original masks, recipes and exports remain untouched. It found carpeted
staircase (.322 detector, .676 SAM) and railing (.3005 detector, .833 SAM),
where the original broad prompt only found wooden floor. The stair mask also
includes adjoining floor, so it is not promoted to verified segmentation.

`build/room106-camera7-stair-refinement` tests the existing structural/SAM
refiner with an explicit diagnostic alias from carpeted staircase to staircase,
without rewriting the original cache. Its selected proposal has SAM .506,
25038 pixels and repeated-edge correlation .987; floor contamination remains.
The existing world-bound associator automatically groups that observation with
ROOM106 camera1 (recorded in association-audit.json). A frozen two-view fitting
trial is now running under `build/room106-two-view-stair-trial`, using both
proposed masks and preserving all original exports. No acceptance or shared
geometry improvement is claimed until that trial and visual checks complete.

## Two-view stair fit and suspended underside hypothesis

The two-view trial completed with a single associated flight and IoUs .458242
and .389404 (cameras1/7). It passed silhouette/proportion checks but failed
all-camera depth: approximately+.050 error in cameras3/4. Camera7 improved by
.03645. Frozen source/export provenance was verified. The diagnostic HTML is
`build/room106-two-view-stair-trial/comparison.html`; its candidate is rejected.
Mutual SIFT plus calibrated triangulation found zero valid matches between
these images, so no point-correspondence support is claimed.

`build/room106-two-view-depth-attribution` separates stairs and visual-wall
corrections. Walls alone leave camera3/4 errors unchanged; the solid flight
alone produces the degradation. The source generator fills every step down
to its base plane, adding a large inferred volume underneath the flight.
This motivates a suspended structural alternative rather than loosening depth
gates or discarding the other camera evidence.

Added `structure:suspended` and bounded `slab_depth` to the generic staircase
generator, preserving exact default solid geometry and all tread top surfaces.
Adjacent closed step prisms overlap vertically; the landing becomes a thin
slab. Hidden underside and load-bearing structure remain estimated. Fixed-pose
trials at depths60/100/160 under `build/room106-suspended-stair-probe` reproduce
the previous solid OBJ exactly before changing only that structural parameter.
All three pass the original all-camera depth threshold; maximum degradation
is .0031/.0035/.0042. Camera1/7 mask IoUs remain above .35, around .38/.36.
Upper-end support remains missing, and no gameplay traversal is established.

Optional `suspended_stair_starts` adds bounded automatic structural hypotheses
alongside the solid observation-derived seeds; no default batch recipes change.
Provenance survives optimization. A full source-integrated two-view run with
this flag initially failed because coordinate search treated the categorical
structure string as a numeric parameter. Structure now remains fixed within
each topology hypothesis, like lamp profile; the rendered opt-in test executes
an optimization round to cover this. The corrected full run at
`build/room106-two-view-suspended-stair-trial-v2` completed; its rejected outcome
and subsequent budget correction are recorded below.
Forty-four shape tests cover rendered opt-in fitting,
top-surface preservation, adjacent-step continuity and invalid slab rejection.

The refiner now supports explicit carpeted/wooden/stone staircase phrases
without category aliases. The actual camera7 inference with native label
reproduces the exact previous mask bytes in
`build/room106-camera7-stair-refinement-native-label`; a source-label mismatch
now invalidates new refinement records. Ambiguous combined labels stay excluded.
One dispatch/cache integration test,45 batch regressions,5 part-review tests and
Ruff pass. Old cache records without source_label remain compatible.

## Structure transfer and optimization-budget correction

`build/room10b-suspended-transfer` applies the same suspended underside to the
previously accepted ROOM10B staircase at fixed pose/step positions. Its baseline
OBJ exactly matches the prior accepted solid artifact. Thickness60/100/160 gives
IoUs .6775/.6804/.6843 (solid baseline .6887); all pass all-camera depth regression,
with maximum degradation around .0001. No upper-support or runtime claim follows.
This is a transfer check on another room, not an unseen-room reconstruction
benchmark or new fit. Source snapshots stayed unchanged during the experiment.

The full ROOM106 suspended search version2 completed but selected solid starts,
failing both depth and minimum-view support. Inspection revealed that the four
optimization slots were shared by solid and suspended hypotheses, so initially
better solid mask fits could prevent all thin seeds from being optimized.
The search now budgets four starts independently per structure and includes
structure in family identity. Original solid defaults retain the same budget.
Optional structure_search reports which hypotheses were actually optimized.

A rendered test deliberately supplies five solid hypotheses initially better
than the thin one; four solid and one suspended seed are nevertheless optimized,
and the solid target still correctly selects solid. Explicit candidates with
the same initialization name but different structures are both retained. Forty-
five shape tests, the focused explicit-family test,45 batch regressions and Ruff
pass. The frozen ROOM106 run with corrected family allocation completed at
`build/room106-two-view-suspended-stair-trial-v3`; results follow below.

## Whole-scene depth during fitting (experimental)

The corrected ROOM106 version3 selects a suspended staircase and reaches
silhouette IoUs .4261/.4187 in the two observed views. Proportions, support and
consistency pass, but final depth regression rejects it: camera4 degrades .006505
against the unchanged .005 tolerance. Camera7 improves .035862. Provenance
confirms original exports and tool sources stayed unchanged during that run.

`build/room106-slab-depth-diagnostic` holds this selected pose fixed and tests
thickness60/100/160. All still fail the global gate (maximum degradation about
.0052/.0057/.0064). Thickness alone cannot explain or fix the mismatch. The
upper-floor bridge prototypes version4 and version5 are also rejected; none is
installed or adopted as a reconstruction rule. Level/contact plausibility alone
does not establish consistency with the observed room.

Optional `scene_depth_fit` now adds a penalty while searching geometry, using
all room cameras except the object observation reserved for evaluation. It
renders the candidate in isolation, merges its depth with the immutable scene
using the nearest surface, and compares each view to that scene's baseline depth
error. The maximum positive regression above .005 adds20 times that excess to
the object loss. Training cameras without an object mask can therefore constrain
intrusions. Reserved-view scores contain no such training penalty. The separate
final all-camera acceptance gate remains unchanged. This is guidance from model
estimated depth, not a guarantee or measured geometry. Default recipes keep the
feature disabled. Forty-six shape tests and45 batch tests pass. The new rendered
case supplies ambiguous frontal silhouettes plus reliable floor evidence from
an unobserved lateral camera: the optional penalty switches selection from
solid to suspended. Multiplying reserved-camera depth by100 leaves all training
scores and the selected geometry unchanged while changing heldout evaluation.
The frozen ROOM106 trial at `build/room106-scene-depth-fit-trial` completed
with source/export provenance intact. Depth safety now passes in all eight
cameras (maximum degradation .002001), but the selected solid candidate has
IoUs .56044/.27610: final support rejects it. The depth constraint alone does
not guarantee enough agreement with both observations. `comparison.html` shows
the rejected proposal explicitly. A warm-start refinement diagnostic is testing
whether retaining the previous automatically fitted thin shape before depth
refinement avoids this objective tradeoff; no full-pipeline success is claimed.


## Independent ROOM207 descending-stair failure

`build/room207-scene-depth-fit-transfer` runs the same optional settings on a
room not used to tune the straight-flight search. All cached exact staircase
labels are supplied, with detector/mask thresholds unchanged. The exported
checkpoint and source hashes stay intact. The result is rejected before fitting:
all four available support hypotheses report `insufficient_extent`.

Inspection of camera0 and camera4 originals shows a stair with a turn/landing.
Camera4's SAM prediction .79516 falls below the unchanged .8 observation gate;
the duplicate camera0 masks merge into one observation. No heldout view exists.
`initialization-audit.json` records why there is no floor-anchored seed despite
a raw depth seed being available: the lowest observed depth band estimates
Y4144, while camera0 is atY-54. Projecting that band onto assumed floorY0 needs
scale .01286, below the permitted .25..4; the three higher collision-box supports
need negative scale. These are depth proposals, not measurements of the floor.

Consequently the new whole-scene penalty cannot help this room yet: it only
operates after candidate initialization. Descending stairs, uncertain floor
levels and multi-flight topology need explicit reusable hypotheses and floor
opening validation. A confidence-threshold reduction or a claimed successful
transfer would conceal this failure. The existing export remains unchanged.


## Staged whole-scene depth polish

`build/room106-depth-polish-probe` takes the previous automatic suspended fit
as a warm start and reconstructs its fixed scene including the selected ceiling
patch. The baseline OBJ matches the prior full trial exactly. Five balanced
refinement rounds with the scene-depth objective raise two-view silhouette IoUs
to .52466/.48439. All eight depth deltas pass .005 (maximum .003551), and procedural
proportions pass (22 steps, slope30.85 degrees). This fixed-scene diagnostic does
not establish full-pipeline acceptance, upper-end support or runtime traversal.

Optional `scene_depth_polish` now schedules a similar refinement inside
`fit_objects`: the original mask search selects geometry/support/enclosure first;
then a single warm start is optimized with whole-scene depth and balanced views.
The initial search disables direct `scene_depth_fit` when staged polishing is
requested. The polish enables depth fitting independently, never recurses and
never adds new topology hypotheses. The initial fit remains nested in
`depth_polish.initial_report`, with its initial shape preserved. Final diagnostics
and all-camera acceptance run on the polished mesh. No gates are relaxed, and
all defaults remain unchanged.

A rendered end-to-end test uses a generated suspended staircase observed by three
cameras plus a fourth unobserved depth view. It verifies two training cameras,
reserved-camera exclusion from polishing, successful final acceptance and exact
agreement of the exported mesh's recomputed depth errors with the final report.
The frozen original-input trial completed at
`build/room106-staged-depth-polish-trial`: accepted as a procedural interpretation
with all final gates passing. IoUs .52466/.48439 exactly reproduce the diagnostic;
all-camera depth regression peaks at .003551 and camera7 improves .049388.
Recomputed headroom is NOT clear: polishing widens the flight beyond the prior
ceiling patch. Headroom is currently diagnostic rather than a final acceptance
gate, so accepted here does not mean traversable. Source snapshots and original
exports are unchanged. A headroom constraint must be included during polishing
before this can qualify as a usable stair assembly.
The comparison is in `comparison.html`. Only two object observations exist, so
there is no heldout object view here. The upper-floor connection and runtime
traversal remain unverified; these scores are not fidelity percentages.


The independent local-normal diagnostic
`build/room207-local-stair-plane-probe-v2/index.html` smooths estimated depth,
computes calibrated world normals, groups connected inclined regions by
direction and fits each local flight independently. It produces three plausible
partial seeds in camera0, including a four-step, width1676/run1439/rise785
candidate. Camera4 yields no seed passing these scale/part checks. This avoids
the shallow global regression across a turned staircase, but does not recover
the long flight or the turn. Estimated bases nearY3900 remain below the assumed
floor; isolated renders exclude that floor explicitly. No floor opening or
support has been validated, and the prototype is not integrated into batch code.


## Clearance-constrained polishing correction

Both the first staged ROOM106 trial and the ROOM10B transfer demonstrate that
passing visual gates can still leave insufficient clearance. ROOM106 widens
beyond the earlier ceiling patch:9 samples are blocked, minimum clearance860.
ROOM10B also accepts visually but has21 blocked samples. These historical
artifacts remain marked as visual-only acceptance, not usable assemblies.

`build/room106-headroom-depth-polish-probe` tests a known-clearance guard on the
same automatic warm start and fixed scene. It reaches IoUs .45494/.40994 with
all-camera depth regression below .005 (maximum .004153), zero obstructed samples
and known clear headroom. It remains a fixed-scene diagnostic.

Polishing now enables `constrain_stair_headroom`: candidates are sampled against
immutable inferred ceiling surfaces during search, and known low-clearance
candidates cannot beat a valid fit by their silhouette alone. The final staged
acceptance and creative reinterpretation path also check known ceiling
obstructions. Missing overhead evidence remains explicitly unknown, rather than
being treated as proof of usable clearance. Default unstaged behavior is
unchanged. A rendered regression supplies a wider candidate matching the mask
better but extending under a low side ceiling: the guarded search instead picks
the narrower clear candidate. End-to-end coverage checks the final clearance gate
and preserves unknown-ceiling diagnostics when no overhead geometry exists.


The clearance-corrected ROOM10B full transfer completed at
`build/room10b-staged-depth-polish-transfer-v2`: all final visual/depth/proportion
and known-headroom gates pass, IoU .59567, maximum depth regression about .00001,
minimum sampled clearance1800, zero unknown or obstructed overhead samples.
The source export and source snapshots remain unchanged. A textured bake at
`build/room10b-staged-depth-polish-textured-v2` passes runtime package limits:
9324 vertices,3108 triangles,37 textures,39 materials, OBJ about947KiB.
The isolated Wine probe `build/room10b-staged-depth-polish-runtime-v2/result.json`
passes mesh loading and a camera transition in46.57s. It does not validate visual
fidelity, end support, new geometry collision, item transfer or saving.

Appearance inspection still shows chandelier pixels projected onto room
surfaces. In camera2, fixture detections .3781/.3501 have SAM confidence
.816/.865 but fall below the appearance veto's .4 detector threshold. Unclassified
collision surfaces also bypass that veto. A separate masked-texture prototype
is testing explicit fixture synonyms with detector>=.3 and SAM>=.8 as a veto
on incompatible surfaces, including unclassified groups, without promoting
those detections into geometry or allowing them to select source materials.
No source-code adoption or general quality improvement is claimed yet.

Review sequence: keep the fitting/clearance change (shape-fit and interpretation
modules, shape and staged-fit tests, these notes and batch documentation) as one
focused six-file change. Any foreground texture-veto implementation belongs in
a separate usable change with its own appearance tests and documentation; do
not combine it with the fitting change in a pull request.


The full corrected ROOM106 trial completed at
`build/room106-staged-depth-polish-trial-v2`: final visual/depth/proportion and
known-headroom gates pass; IoUs .45494/.40994, maximum depth regression .004153,
minimum sampled clearance1800 with no unknown or obstructed samples. Source
and original export provenance are intact. `end-connection-audit.json` in both
corrected trials confirms base support but missing upper support. Thus neither
is a complete usable stair assembly despite passing visual and clearance gates.

The foreground appearance veto is now implemented behind optional recipe
`foreground_fixture_guard` in the semantic texture guard, baker and both batch
bake paths (all-view and excluded-view). It remains off by default. Three focused
tests cover incompatible/unclassified geometry, same-category preservation,
overlapping architecture masks, weak/mixed-label abstention and a rendered bake
that removes a projected bulb while preserving exactly the same depth image.
Forty-five batch and four fixture-palette regressions pass; Ruff passes.
The integrated ROOM10B bake at
`build/room10b-foreground-fixture-guard-integrated` exactly reproduces the
prototype mesh and all four preview PNGs. Package limits pass. Bright pixels in
camera2 decrease635→192; the remaining192 belong to the existing lamp primitive,
not room surfaces. This is a targeted appearance correction, not a fidelity
score or proof of correct fixture geometry. Camera3's992 bright pixels disappear;
camera0 is unchanged. No additional model inference or image upload is involved.


## Upper-gallery occupancy evidence audit

`build/room106-upper-floor-votes` clusters corroborated railing-line heights
(within150 units; medianY-6555.67), projects500-unit grid cells on each plane,
and checks explicit floor masks, valid estimated depth, fixture veto and log
error<=.2. No cell gets even one supporting view under the original floor masks;
this is missing/contradictory evidence, not evidence that no upper floor exists.
Camera0 gives238 depth contradictions; other accepted floor masks supply no
support. Original exports and code snapshots are unchanged.

A four-view/two-room local DINO/SAM probe at
`build/room3d-upper-floor-material-prompt-probe` uses generic carpeted/tiled/balcony
floor categories. ROOM106 still returns merged floor labels, with camera7 SAM
about.700. ROOM207 camera0 returns a balcony-floor proposal (.316 detector/.890
SAM). No generated room geometry is accepted or modified from these proposals.
A separate single-category carpet prompt tests whether floor-material recognition
is actually missing the upper carpeted gallery. Sparse SIFT points in ROOM106
cannot establish its extent: only6 survive in total, and high points are not
semantically established floor correspondences. No floor level or depth
confidence is changed merely to force a landing to pass.
# Procedural perimeter-gallery experiment (2026-10-12)

`room3d_assemblies.gallery_for_stairs` now proposes a perimeter walkway and
connector at an existing parametric stair's top height. Inputs are an XZ
envelope, walkway width, slab thickness and minimum connector depth; no room
identifier or image-specific coordinates enter the generator. It subtracts the
entire flight footprint, partitions connector/perimeter overlap to avoid
coplanar duplicate faces, and removes collinear clipping vertices before
eight-vertex prism subdivision. Geometry is explicitly estimated: the collision
envelope does not establish the original upper floor's extent or wall support.

`tests/test_room3d_galleries.py` checks exported OBJ triangles at five rotations:
both stair ends contact designated floor/gallery surfaces, the gallery does not
obstruct sampled treads, all exported triangles have positive area, and sampled
top faces do not overlap. Missing overhead remains unknown; those synthetic
tests do not certify headroom without an observed ceiling. Invalid dimensions
are rejected. The two new tests and the 12 assembly, 14 connection and three
floor regression tests pass; Ruff passes for the changed Python files.

The real-room experiment is isolated in
`build/room106-procedural-gallery-probe-v2/index.html`, with recipes, OBJ meshes,
eight-camera depth comparisons, connection/headroom reports and unchanged-source
hash provenance. Widths 1500, 2500 and 3500 all connect both ends and preserve
sampled 1800-unit clearance with the existing ceiling, but all fail the existing
0.005 whole-scene depth tolerance: maximum error increases are approximately
0.01982, 0.02358 and 0.02816, primarily in camera 4. Camera 7 also fails. Visual
inspection of camera 7 shows large estimated surfaces; this is not a fidelity
improvement. None is adopted in the batcher or original room exports. These are
editable creative proposals, not recovered floors or gameplay traversal proof.
Next work must constrain gallery extent/placement from independent evidence or
provide a deliberate manual draft workflow, rather than relaxing acceptance.

# Compact landing selection and batch integration (2026-10-12)

Following the rejected full perimeter galleries, a bounded compact-landing
experiment used the same accepted stair poses in ROOM106 and ROOM10B. ROOM106's
400-unit slab passes all eight camera depth checks (maximum increase 0.003758),
both designated walkable end contacts and known sampled clearance of 1800.
Camera 1's object score is unchanged; camera 7's loss rises from 0.516343 to
0.518274 and IoU changes from 0.409936 to 0.406337, within the preserved-fit gate.
This adds an estimated exit surface; it does not recover the complete upper
floor. Larger tested slabs (800/1200/2000) fail depth. ROOM10B's proposed slabs
have geometric contact, but its bottom contact is `boundary_4`, not a designated
walkable surface; selection therefore abstains rather than claiming traversal.

The reusable implementation is `tools/room3d_landing_fit.py`, integrated into
`fit_objects` through default-off `creative_stair_landings` plus
`creative_fallback`. Largest training-safe candidate is selected before held-out
validation, with no held-out-driven backtracking. Depth is measured against the
original pre-object scene in batch integration, not a second accumulated 0.005
allowance. Final object-fit preservation is separately checked before adoption.
Missing overhead never proves clear headroom. Accepted final diagnostics/images
are updated; rejected assemblies leave the previously accepted flight intact.

Three rendered tests cover actual batch integration, unchanged training trials
and selected candidate under altered held-out depth (including held-out
rejection), and abstention with unknown headroom. Artifacts:
`build/room3d-compact-landings-probe` (eight initial size/room trials),
`build/room3d-compact-landing-fitter` (reusable selector, reports, comparison and
unchanged-source provenance), and `build/room106-compact-landing-textured`
(projected appearance and runtime-package format check). The textured mesh has
19476 vertices, 6492 triangles, 70 textures and 72 materials. Camera 7 was
visually inspected: staircase texture stretching and estimated geometry remain;
this is a connection improvement, not a demonstrated fidelity improvement.
No new runtime traversal test or all-room adoption is claimed. The pre-existing
long-running assisted batch still has this flag off.

Focused review scope for this step: landing-fit module, shape-fit integration,
landing-fit tests, batch documentation and these notes (five files). Keep it
separate from the perimeter-gallery generator and foreground-texture changes.

# Live cross-room evaluation and fixture transfer (2026-10-12)

`evaluate_room3d_details --completed-only` now evaluates hash-checked completed
detail reports without a final manifest, excluding partial or changed inputs.
It reports category rejection reasons, groups scenario variants without treating
them as independent unseen-room tests, and labels execution-source provenance as
unrecorded when no verified frozen snapshot is available. Accepted creative
replacements no longer contribute the rejected fit's failed gates to rejection
counts. Evaluation output must remain outside input runs. Four evaluator tests
pass, covering altered/unfinished checkpoints, variant grouping, accepted
creative replacements and protection against writing into an input run.

The first snapshot at `build/room3d-live-detail-evaluation` contained 112 entries
in 56 room groups: 64/264 lamp proposals and 12/62 railing proposals accepted;
0/62 stair proposals accepted. These counts include variants and the older
general-batch configuration, not the recent experimental stair successes in
isolated outputs. They are neither fidelity percentages nor independent-room
success rates. Most lamp/railing rejections lack sufficient supported
observations; most stair rejections also have implausible proportions. The
evaluator can be rerun as the batch continues.

A controlled transfer of the existing fixture pipeline to ROOM203 and ROOM204
used identical cached masks/depths and source snapshots for baseline versus
single-view attachment/part fitting. Both rejected the selected fixture in both
configurations; see `build/room3d-fixture-transfer-203-204`, with reports and
unchanged-source/export provenance. This does not establish that every fixture
in those rooms fails. ROOM203's initializations reveal several plate centers at
the finite wall's top: parts of their attachment plates extend above the wall and
the physical contact check correctly fails. The better silhouette belongs to an
unattached candidate, so part refinement reports `unsupported_profile`.

Next experiment: bounded plate-half-extent insets within the same finite wall,
plus selecting a visible attached hypothesis for the explicit single-view
creative path before final part/mask/depth gates. No wall extension, new room
coordinates or threshold reduction is justified by this finding. Keep evaluation
changes (tool, tests, batch docs: three files) separate from fixture-geometry
changes when preparing reviewable commits/PRs.

# Finite-wall fixture inset: physical improvement, visual rejection retained

`room3d_fixtures.inset_wall_lamp` adds a bounded in-plane correction when the
attachment plate extends beyond its finite wall. It tests half-plate-extents plus
10 units, keeps the nearest valid five-point plate contact, preserves the source
anchor as provenance and returns nothing for an already valid or absent support.
The default-off `inset_fixture_anchors` flag inserts these seeds into fitting.
When that flag, creative fallback and single-view wall fixtures are enabled,
visible physically attached solutions are selected before floating silhouettes;
the existing final mask/depth/door/contact acceptance gates still apply.

Two rendered/geometric tests cover top/corner insets, absent supports,
nonmutation and opt-in selection over a better floating fit. The 46 shape tests,
five fixture-part tests, three landing-fit tests and three attachment tests pass;
Ruff passes on the changed Python files. Keep this fixture step focused on
fixtures, shape-fit integration, inset tests and the two relevant docs (five
files), separate from evaluation changes.

The transfer experiment `build/room3d-fixture-insets-transfer-203-204` completed
with source/export provenance. ROOM203's selected plate moved 81.2 units inside
`boundary_0`, with valid contact; final IoU is only 0.36170. ROOM204 also selects
a valid wall contact but reaches only 0.53061. Both still fail the unchanged
single-view support gate, so neither is adopted. This is a physical seed/selection
correction, not a demonstrated visual improvement or rollout to the general lot.

The recorded `bright-region-audit.json` identifies the next model limitation:
ROOM203's highlight components have areas [2,2,15,11], largest fraction 0.5;
ROOM204 has [12,9], largest fraction 0.57143. Contrast is adequate (0.2636 and
0.5979), so both abstentions arise from fragmented/multiple bright regions rather
than weak brightness. Current part fitting intentionally assumes one dominant
shade and requires 0.6 component coverage. The next geometry step should add
reusable multi-light wall fixtures and component-aware fitting instead of
relaxing the single-shade gate. Bright components are appearance evidence, not
proof of bulb identity or recovered ornamentation.

# Multi-light wall-fixture generator and transfer (2026-10-12)

The new `wall_candelabra` profile provides two to four separated shades, a common
stem/finite-wall plate and curved support branches. `span` controls lateral
spacing; intersecting shades and unsupported counts are rejected. Shade parts
remain contiguous after the stem/plate and carry explicit `part_roles`, so
appearance cannot mistake branches for luminous shades. Legacy `wall_sconce`
geometry keeps its single-light behavior. Anchoring/insets preserve the new
profile and its full topology.

Default-off `multilight_wall_fixtures` adds at most four valid wall-support seeds
with eight two/three-light size/height alternatives each. Selection and support
fitting use training views only. `bright_regions` accepts two to four contained
highlight components covering at least 80% of selected bright pixels, excluding
small glints; the old one-shade test still abstains on fragmentation. Part fitting
uses distinct shade/component assignments and weights the weakest match, rather
than allowing one large shade to explain several observed lights. Components
remain baked appearance hypotheses, not established bulb identities.

`build/room3d-multilight-transfer-203-204` completed with unchanged-source/export
provenance. ROOM203 still picks a single-shade candidate (IoU 0.46411) and rejects
it. ROOM204 accepts a supported three-light creative fixture: IoU 0.72436, maximum
all-camera scene-depth increase 0.000109. Its part matching observes two bright
components and scores only 0.19147; part refinement is unchanged. The third
generated light is estimated creative detail, not recovered or independently
verified. Single-view acceptance does not establish held-out object consistency
or runtime navigation.

`build/room204-multilight-textured/index.html` includes original/before/after
views and enlarged fixture crops. Local palette uses the selected fixture's own
mask and the explicit shade/support material roles. Source export remains intact;
the new mesh passes package-format checks (11220 vertices, 3740 triangles,
34 textures, 39 materials). Enlarged original/after crops were visually inspected:
the bright fixture is recognizable but blocky, and its third light and sharper
highlights differ from the original. This is not a claim of 80% fidelity.

Validation: four multi-light geometry/export tests, seven fixture-part tests,
46 shape regressions, 45 batch regressions, four tint tests, two inset tests and
three staged-depth tests pass; changed Python files pass Ruff. Review in focused
steps: geometry/fitting (shapes, fixtures, shape-fit, fixture-parts, two tests and
these notes: seven files); appearance-role export (batch exporter, multi-light
export test, appearance documentation: three files). The general running batch
still has these experimental flags off.

Separate batch-completion issue found from live logs: ROOM3100/3101 have one
camera and DA3 raises `Degenerate covariance rank` in trajectory scale alignment.
Their two failed entries must not be called live merely because folders exist.
The batch parent/workers continue on other rooms. Single-view scale support is
pending; disabling one API option alone will not fix this because the installed
API calls Umeyama even when `align_to_input_ext_scale=False`. ROOM310's camera is
outside its small collision envelope and no segmentation masks were produced,
so a floor/wall-based metric scale must not be invented for it.


### Single-camera depth and empty-render diagnostics (2026-10-12)

The local DA3 wrapper now bypasses trajectory alignment for exactly one input
camera, retaining the supplied camera and scaling relative depth by the camera's
look-at distance. This is an explicit editing prior, not a measured metric scale.
Two unit tests verify arbitrary model-scale invariance and invalid-input rejection.
Actual local runs complete for ROOM3100/3101; all four ROOM10B multiview depth
arrays remain exactly equal to the previous cache. Evidence is in
`build/room310-single-view-depth/verification.json`.

The frozen full pipeline for ROOM310 completed but its collision-derived assisted
mesh had zero visible pixels. Therefore processing success does not mean this room
was reconstructed. The new batch result records per-camera assisted coverage and
summarizes missing/empty views, propagates depth scale/evidence into both result
and assisted reports, and warns in the comparison HTML. The catalog distinguishes
`exported_empty_geometry` from a visible export. Legacy reports without coverage
remain explicitly unmeasured; no old checkpoints are rewritten.

`tools/room3d_depth_surface.py` exports a separate editable open depth surface
with discontinuity rejection and model-confidence groups. Even its reliable
faces are proposals, not observed geometry. A native-camera reprojection is not
independent fidelity evidence. Hidden geometry and navigation remain unresolved;
this exporter does not install or replace a room in the game or main batch.
Three rendered tests verify exact plane texture/depth reprojection in a rotated
camera, rejection of triangles across a depth step, missing coverage from a
shifted camera, uncertainty groups, output preservation and invalid-input rejection.

Review scopes remain separate: single-camera inference (depth module, its tests,
notes); editable draft (surface module, its tests, notes); export diagnostics
(batch, catalog, catalog tests, notes). Experimental source changes are not
retroactively active in the already-running batch workers. Neither empty-render
detection nor a single-view surface closes the five original reconstruction goals.

The new frozen end-to-end trial `build/room310-visibility-validation` completed
for both scenario variants. `diagnostic-verification.json` confirms zero assisted
coverage, `metric_scale_verified: false`, matching scale evidence in both reports,
and catalog classification `exported_empty_geometry`. The separate catalog is
`build/room310-visibility-catalog/index.html`. No source assets, old room exports
or checkpoints were modified. The validated editable guide is
`build/room310-depth-surface-validated/index.html`: 4941 vertices, 9549 triangles,
4341 uncertain faces and 51 rejected jump faces. Source-view coverage is 0.994818,
which measures reprojection visibility only. Source and shifted previews were
inspected; the latter exposes the open-surface limitations. The running general
batch continues independently with its previously imported code and flags.

### Parametric finishing objects (2026-10-12)

`room3d_additions.append_objects` now inserts the existing reusable staircase,
railing, lamp and column generators into a separate retouch variant. A small JSON
list supplies name, shape parameters, world origin and optional radians yaw;
`room3d_retouch --additions` shares ordinary group transforms and optional texture
reprojection. Every addition is explicitly estimated/unverified, receives its own
editable group and uses neutral material before reprojection. Shape inference is
not rerun. Original vertices, UVs, textures and export files are preserved. Invalid
specifications and duplicate groups are rejected before output creation. The
browser editor can transform these groups after insertion but cannot create them
directly yet. This finishing step does not fulfill automatic missing-volume
reconstruction or certify attachment/traversal.

Four tests cover rendered insertion and transformation, unchanged source hashes
and existing UVs, real texture reprojection, rejection before writes, and all four
shape families as distinct editable groups. Three existing retouch tests and Ruff
pass. The browser regression now accepts a specific group so the new object can
be tested rather than only a pre-existing wall.

The actual ROOM204 demonstration is `build/room204-parametric-addition-demo`, with
an offline editor `build/room204-parametric-addition-editor.html`. It reuses the
already tested creative lamp proposal from the separate multi-light trial; this
is workflow validation, not a new unseen-room generalization result. Source-file
hashes were verified unchanged. Camera0 original/baseline/retouched comparison was
visually inspected: reprojection darkens the fixture and the shape remains blocky,
so no fidelity improvement is claimed. Surface-only ROOM310 still requires
structural reconstruction; simply thickening its depth sheet would not supply a
faithful volumetric room.

Review scope: additions module, retouch integration, additions tests, browser
regression, batch usage documentation, these notes (six related files). No game
runtime, original assets or running-batch outputs are changed by this source step.

### Local palette association for finishing fixtures (2026-10-12)

The first insertion demo darkened the lamp because all manually added geometry
was interpreted (no camera projection), but the fixture had neither own-mask
observations nor shade/support materials. Added table lamps, wall sconces and
multi-light wall fixtures now retain those material roles. During retouch
reprojection, a shared added lamp is associated with eligible SAM lamp masks by
its visible rendered silhouette. Whole-scene depth removes occluded pixels;
minimum overlap 0.35 and a 0.1 margin on both the object and mask axes reject
ambiguous ownership. Score/confidence thresholds and a 16-pixel minimum reject
weak/tiny evidence. Reprojection uses only associated masks for the existing
fixture palette estimator and retains the recipe's foreground fixture guard.
No source images or game assets change, and no model is rerun. Geometry, source
association and material appearance remain separately reported.

Two additional rendered tests exercise successful association, two competing
masks, two competing objects, low detector/mask confidence and whole-scene
occlusion; a full retouch bake checks separate observed shade/support colors,
excludes bright unrelated pixels and verifies unchanged source files. Six
addition tests, four palette tests, four multi-light tests, three existing
retouch tests, three foreground-guard tests and 45 batch regressions pass (65
Python tests). Changed Python passes Ruff and the new object's translation,
camera switch, hiding and reset pass in headless Chrome.

`build/room204-parametric-addition-tinted/appearance-comparison.html` compares
before/after. Source geometry (including all exported vertex attributes) remains
exactly equal to the first variant, source room hashes remain unchanged, and a
pixel comparison across all five cameras confines the 131 changed pixels to the
added fixture in camera0; other cameras are unchanged. Its associated own mask
has silhouette overlap 0.72903. The dark support and bright blue-white shades
were visually inspected; the geometry remains blocky and creative, with no new
fidelity claim. Package-format validation passes: 11220 vertices, 3740 triangles,
34 textures, 39 materials. `appearance-verification.json` and `provenance.json`
record these checks. Existing ROOM204 shape reuse is not unseen-room evaluation.

Review scope: additions, retouch, reconstruct material-role bridge, additions
tests, batch usage documentation, these notes (six related files). The general
batch remains live with earlier imported source; these changes validate the
finishing path and do not retroactively update its room exports.

### Combined automatic pipeline evaluation started (2026-10-12)

Added an explicit experimental `creative-v2` detail profile to the normal batch
CLI, with the existing recipe behavior as the default. It combines the previously
isolated stairs/landings/fixtures/appearance improvements without per-room shape
code. Explicit room settings override profile defaults. Four tests verify exact
existing recipes, independent copies and room overrides, distinct effective
checkpoint recipes, invalid inputs and CLI rejection outside reconstruction.
45 existing batch regressions and Ruff pass.

The frozen full-pipeline trial is running under
`build/room3d-creative-v2-integrated-trial`; its driver is
`build/run_room3d_creative_v2_trial.py`. Development rooms are ROOM106/ROOM204;
additional automatic pipeline cases are ROOM308/ROOM401. It copies only these
completed rooms' model caches and baseline exports, records source/input hashes,
and runs the source snapshot with all masks. No hand-selected stair/lamp masks,
room-specific shape recipes or extra special prompts are injected. The baseline
execution source was not frozen, and that limitation is recorded. Successful
single-object experiments with hand-selected masks (including ROOM106's extra
camera7 stair prompt) are not equivalent evidence for this full automatic trial.
Do not claim rollout success until the new results and visual comparisons have
been inspected. The general batch remains running independently and unchanged.

Review scope: profiles module, batch CLI, profile tests, batch usage documentation
and these notes (five related files). The integrated trial is unfinished at this
record; neither profile tests nor starting its process establish reconstruction
quality or goal completion.

### Automatic recovery of the previously special-prompted stair view (2026-10-12)

The combined four-room trial is still live and has not completed a room yet.
Its provisional ROOM204 lamp fit reproduces the prior multi-light shape and
mask IoU 0.72436, but this inner optimization result does not establish final
acceptance or completion of the whole room. Do not substitute it for the terminal
room checkpoint or visual evaluation.

Added optional `stair_prompt_recovery` before batch object proposals/refinement.
The local recovery helper requires an existing exact stair mask in another view;
it retries missing/weak views with a fixed prompt list and keeps at most two
nonduplicate hypotheses with repeated-tread and predicted-depth evidence. Original
masks are retained, confidence is not promoted, and final fitting/geometry gates
remain required. Cached weights are used offline from the batch subprocess.
Recovery outputs get unique directories so a later retry cannot overwrite an old
proposal set. The feature remains off in `creative-v2` pending full fitting tests.

Three focused tests verify trigger/abstention policy, geometric-evidence and
confidence requirements, duplicate rejection, better retries over weak original
masks, exact negative-cache preservation, image hash rejection before writes and
no model loading when recovery is not triggered. The existing stair-label test,
45 batch regressions and Ruff pass (49 Python tests). Actual negative-room hook
integration completed successfully in
`build/room308-stair-recovery-hook-validation`; frozen source hashes remain
unchanged and no focused model inference was triggered.

The actual offline three-room recovery trial completed with frozen sources under
`build/room3d-focused-stair-recovery`: ROOM106 adds exactly one proposal in camera7,
none in cameras0/2/4/5; ROOM401 adds none in cameras0/1/3/4; ROOM308 does not trigger
and adds none. Completed original export checkpoints remain valid and segmentation
hashes are unchanged. The recovered ROOM106 mask then passed the existing tread
refinement in `build/room106-focused-recovery-refinement` and is exactly equal to
the earlier special-prompt refinement. The automatic associator groups camera1
and camera7 together; a separate camera3 stair hypothesis remains separate.
The recovered mask retains low SAM confidence (0.50637) plus structural evidence.
This removes the manual camera7 prompt/mask selection step, not the need for a
subsequent full geometry fit, independent heldout checks or navigation review.
Evidence is in `association-audit.json`, including exact mask reproduction.

Review scope: recovery helper, batch integration, recovery tests, batch usage
documentation, these notes (five related files). No room IDs or hand-selected
camera IDs are embedded in the recovery algorithm. The already-running combined
trial and general batch use their earlier source and do not pick up this change.

### Combined trial completed; quality limits confirmed (2026-10-12)

The frozen four-room `creative-v2` trial terminated successfully with unchanged
source snapshot, cached inputs and original exports. Completed room checkpoints
and package-format limits are verified separately in
`completed-export-verification.json`. Acceptance changes are ROOM106 0→3/7
(two lamps, one stair), ROOM204 0→3/4 (lamps), ROOM308 1→1/3, ROOM401 0→0/1.
These counts include creative interpretations, not faithful recovered objects.
None of these object groups has a reserved-camera evaluation. The updated
comparison report is `build/room3d-creative-v2-live-evaluation/index.html`.

Visual inspection covers ROOM106 cameras1/7, ROOM204 cameras0/2/4, ROOM308 camera0
and ROOM401 camera2. ROOM106's accepted stair is single-view, overwide, and looks
poor from camera7; the gallery/upper scene still has major missing or clipped
surfaces. ROOM204 lamps are recognizable but blocky and overly bright; its room
also has large unmodeled/neutral wall areas. ROOM308 retains rough/warped surfaces.
ROOM401's alleged stair mask appears to cover draped furniture rather than a
visible flight; rejection is useful abstention, not a successfully reconstructed
stair. The combined profile is therefore not promoted to default or rolled out
as a quality-complete solution.

A separate fully automatic ROOM106 run with `stair_prompt_recovery` is live in
`build/room106-full-stair-recovery-trial` (driver
`build/run_room106_full_stair_recovery_trial.py`). It freezes code/inputs and
performs the focused retry inside the ordinary batch path. It has reproduced the
camera7 proposal and passed stair refinement; its full geometry fit is unfinished.
Do not replace the single-view trial's limitations with the earlier isolated
manually selected-mask result.

### Shade size/spacing fitting experiment (2026-10-12)

Added optional `fixture_size_part_fit` to the existing part fitter. Bounded shape
variants vary shade size and separation while preserving wall anchor and shade
centre height. Two new tests check anchor/centre/spacing/roles and a rendered
synthetic recoverable size error, including a reserved-image sentinel. The
existing whole silhouette, part improvement and attachment conditions remain;
outer all-camera scene gates are unchanged. Candidate rejection reasons are
reported. Nine part tests pass. This feature remains off by default/profile.

The actual `build/room204-fixture-size-refinement` trial tests every accepted
supported lamp from the completed whole-room export, without selecting masks by
hand. It makes no changes: the two multi-light fixtures remain unchanged and the
single-light fixture abstains without reliable brightness evidence. A fixed
initial whole-scene depth reference prevents tolerances accumulating across
objects. Shapes and textures are not adopted as an improvement. This failure
supports separating bright regions from the full lampshade body instead of
forcing the whole shade to shrink to highlights. Full original-room and source
snapshot hashes remain unchanged; all proposals/reports and before/after renders
are retained. Later source adds rejection diagnostics; the trial's frozen source
is retained exactly as executed.

Review scope: fixture-part fitter, its tests, batch usage documentation and these
notes (four related files). Starting further trials, accepting some shapes, and
package checks do not complete the original reconstruction/generalization goals.

### Automatic recovered ROOM106 and shade appearance audit (2026-10-12)

The full recovered ROOM106 trial has finished with return code 0 and unchanged
frozen source/input/original-export evidence. It accepts two lamps and a shared
stair fitted from cameras1/7, without manual mask selection. Visual inspection of
both stair views still fails: the flight is far too wide, has an incorrect
platform arrangement, intersects the room, and leaves the gallery/rails
incomplete. `build/room106-full-stair-recovery-trial/visual-inspection.json`
records this failure. Two-view acceptance and physical gates are insufficient;
this output is not promoted. The next geometric work must constrain the shared
flight footprint/layout and intersections using all reliable observations.

Added optional `fixture_core_appearance` to separate bright shade regions from
the full fixture geometry. It uses only the fixture's verified training masks,
contrasts, and visible projected shade area to estimate a bounded centered
pattern. The pattern is creative/baked appearance, not recovered light shape,
albedo or emission. Source quads share a continuous chart across their triangles.
Three new tests cover own-mask selection, reserved-image non-access, low-contrast
abstention, area response, chart continuity, and exact rendered geometry/depth.
69 related bake, clipping, surface, fixture, guard, retouch, additions and batch
tests pass. Ruff passes the affected source/test files.

Frozen comparisons cover every accepted fixture in ROOM204, ROOM106 and ROOM308,
plus ROOM401 as a negative case. Bright blocks become smaller bright regions;
ROOM401 stays unchanged. Exact baked vertex positions/colors and non-core
UVs/materials remain identical, all rendered depth/coverage stays identical, and
changes are confined to visible shade materials. The initial ROOM308 visibility
check against unsplit source faces failed at a shared-depth rasterization tie;
the retained trace is not silently discarded. Verification against the actual
subdivided baked shade materials passes and additionally checks all non-shade
attributes/materials exactly. Geometry fidelity and navigation remain unresolved.

Review scopes: appearance backend (preview, reconstruct, new appearance module,
new tests and these notes); separate batch integration (batch, usage docs and
integration evidence). No default profile change, no mass replacement of completed
exports. Retouch reprojection of previously baked parametric roles/charts remains
an explicit limitation; ordinary transforms retain the baked UVs.

The frozen `build/room204-core-batch-hook` run additionally verifies the ordinary
batch and heldout bake hooks with an existing fitted recipe and cached masks.
It finishes with nonempty core reports and unchanged source/input/original-export
checks. No geometry optimizer quality claim follows from this hook test.

### Stair tread extent diagnostic (2026-10-12)

Added opt-in `constrain_tread_extent` to avoid cropping predicted tread edges to
the observed mask before scoring. A new rendered-projection test confirms that
excess width is penalized while actual scene occlusion still removes hidden
overshoot. All 47 shape tests and Ruff pass. Default/profile behavior is unchanged.

`build/room106-tread-extent-probe` freezes tools and checks thirteen width/anchor
alternatives on the fully automatic recovered stair's training cameras. Both old
and new objectives still select the existing broad flight. This is a failed
real-room improvement, not an adopted geometry correction. Source masks, depths,
refinements and original export remain unchanged. The orientation diagnostic
shows a large disagreement between the current fitted yaw and the coherent
image-line direction already computed by `tread_run_direction`. The next
experiment is to retain that training-view orientation as an explicit objective,
instead of merely offering it in the initial seeds and allowing it to drift.
Image-line identity remains estimated, and silhouette/depth/physical gates must
still apply. Scope for this separate experiment: shape fitter, shape tests and
these notes (three related files).

Implemented optional `constrain_tread_direction`: the coherent direction from
training-view image lines supplies a bounded axial orientation penalty throughout
the search, rather than only influencing a seed. Its sign does not determine the
ascending end. No coherent evidence means no penalty. The prior report records
the direction, training/excluded cameras, weight and uncertainty. Two further
tests verify depth-independent orientation/sign ambiguity/insufficient-evidence
abstention and that changing reserved-view lines leaves the training prior and
training scores exactly unchanged. All 49 shape tests pass; Ruff passes.
Both new stair options remain off by default/profile.

The fully automatic comparison run `build/room106-tread-direction-trial` uses a
frozen source snapshot and cloned cached model inputs, the same `creative-v2`
profile, focused stair recovery, and the two opt-in tread objectives. It is still
running; no geometry outcome or successful correction is claimed yet. Its driver
is `build/run_room106_tread_direction_trial.py`. Inspect the actual result and both
camera views before deciding whether the direction hypothesis helped.

### Preserve fixture appearance during finishing (2026-10-12)

Retouch reprojection previously lost parametric shade/support roles from an
already baked mesh. The opt-in backend preservation path, enabled by the retouch
tool, now retains recognized generated fixture role materials, their Kd/texture
pixels, vertex colors and UVs. Preserved faces are not subdivided again. New
material names retain their roles for subsequent backend rebakes. Only
interpreted lamp groups qualify. Preserved full-view appearance is forbidden in
heldout evaluation; it cannot silently reuse pixels from a reserved view.

Two additional appearance tests exercise the actual retouch entry point with a
textured shade, colored support and vertex colors, a second backend rebake, role
name validation and early heldout-preservation rejection. All five appearance
tests and Ruff pass; the related regression suite passed 70 tests before adding
the final heldout guard, and that guard subsequently passed its targeted suite.
The final related regression suite passes 71 tests, including that guard and
retouch with the optional core flag enabled. Retouch may estimate cores for
freshly added shapes when the recipe requests it, while already baked fixture
charts/materials are preserved.

`build/room204-preserved-fixture-retouch-cached` freezes code and input caches and
tests a no-edit reprojection and an illustrative translated lamp. For all five
cameras in each variant, the complete image exactly matches the expected textured
geometry, as do fixture appearance and scene coverage/depth (depth within numeric
serialization tolerance). Input/source snapshots and original exports are
unchanged. Visual inspection confirms the fixtures and room appearance survive.
The retained initial no-cache trial preserved fixture pixels but changed other
surface projections; cached SAM/DA3 inputs are necessary to reproduce the original
appearance decisions. Neither trial validates changed gameplay collision or
navigation. Review scope: appearance helper, baker, retouch, tests, batch usage
docs and these notes (six related files).

An independent whole-room validation is running at
`build/room210-tread-direction-validation`, with a copied baseline, frozen tools
and cloned model inputs. ROOM2100 is the first remaining completed room in sorted
order with at least three confident exact-category stair views, excluding prior
trial rooms. Its baseline has a shared stair observed in cameras3/4/6, reserving
camera6. The full automatic creative-v2 pipeline uses focused recovery plus the
two optional tread objectives; no masks or shape dimensions are selected manually.
This is a pending generalization/heldout test, not an accepted reconstruction.

### Completed tread-direction trials and bound visual evaluation (2026-10-12)

Both full trials finished with return code 0 and unchanged frozen tools, cloned
inputs and original exports. ROOM106 still accepts two lamps and one stair. The
fitted stair yaw changes from -5.37 degrees to -72.42 degrees, closer to the
training image-line hypothesis near -60.06 degrees. Visual inspection of source
views1/7 still fails: the flight dimensions/layout and large exposed faces are
incorrect, gallery/rails are incomplete, and texture projection is distorted.
No shape is promoted as a successful reconstruction. The optional direction
prior improves one measured relationship, not the complete room.

ROOM210 accepts neither stair hypothesis. The shared proposal reserves camera6
and obtains low heldout silhouette IoU (~0.096); its training line evidence cannot
establish a coherent horizontal direction, so the direction prior abstains. In
reviewed originals3/6, the visible structures are wooden walls/posts rather than
a clearly visible flight. Rejection is useful, but the room still has large
missing/neutral surfaces and does not pass visual reconstruction review. The
existing detector's confident category labels are not reliable object identity.

Added `room3d_visual_audit.py` and evaluator `--audits` support. Explicit visual
findings live outside completed exports and bind to checkpoint hash/key plus
full export inventory and listed reviewed images. Changed materials, images,
geometry, checkpoints or incomplete exports invalidate approval. Seven evaluation
tests cover this lifecycle, report integration, input preservation, scope/camera
validation and malformed reviews; Ruff passes. Review scope: audit helper,
evaluator, its tests and the two room3d documents (five related files).

Actual recorded findings for the old recovered ROOM106, new ROOM106 and ROOM210
are in `build/room3d-bound-visual-audits`. The combined report is
`build/room3d-tread-direction-evaluation/index.html`. These reviews do not infer
anything about unviewed/hidden surfaces or runtime navigation.

The DA3 cache retains correctly resized conditioned intrinsics (336/320 focal
scale) and input-aligned extrinsics; no simple focal-resize bug was found. Existing
appearance-only floor calibration indicates different view-wise scale biases,
and ROOM106 camera7 has no confidence-qualified floor mask/calibration samples.
A frozen local focused-floor probe is running at `build/room3d-focused-floor-probe`.
It retries floor/carpet material prompts only for missing calibration views in
ROOM106/210, keeps original confidence values, excludes competing object masks,
and measures calibration against the estimated rendered floor. It does not merge
caches, modify depth scales, fit geometry or claim independent metric evidence.

The focused-floor probe completed with unchanged inputs/tools/exports. Only
ROOM106 camera7 needed a retry; ROOM210 already had accepted calibration in every
view. One rug proposal qualifies by the original detector/SAM thresholds and
provides 1803 floor samples, but its held-pixel plane-calibration error (~0.166)
exceeds the existing 0.08 limit. The proposed scale (~1.194) is rejected. Tiled-floor
masks remain below the existing SAM threshold. No thresholds are relaxed and no
scale correction is adopted; the missing floor calibration is not solved.

A thirteen-candidate width/anchor diagnostic on the newly oriented, fully
automatic ROOM106 flight is running at `build/room106-oriented-width-probe`.
Its final fitted width still exceeds 9000 game units, although initial seed
widths are capped at 4000 and the existing proportion checks only constrain
step rise/depth and slope. This diagnostic tests whether a narrower creative
flight can agree with the same source evidence after fixing orientation; no
candidate is automatically adopted or claimed faithful.

The newly oriented width diagnostic completed: both existing objectives still
prefer the overwide flight. A further controlled diagnostic at
`build/room106-rendered-mask-width-probe` scores the actual scene-visible mask
without letting DA3 depth censor predicted pixels, while retaining the original
depth residual term. This tests whether model-only occlusion hides visible excess
geometry from the optimizer. No production objective or default is changed yet.

The rendered-mask diagnostic completed without adopting a candidate. Five fully
baked creative width/structure variants are available in
`build/room106-creative-width-visuals/index.html`. Visual inspection of the
compact-right flight from cameras 1 and 7 still shows incorrect shape/placement
and missing upper gallery, despite passing the global depth gate. Width bounds
alone do not solve this failure. All variants use the same estimated appearance
policy; the row named `before` therefore is not a pixel-identical reproduction
of the original unstyled export. Original exports and input caches are unchanged.

`build/room106-floor-calibrated-ranking/results.json` compares these fixed
variants with raw depth and depth calibrated against the unchanged baseline
floor. Calibration is fixed before candidate scoring, uses the existing tiled
fit/validation split and thresholds, and never mutates model arrays. Seven views
qualify; camera7 has no qualified floor samples. The floor prior roughly halves
DA3 depth in cameras3/4. Baseline mean depth error falls from 0.3591 to 0.2535,
but both score variants still rank the visually unsuccessful compact-right flight
first; their pass/fail decisions are identical for all five candidates. Scale
bias exists, but removing it does not repair this particular ranking failure.
This is a diagnostic against estimated geometry, not independent fidelity or
reserved-view validation. Frozen tools, input inventories and the complete
original export are verified unchanged; the diagnostic script passes Ruff.

The isolated ascent-sign comparison completed in
`build/room106-ascent-sign-visuals/index.html`: it reverses the flight's horizontal
run while preserving its footprint and height range, and tests the same
width/anchor alternatives. Cameras1/7 were visually inspected for the wide,
left and centre reversals, and camera7 for the right reversal. Reversed steps
are more recognizable, but none resolves placement/proportions or the missing
gallery. The wide, centre and right reversals pass the depth gate despite these
visible defects. Original exports, model caches and frozen sources remain
unchanged. No candidate was adopted.

`build/room106-ascent-sign-score-probe/candidates.json` independently scores
eight width/anchor/sign combinations using the recovered shared staircase
observations and both original and scene-visible silhouette objectives. Both
objectives retain the original overwide ascent sign. Reversing that wide flight
raises camera7's scene-visible mask IoU from 0.5049 to 0.5452 while lowering
camera1's from 0.5613 to 0.4038. This is a cross-view tradeoff, not a successful
general correction. The scoring probe retains original solid construction;
the appearance trial uses regular suspended steps, so these are distinct
controlled comparisons and their results must not be treated as identical
candidates. Tread orientation remains axial: neither the orientation prior nor
one improved camera proves the ascent sign, endpoint placement or room fidelity.

### Tread evidence diagnosis and explicit layout prior

Visual overlays in `build/room106-tread-evidence-inspection` reveal that the
accepted camera1 tread evidence actually follows the sloping railing. Camera7
also includes railing/floor-carpet edges. These are plausible depth-slope
patterns, but not dependable tread observations. Fitting their orientation can
therefore reinforce the wrong flight geometry.

`room3d_stair_edges.py` now offers an optional explicit XZ horizontal-axis prior.
It compares image backprojection planes with supplied axes without using model
depth. Recipes forward `tread_horizontal_axes` to the refiner; CLI callers use
repeated `--horizontal-axis X Z`. Normalized axes are recorded and cached
refinements refuse mismatched priors. The default and existing profiles retain
the unrestricted search; no room identifiers or manually traced lines are used.

The frozen, inference-free four-room trial in
`build/room3d-layout-tread-probe/index.html` uses the same explicit orthogonal
assumption for every observation. ROOM106 camera1's one false region and
camera7's 38 regions are rejected. ROOM10B camera2 retains all eight candidate
regions, with visible actual treads in the best region. ROOM210 still retains
one railing-edge region in camera4, so this prior is insufficient as an identity
test. ROOM401 has no regions in either mode. Original masks, confidence, depths,
exports and frozen tools are verified unchanged. No geometry is adopted.

Keep this reviewable change to five files: stair-edge helper, batch forwarding,
projective/cache tests and the two room3d documents. The separate build-folder
diagnostics are local evidence. The feature does not constitute complete
stair/railing recovery or a validated replacement for the current batch profile.

Validation: 49 shape tests, 45 batch tests, four stair label/recovery tests and
four new projective/cache tests pass; Ruff passes for all three Python files.
The actual offline refinement CLI completed at
`build/room106-layout-refinement-hook-v2`: six records, zero accepted structural
refinements, axes recorded on every record, complete original export and input
inventories unchanged. This is a refinement CLI check, not a full new geometry
run. The first hook trial is retained separately: its CLI succeeded, but its
inventory assertion failed because Python created `__pycache__` files in the
frozen tools directory. The second trial records its before-inventory and
disables bytecode writes; no source/input differences are ignored.

The full fitting/baking trial with the explicit orthogonal prior completed at
`build/room106-layout-aware-full-trial`. It uses a frozen copy of the same
automatically recovered segmentation and depth inputs as the earlier direction
trial, without repeating detection/recovery or selecting masks manually. It
tests how removing unreliable tread evidence changes actual accepted
geometry; the refinement-only result cannot answer that question. Source code,
recipes, commands and initial input inventories are recorded before execution.

That full trial exited successfully and verifies unchanged frozen sources,
inputs and original export. Only two lamp hypotheses were accepted; both railing
hypotheses and the remaining staircase hypothesis were rejected. Without the
false structural edge evidence, the original camera1/7 masks no longer qualify
for the staircase fitter. The final recipe contains no staircase, and actual
camera1/7 renders visibly lack the flight and complete gallery. A completed-export
bound visual audit records `needs_further_geometry_work` in
`build/room3d-bound-visual-audits`. The filter is therefore not promoted as a
room-reconstruction improvement: removal of false evidence must be paired with
an independently constrained creative replacement, not mistaken for success.

Completed before/after evaluation is available in
`build/room106-layout-prior-evaluation/index.html`, with both completed exports
and their externally bound failure reviews. The evaluator now recognizes frozen
source manifests that use bare module names as well as `tools/`-relative paths;
it verifies the corresponding snapshot bytes. Eight evaluation tests cover both
formats and reject modified snapshots. This is a separate three-file reporting
change (evaluator, evaluation test, this plan), not a geometry-quality improvement.

### Bounded creative replacements from approximate masks

An opt-in creative route addresses the gap exposed by the layout-filter trial:
discarding false tread lines otherwise excludes the correct moderate-confidence
stair masks entirely. Exact stair-category masks can now remain approximate
design cues without confidence inflation. Training-only initialization explores
regular suspended stairs on explicit layout axes, both ascent signs and bounded
widths; it preserves seed centers/base heights and subjects candidates to the
ordinary geometry/appearance and heldout evaluation path. No room IDs or manually
traced image regions are encoded in the generator. Maximum width, layout angle
and step-proportion constraints also guard polish/final procedural adoption.

Three new tests exercise category/threshold/default behavior, unchanged masks
and confidence, actual replacement shape parts/center/base/width/directions,
rotated layout support, and actual three-camera `fit_group` evaluation with
injected polish starts. Changing reserved-camera mask qualification leaves the
training score/prior unchanged; reserved-only qualification cannot activate the
prior. Ruff passes. The five-file review scope is creative-stair helper, shape
fitter, creative-stair tests and the two room3d documents.

After adding the final fallback/proportion guards, all 49 shape-regression and
45 batch tests also pass, alongside the three new creative-stair tests. These
checks establish bounded behavior and unchanged defaults, not visual fidelity.

`build/room106-creative-replacement-trial` is the completed full-room experiment,
using the same recovered caches and original images as the preceding two trials.
It is frozen at the initial generator/observer integration; later explicit final
fallback/report/proportion guards are not retroactively inserted into that source
snapshot. Keep its evidence qualified accordingly and do not restart it just
because observation yields or because a later source revision exists. A source
manifest, command, recipes and input inventories precede execution. Visual
improvement, cross-room generalization and complete reconstruction remain unproven.

The ROOM106 experiment completed with two accepted lamps and no accepted stairs
or railings. Its approximate mask groups are cameras1/6, camera3 and camera7;
none supplies a reserved view. The bounded candidate in cameras1/6 has base
Z≈29865, beyond the room boundary maximum Z≈28293, and zero silhouette IoU in
both views. Final camera1/7 renders visibly lack the flight. Immutable inputs,
source snapshot and original export are verified; a completed-export-bound
failure audit records the inspected views. Width/orientation limits do not fix
the badly localized DA3-derived seeds.

A current-source control, `build/room300-creative-replacement-validation`, also
completed with unchanged inputs/snapshot/export. It was selected lexicographically
from previously untested rooms with exact staircase detections in at least three
views at detector≥0.3 and SAM≥0.65, before image inspection. Those masks correspond
visually to courtyard pavement/planter edges. Raw association yields three separate
single-view groups, so no reserved-view validation is claimed. All three stair
hypotheses are rejected; one railing is accepted. Inspected final cameras0/3/5
have no staircase, but retain large neutral blocks and incomplete vegetation and
surface geometry. This is a useful rejection control, not a full visual pass.
The original copied baseline uses older source/options and is not a controlled
causal estimate of the new flag's effect. Combined reports are in
`build/room3d-creative-replacement-evaluation/index.html`.

The current observer also preserves the ordinary path for masks with accepted
repeated-tread evidence. A regression assertion verifies the creative flag does
not relabel those masks or alter their confidence. The new three-test suite and
Ruff pass after this qualification change.

`build/room106-mask-volume-probe` diagnoses an alternative localization without
model depth: project a 96×96 XZ grid within existing room-boundary bounds at
500-unit height intervals into training masks1/6, permitting a 3-pixel mask
dilation. The shared floor support has 120 samples around X≈15174, Z≈17582, and
its projections were visually inspected near the actual stair base in both
images. Higher mask-intersection slices remain nonempty through 6000 units.
These are estimated mask volumes conditioned on the existing floor prior, not
recovered metric geometry, navigation or independent validation. Raw depth still
determines group association; no camera7 data or candidate geometry is used in
this localization. Original segmentation/depth inventories remain unchanged.
The next initializer should exploit these camera/mask constraints instead of
only moving seeds already misplaced by predicted depth.

### Camera/mask volume initialization

The creative-stair helper now implements that reusable initializer. It intersects
training masks on a 96×96 XZ grid inside existing boundary bounds at floor/upper
planes from 0 to 6000 units in 500-unit steps, with a three-pixel tolerance. Floor
and upper support supply endpoints on each supplied layout axis/sign; widths,
regular steps and inclinations keep the existing physical design limits. No
model depth, manually selected pixels or room IDs enter localization. An empty
depth-seed list no longer prevents this fallback from running. Original model
observations, cameras and gameplay geometry remain untouched.

Four creative-stair tests pass, including actual three-camera render fitting
with depth initialization disabled. Replacing the reserved mask with an empty
mask leaves all starting shapes, training scores and selected training result
unchanged. Another test proves that scene-only silhouette visibility preserves
real scene occlusion and the model-depth residual. This visibility policy applies
only to the explicit approximate creative route; global depth gates remain in
place and default behavior retains model-based foreground occlusion.

The frozen `build/room106-mask-volume-seeds` diagnostic completes with unchanged
model caches, sources and original export. It generates 32 candidates for the
automatically associated views1/6. Its best training scene-visible IoUs are
0.1002/0.0933; generated untextured full-scene previews were inspected in both
views and remain largely occluded. This is a successful initialization code
path, not a successful visual reconstruction or complete acceptance check.

Current-source full trials are running at `build/room106-mask-volume-full-trial`
and `build/room300-mask-volume-validation` with frozen source/inputs/recipes.
They retain every mask and all ordinary global geometry/appearance gates, and
must be reviewed after completion. No experiment is restarted or retroactively
modified to match a later source change. Keep this five-file step (creative
helper, fitter, creative tests, two room3d documents) independently reviewable.


Follow-up scope: keep visual enclosure proposals independently reviewable
(`room3d_supports.py`, a focused visual-enclosure test file, these two documents).
The mask-volume ROOM106 diagnostic attributes foreground obstruction mainly
to generated boundary extrusions, rather than the ceiling. Training-only
approximate multi-view mask agreement now permits a bounded experimental
visual-height search, without changing default behavior or source collisions.
Tests cover opt-in isolation, geometry immutability, weak/wrong projections,
single-view/structurally observed exclusion and reserved-view independence.
The diagnostic height750 version exposes more recognizable steps but does
not recover the gallery or pass visual reconstruction; a full automatic trial
of this source step is required before promotion. No room-specific boundary
identifiers or manual masks enter production code.

The new full automatic enclosure trial is running separately at
`build/room106-visual-enclosure-full-trial`, with its own frozen source, inputs
and recipe. The older mask-volume trial remains running from its original
snapshot; neither is restarted or mutated. Focused enclosure tests (3), shape
regressions (49) and Ruff pass for this step. Real ROOM106 observations confirm
that cameras1/6 qualify as approximate training observations; only camera1's
failed-fit diagnostic proposes boundary changes in the current full fit.

The mask-volume full ROOM106 trial has now completed with immutable sources,
inputs and original export. It accepts two lamps and rejects all three stair
hypotheses. The shared1/6 fit passes global depth but fails silhouette support;
its creative fallback also lacks a supported top-end connection. Inspected
final cameras1/6/7 show missing stair flights, gallery, columns and arches.
Checkpoint-bound audit `room1060-060295865ebc.json` records failure, and
`build/room3d-mask-volume-evaluation/index.html` compares this completed trial
with the completed ROOM300 negative control. Neither is a visual pass. The
separate full enclosure trial remains active; it must not be equated with the
frozen seed diagnostic or judged from preliminary object files.


Follow-up scope for joint creative landings: interpretation helper, fitter,
landing tests and these two documents (five files). The creative fallback
previously required an upper end connection before acceptance, while the
landing stage ran only after acceptance. The new explicit option explores a
landing and stair together using the existing training/reserved-view fitter.
Selection is not preempted by reserved-view depth or placement evidence; final
assembly acceptance still checks them. Tests render three cameras, verify
default rejection versus joint acceptance, preserve the flight endpoints, and
change only reserved depth to force rejection with identical training candidate
selection. A successful integrated fallback carries its assembly report so
the caller does not incorrectly treat the last landing as a staircase or add
a duplicate assembly.

The integrated ROOM106 fallback replay now completes: all six final gates
(placement, original-baseline .005 depth tolerance, door clearance, stair
support, upper/lower walkable contacts and headroom) pass with a selected
landing. Source snapshot, input caches and original completed export remain
unchanged. It remains a component replay with two real observations and no
reserved view, not full-room fidelity or traversal proof. Four rendered landing
tests,49 shape regressions,three depth/headroom regressions and Ruff pass.
The source step is now frozen in the separate full automatic
`build/room106-joint-landing-full-trial`; the earlier enclosure trial continues
unmodified from its original snapshot. Inspect both completed final exports
before promoting either experimental option.

The full visual-height trial completed with two lamps accepted and all stairs
rejected. The best shared stair fit selectsheight750 and training IoUs
0.447/0.264, but fails silhouette support; its fallback still lacks a top
connection. All eight final assisted PNGs are byte-identical to the earlier
mask-volume full trial. External checkpoint-bound audit
`room1060-a961c6100ef7.json` records the reviewed failure;
`build/room3d-enclosure-evaluation/index.html` includes the completed trials.
The integrated joint landing replay now has same-policy textured comparisons
at `build/room106-joint-interpretation-probe/index.html`. Viewed1/6/7 show
recognizable steps but still incorrect gallery, upper geometry and missing
columns/arches; it is not a visual pass or an adopted full-room result.

An independent full joint landing trial is running at
`build/room101-joint-landing-validation`. Selection was lexicographic from
completed metadata outside prior detail-trial rooms, requiring a shared stair
proposal in at least two views; ROOM101 group3/4 was the first match. Only
after freezing this selection were the input images reviewed: both show a real
wooden stair flight. All masks remain in the full pipeline, with unchanged
local model caches and source snapshots. Neither real two-view group provides
a reserved camera; only the synthetic three-view test currently proves that
implementation property. Baseline exports have older flags/source and are not
controlled causal comparisons.

The separate local ROOM101 focused recovery now completes with zero added
proposals in cameras0/1/2/5. Original export, cached inputs and frozen sources
are verified unchanged. All six input images were inspected:3/4 show the real
flight,2 has distant partial architecture and5 largely the underside/landing.
The recovery retains its original exact-label, mask and repeated-tread guards;
merged detector phrases are not silently relabeled or promoted. Therefore the
full ROOM101 trial retains two shared observations. Do not claim a real reserved
stair view from the synthetic test or from all-camera scene-depth checks.

The full joint landing ROOM106 trial now completes with immutable sources,
inputs and original export. Its final recipe includes three door panels,
barrel, ceiling, two lamps, staircase and landing; assisted.obj is2,058,928
bytes. The shared creative flight uses width4000,run5997.53,rise3976.79,22steps
and an estimated2000 landing. Its all-eight-camera original-baseline depth
deltas stay within.005. The evaluator confirms one accepted shared creative
stair assembly with its own known headroom and no inherited rejected-fit IoU.

All eight final camera renders were inspected. External checkpoint-bound audit
`room1060-24d4a9c24721.json` records failure: steps are recognizable but the
gallery, columns and arches remain missing, large upper ceiling/black regions
and provisional vertical blocks obscure the room, and projected architectural
textures remain distorted. Cameras3/4 show severe enclosure obstruction and7
incorrect upper geometry. The comparison is
`build/room3d-joint-landing-evaluation/index.html`. This is a completed automatic
code path, not a visual pass; neither experimental option is promoted. The
remaining bottleneck includes the base room enclosure and upper floor, beyond
the stair generator. The independent ROOM101 full trial remains running.

The independent ROOM101 full joint landing trial also completes with unchanged
snapshot, caches and original export. It accepts five lamps plus one shared
staircase with landing. All six final renders were inspected; external audit
`room1010-7179a24ad516.json` records visual failure: recognizable steps in3/4,
but wrong proportions/placement, intrusive enclosure blocks in0/1/3/4, stretched
projections and incomplete upper/underside architecture. The shared group
still has only views3/4; focused recovery adds no third observation.
`build/room3d-joint-landing-independent-evaluation/index.html` combines both
completed joint trials with the courtyard control. This proves the automatic
assembly path runs in two rooms, not an80% fidelity result or a profile
promotion. Base enclosure topology/height and upper surfaces now require direct
investigation rather than another stair-only fit cycle. The all-room batch
continues independently using its earlier recipe; these experiments are not
retroactively applied to its completed entries.

Enclosure diagnostics are now complete, with immutable inputs/sources/original
exports. `build/room106-ceiling-level-probe` uses existing image-line level
hypotheses corroborated across0/1/6 (medianY−6555.67) to try ceiling clearances
1800/2500/3500. None passes the unchanged all-camera depth gate; maximum
regressions are.0567/.1070/.1720. Inspected untextured views1/5 also show gaps
between the unchanged wall tops and raised roof. A roof-only change is not
adopted; the level hypotheses are not verified original floor elevations.

`build/room101-floor-occlusion-probe` attributes6257 pixels of a high-confidence
visible-floor mask in camera5 to generated boundary11, with5422 reliable model
pixels and observed depth4835.5 versus proxy surface1887.8. This boundary is a
non-wall collision rectangle already covered by the accepted stair AABB; the
shared-wall-only shortening helper correctly refuses it as a different class.
`build/room101-replaced-proxy-probe` selects it automatically from floor/depth
contradiction and>=80%AABB overlap, without hard-coded room/boundary IDs.
Heights300/750 and omitting only its visual proxy pass the all-six-camera depth
gate, retaining all actual primitives and original collision records. Same-
appearance textured views3/4/5 were inspected: floor under the stair is modestly
more exposed in5, but the enclosure and stair proportions remain poor. No
geometry is adopted or profile promoted. The comparison is the probe's
`index.html`. Before integration, replace provisional AABB overlap with actual
ground-flight/connected-landing footprint containment, exclude structural walls
and raised supports, retain fixed evidence and all-camera checks, and test the
rule outside this example.
