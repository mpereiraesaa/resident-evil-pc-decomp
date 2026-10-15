# Character model preview

Set `RE1_MODEL_VIEW` before launching the game to inspect a character without
starting gameplay. The preview loads the original `.EMD` with `LoadEntityEMD`,
sets up its joints with the usual animation functions, and submits each joint
through `render_entity` and the game's TMD renderer. The model turns
automatically, completing a revolution in about eleven seconds. Close the
window to exit.

For scripted asset checks, `RE1_MODEL_VIEW_EXIT_FRAMES=5` exits after five
rendered frames. Leave it unset for interactive inspection.

From the repository root with the Windows build and its usual `config.ini`:

```sh
RE1_MODEL_VIEW=wesker wine build/windows/residentevil.exe
```

The Linux build uses the same variable:

```sh
RE1_MODEL_VIEW=jill build/linux/residentevil
```

Names: `chris`, `jill`, `barry`, `rebecca`, `wesker`, `zombie`,
`zombie-jill`, `naked-zombie`, `naked-zombie-jill`, `green-zombie`,
`green-zombie-jill`, `cerberus`, `hunter`, and `tyrant`. Names without `-jill`
select Chris's EMD block; the suffix selects Jill's block. This matters even
when the enemy type id is odd.

Every entry in the original 53-entry model table can also be selected by its
decimal index: `RE1_MODEL_VIEW=40` is Wesker; `RE1_MODEL_VIEW=j:4` is the Jill
scenario's white-coat zombie. Indices and filenames are listed in
`src/game/EntityModelLoader.cpp`.

For the original mansion main hall view, add `RE1_MODEL_VIEW_SCENE=main-hall`:

```sh
RE1_MODEL_VIEW=chris RE1_MODEL_VIEW_SCENE=main-hall wine build/windows/residentevil.exe
```

This loads Stage 1 `ROOM1060.RDT`, its camera 4 background PAK, camera transform,
field of view, ambient light and three point lights. The character stands on
the upper hall passage at `(16000, 0, 23500)`, facing the camera. Other camera
selections place the character at the game's ground-floor starting position
`(17000, 0, 5000)`.
Any selected model can be placed there. The hall view stays fixed so it can
be compared with gameplay; the default studio preview rotates on black.
Set `RE1_MODEL_VIEW_CAMERA=0` for the earlier overhead view, or select any
other main hall camera from `0` through `7`. The optional
`RE1_MODEL_VIEW_X`, `RE1_MODEL_VIEW_Y`, `RE1_MODEL_VIEW_Z` and
`RE1_MODEL_VIEW_ANGLE` variables can adjust the placement for a capture.
`RE1_MODEL_VIEW_FOV=512` increases projection distance for close inspection of
small models (accepted range 64–512). It uses the ordinary TMD projection and
affects only the viewer, not gameplay.

Enemy and NPC models play their EMD animation 0. Chris and Jill remain in
their rest pose unless `RE1_MODEL_VIEW_PLAYER_ANIM=0` (or another valid
animation index) is set; this samples the ordinary animation pair in their
CHAR EMD. The hall preset does not run room scripts, shadows or camera mask
sprites. Without either variable, the game takes its normal boot path.

## Texture smoothing comparison

The game uses point sampling for TMD model textures. Set
`RE1_MODEL_TEXTURE_FILTER=linear` to use bilinear sampling for all in-game
TMD models, including the character viewer. This softens individual texels
but does not add face, clothing or mesh detail. Leaving it unset preserves
the original filter. The Windows renderer's `[Display] SMAA=1` is a separate
edge smoothing pass and needs `smaa/SMAA.hlsl` beside the executable.

## Chris geometry pilot

Set `RE1_CHRIS_MESH=1` to refine the original Chris player mesh when it is
loaded, in gameplay or in this viewer. This affects `Char10.emd` only and is
off by default. It runs after the normal EMD/TMD load, averages normals at
matching vertex positions within each rigid joint, and subdivides every
loaded triangle into four using curved edge midpoints. UVs and colour are
interpolated; the original joint hierarchy and animation data are retained.
Set `RE1_CHRIS_MESH=2` for the revised pass: it also shifts source vertices
slightly toward the average of their neighbours, with more rounding on the
head and upper sleeves. Both modes produce the same number of triangles.

```sh
RE1_CHRIS_MESH=2 RE1_MODEL_VIEW=chris RE1_MODEL_VIEW_SCENE=main-hall \
  wine build/windows/residentevil.exe
```

In the rest-pose viewer, 15 of Chris's 16 TMD objects are drawn: 677 source
polygons become 2,708 curved triangles. This visibly rounds the forearms and
lower legs. The shoulder sleeve, face, and rigid-joint seams still need
specific shape work; polygon subdivision alone cannot create a new sculpt.
The player has no movement animation in the viewer, so inspect running,
turning, and weapon poses in gameplay before enabling this generally.

## Chris HD face texture study

The installed Latino/HD pack already replaces both `Char10.emd` palette pages
with 1024x1024 WebPs. Enable that pack through `[Assets] Mods` (see
`docs/MOD_OVERRIDES.md`) to see its real face detail in the viewer. The local
`mods/chris-face/` override adds a restrained skin texture pass to just the
face region of those two pages. Put it before `mods/latino` in the Mods list:

```ini
Mods=../../mods/chris-face;../../mods/latino
```

For an enlarged camera 4 comparison, use `RE1_MODEL_VIEW_X=19500`,
`RE1_MODEL_VIEW_Y=450`, and `RE1_MODEL_VIEW_Z=25100` along with the viewer
variables above. The positive Y offset is only for framing the face; it lifts
Chris from the gallery floor. The close-up shows the added pores and complexion
variation, but the head silhouette and vertex-lit shading remain those of the
refined TMD. The side-by-side capture is
`build/model-viewer/chris-face-detail-comparison.png` (HD pack on the left,
face study on the right). Test gameplay animations before using the override
outside the viewer.

## Entity mesh refinement

For the authored model upgrades, set `RE1_HD_MODELS=1` and point the game at
the installed replacement directory. The switch is off by default; setting
`RE1_HD_MODELS=0` explicitly disables subdivision, patches, and replacements.
From the repository root:

```sh
RE1_HD_MODELS=1 \
RE1_HD_MODELS_DIR="$PWD/tools/data/entity_mesh_replacements" \
RE1_HD_MODELS_PATCH_DIR="$PWD/tools/data/entity_mesh_patches" \
wine path/to/residentevil.exe
```

`RE1_HD_MODELS=1` selects the 16-triangle refinement used by the installed
meshes. The directory variables are required to load those authored files;
they are not inferred from the working directory. The older
`RE1_ENTITY_MESH`, `RE1_ENTITY_MESH_REPLACE_DIR`, and
`RE1_ENTITY_MESH_PATCH_DIR` names remain available for export scripts and
earlier setups. The HD names take precedence when both forms are set.

Set `RE1_ENTITY_MESH=2` to apply the same render-only curved subdivision and
moderate vertex rounding to every character and enemy EMD loaded through
`LoadEntityEMD`. This includes the player, NPCs, alternate costumes, and
creatures in both scenarios. `RE1_ENTITY_MESH=1` subdivides without moving
source vertices. `RE1_ENTITY_MESH=3` adds a second curved subdivision for
16 triangles per original triangle; the extra density costs memory and render
time and makes only a modest difference in the main-hall Jill close-up.
The option is off by default. `RE1_CHRIS_MESH=1` or `2` still
overrides the setting for `Char10.emd`, preserving the Chris-specific head and
sleeve tuning. Animation frames, rigid-joint transforms, source EMD files, and
UV layout are unchanged.
The smoothing now holds each rigid part's open rim in place. This prevents
the subdivision pass from pulling sleeve, neck, and limb edges farther from
their neighbouring parts; it does not turn those joints into a skinned mesh.
At Chris animation call 8, matched 1280x960 captures before and after this
change are `build/model-viewer/entity-mesh/chris-seam-before.png` and
`chris-seam-after.png`. The difference affects about 2,000 pixels and is
visually small; the visible shoulder/arm separation remains. Jill and Hunter
also completed 30-frame 16x viewer runs with the held-rim pass.

`RE1_ENTITY_GEOMETRY_AUDIT=1` logs each loaded part's bounds and its open
boundary loops. Chris has just three open loops across his 15 drawn parts;
the conspicuous arm and shoulder joins are mostly between closed rigid pieces.
Two opt-in geometry studies are available for further comparison:
`RE1_ENTITY_CREASE_GUARD=1` keeps normals across folds sharper than 60 degrees,
and `RE1_ENTITY_JOINT_OVERLAP=1..3` lengthens Y-oriented limb ends by 2–6%.
The 4% overlap completed 30-frame viewer runs for Jill and Hunter. Chris's
matched pose captures `chris-overlap-before.png` and
`chris-overlap-4percent.png` show that the shoulder separation still remains;
the study is not enabled by `RE1_ENTITY_MESH` alone. Jill's crease comparison
is in `jill-crease-before.png` and `jill-crease-guard.png`.

```sh
RE1_ENTITY_MESH=2 RE1_MODEL_VIEW=wesker RE1_MODEL_VIEW_SCENE=main-hall \
  wine build/windows/residentevil.exe
```

This workspace also has the verified isolated Win32 build at
`build/model-viewer/entity-mesh-runtime/residentevil.exe`, with its own
`config.ini` pointing to the local assets and HD mod. It avoids the shared
`build/windows-cross` executable used by the room reconstruction work.

The pass creates four triangles from each source triangle. It cannot sculpt
new anatomy or add detail missing from an EMD texture. Check representative
human and creature models in the viewer, then check moving joints, weapon
poses, and multiple enemies in gameplay before enabling it by default.

An automated Wine viewer sweep of the 67 distinct USA EMD paths returned for
all 67 with `RE1_ENTITY_MESH=2`. Sixty-five had drawable TMD geometry and
produced 158,164 refined triangles from 39,541 source polygons in aggregate.
`em100e.emd` and `em110e.emd` (Plant 42 roots) reported no drawable TMD parts
through this viewer path. The per-file counts are in
`build/model-viewer/entity-mesh/audit.json`. Main-hall camera 4 captures of
Jill, Wesker, Hunter and Tyrant are in the same directory. These checks show
successful loading and rendering; gameplay animation and crowded rooms still
need playtesting. `jill-original.png`, `jill.png` (4x), and `jill-16x.png`
are matched camera and lighting captures for comparing the silhouette.

## Editable joint meshes and Jill face pilot

The render mesh can now be exported as per-joint OBJ files with a rig manifest.
`tools/export_entity_objs.py` batch-exports all 67 distinct USA EMD paths;
`tools/assemble_entity_obj.py` imports one manifest into Blender with the
original rest translations. Each OBJ records its source TMD offset, vertex
counts, and a hash. The corrected source export in
`build/model-viewer/entity-mesh/obj-batch-source-corrected/verification.json`
confirms the loaded EMD path for all 67 manifests: 963 drawable parts and
39,562 source faces. The earlier `obj-batch-source/verification.json` did not
check the loaded path. Its viewer selected `em1031.emd` for the `em1030.emd`
entry, understating that model by 21 faces; the earlier 39,541 total is not a
valid distinct-file total. The two Plant 42 root EMDs have no drawable TMD in
this viewer path.
The corresponding 16x refined OBJ export is in
`build/model-viewer/entity-mesh/obj-batch-refined-corrected/verification.json`:
all 67 source paths match, with 65 drawable models, 963 parts, and 632,992
triangles. `em1028.emd` needed one isolated retry after Wine briefly failed
to create a window during the batch; both logs are retained there. Run the
replacement verifier after this export:

```sh
python3 tools/verify_entity_mesh_replacements.py \
  --refined-dir build/model-viewer/entity-mesh/obj-batch-refined-corrected
```

It checks every installed `.r1mesh` against its exported source hash, TMD
offset, elements, vertices, and indices. All 701 installed replacements pass.
`tools/report_entity_mesh_coverage.py` writes
`build/model-viewer/entity-mesh/coverage.json`: all 65 drawable EMDs
currently have an authored replacement, but only 701 of their 963 drawable
parts have a model-specific sculpt. The generic 16x curved
subdivision can still run on all drawable EMDs; this count tracks the separate
model-specific shape and topology work. EMD coverage alone is not evidence of
remake-quality models.

For an edited mesh, `tools/export_entity_delta.py` writes an `.r1delta` patch
of positions and normals. The loader checks its source hash and element counts
before applying it. Vertex order/count, triangle topology, UVs, texture pages,
and the original joint animation remain unchanged. Set
`RE1_ENTITY_MESH_PATCH_DIR` to the directory holding patches; the variable is
off by default. The current Jill face pilot is
`tools/data/entity_mesh_patches/char11_000035ac.r1delta` and targets
`RE1_ENTITY_MESH=3` (16 triangles per source triangle):

```sh
RE1_ENTITY_MESH=3 \
RE1_ENTITY_MESH_PATCH_DIR="$PWD/tools/data/entity_mesh_patches" \
RE1_MODEL_VIEW=jill RE1_MODEL_VIEW_SCENE=main-hall \
RE1_MODEL_VIEW_CAMERA=4 RE1_MODEL_VIEW_PLAYER_ANIM=0 \
RE1_MODEL_VIEW_X=19500 RE1_MODEL_VIEW_Y=450 RE1_MODEL_VIEW_Z=25100 \
wine build/model-viewer/entity-mesh-runtime/residentevil.exe
```

The patch was made by running `tools/sculpt_jill_face.py` on the assembled
refined Jill head, then `tools/export_entity_delta.py --only-joint 1`. It
adjusts the eye sockets, cheek arc, bridge, lip, chin, and lower jaw while
leaving her cap, hair, and neck essentially in place. The matched game capture
is `build/model-viewer/entity-mesh/obj-jill-refined/jill-head-comparison.png`
(unpatched left, sculpted right). The viewer loaded the patch and completed a
60-frame animated run under Wine. This does not verify every gameplay pose or
occlusion against other characters and props; those need in-game playtesting.

The white-coat zombie (`em1000.emd`) now has an experimental head patch in the
same directory: `em1000_00016f2c.r1delta`. `tools/sculpt_zombie_head.py` pulls
the cheeks and eye sockets inward, adds a modest brow, and narrows the jaw.
Its normals were recalculated after the shape edit. The matched side-angle
comparison is
`build/model-viewer/entity-mesh/obj-zombie-refined/zombie-side-comparison-strong.png`
(original left, patch right). The effect is subtle under this dark room light;
it does not add wound geometry. A 60-frame animated viewer run loaded all
7,296 patched head vertices and exited successfully. Gameplay poses still need
checking.

### Replacement meshes with new topology

For geometry that needs new vertices, `tools/export_entity_mesh.py` writes a
`.r1mesh` replacement from an assembled Blender joint. Set
`RE1_ENTITY_MESH_REPLACE_DIR` to its directory. The game checks the original
refined TMD hash, the element count, every vertex and index, then allocates all
replacement buffers before swapping any part. The TMD element keeps its
existing texture/material selection and the rigid animation joint stays the
same. A successful replacement takes precedence over an `.r1delta` for that
part; an absent or rejected replacement can fall back to the delta.

The first topology-changing example is the white-coat zombie head at
`tools/data/entity_mesh_replacements/em1000_00016f2c.r1mesh`. It was assembled
from the 16x refined OBJ, subdivided with `tools/subdivide_entity_joint.py`,
sculpted with `tools/sculpt_zombie_head.py`, and exported while preserving the
interpolated source normals. The viewport grew from 7,296 vertices / 2,432
triangles to 14,592 vertices / 9,728 triangles. Its matched camera 4 crop is
`build/model-viewer/entity-mesh/obj-zombie-refined/zombie-hd-replacement-comparison.png`
(original left, replacement right). The textured difference is still modest;
extra topology alone does not create an authored remake model.

```sh
RE1_ENTITY_MESH=3 \
RE1_ENTITY_MESH_REPLACE_DIR="$PWD/tools/data/entity_mesh_replacements" \
RE1_MODEL_VIEW=zombie RE1_MODEL_VIEW_SCENE=main-hall \
RE1_MODEL_VIEW_CAMERA=4 \
wine build/model-viewer/entity-mesh-runtime/residentevil.exe
```

The replacement completed a 60-frame animated Wine viewer run. A deliberately
wrong source hash was rejected while the original model remained drawable.
The Jill-scenario white-coat zombie (`em1100.emd`) has the same refined head
geometry and source hash, so the replacement is also installed as
`em1100_00016f2c.r1mesh`. `RE1_MODEL_VIEW=j:4` loaded it and completed its own
60-frame viewer run.
The green-coat and naked zombie heads also have separate replacements for
both scenario blocks. Green (`em1011` / `em1111`) grows from 6,816 vertices /
2,272 triangles to 13,632 / 9,088; naked (`em1001` / `em1101`) grows from
3,648 / 1,216 to 7,296 / 4,864. The green zombie uses the gaunt shape pass;
the naked zombie's nearly featureless source head gets a brow, nose, mouth
recess, and jaw shaping from `tools/sculpt_naked_zombie_face.py`. Their Jill
variant game comparisons are
`build/model-viewer/entity-mesh/obj-green-zombie-refined/green-zombie-head-comparison.png`
and `build/model-viewer/entity-mesh/obj-naked-zombie-refined/naked-zombie-head-comparison.png`.
The paired Chris/Jill refined heads are byte-identical within each enemy type,
including the source hash, so the same authored viewport data can be installed
under each EMD filename. Both Chris variants completed 60-frame animated viewer
runs; both Jill variants loaded from the revised viewer aliases.
The same path was checked with Jill's two-element head, then used to turn her
face sculpt into `char11_000035ac.r1mesh` in that replacement directory:
8,064 vertices / 2,688 triangles became 16,128 vertices / 10,752 triangles.
The comparison is
`build/model-viewer/entity-mesh/obj-jill-refined/jill-hd-replacement-comparison.png`;
the final file loaded and completed a 60-frame animated viewer run. For both
models, interpolated source normals were retained after subdivision; simply
recalculating normals on the source's disconnected triangles produces visible
triangular shading. An artist joining and remodeling those faces can export
their own smooth normals instead.
The Hunter (`em1006.emd`) also has paired hand replacements in this directory:
`em1006_000167d8.r1mesh` and `em1006_0001682c.r1mesh`. Each hand grew from
3,744 vertices / 1,248 triangles to 7,488 vertices / 4,992 triangles.
`tools/sculpt_hunter_claws.py` lengthens and tapers only the distal claws,
leaving the wrist vertices in place. The matched main-hall capture is
`build/model-viewer/entity-mesh/obj-hunter-refined/hunter-hd-hands-comparison.png`
(original left, new claws right). Both replacements loaded and completed a
60-frame animated viewer run. The pose shown is a held animation frame; attack
contacts and gameplay frame rate still need checking.
The Jill-scenario Hunter (`em1106.emd`) has byte-identical refined hand meshes
and source hashes at both TMD offsets, so the same replacements are installed
under its two filenames as well. `RE1_MODEL_VIEW=j:10` loaded both and finished
a separate 60-frame animated viewer run.
Barry and Wesker now have separate head replacements, sculpted with
`tools/sculpt_barry_face.py` and `tools/sculpt_wesker_face.py`. Barry's
`em1022_0001b2e8.r1mesh` grows from 8,016 vertices / 2,672 triangles to
16,032 / 10,688. Wesker's `em1024_0001b62c.r1mesh` grows from 8,304 /
2,768 to 16,608 / 11,072; his sculpt reduces the lens bulge and adjusts the
cheeks and jaw. Matched textured main-hall camera 4 head comparisons are
`build/model-viewer/entity-mesh/obj-barry-refined/barry-head-game-comparison.png`
and `build/model-viewer/entity-mesh/obj-wesker-refined/wesker-head-game-comparison.png`.
Both loaded through the normal EMD path and completed separate 60-frame viewer
runs. Their silhouette changes remain subtle at gameplay distance; these are
incremental sculpts, not finished remake-quality heads.
Rebecca's `em1023_00018cd4.r1mesh` adds a restrained cheek, nose, and jaw
sculpt from `tools/sculpt_rebecca_face.py`, then grows from 8,304 vertices /
2,768 triangles to 16,608 / 11,072. The matched main-hall close-up is
`build/model-viewer/entity-mesh/obj-rebecca-refined/rebecca-head-game-comparison.png`.
It loaded through `em1023.emd` and completed a 60-frame viewer run. Its shape
change is also subtle at gameplay distance. `tools/sculpt_normal_update.py`
transforms her exported smooth normals through the sculpt's local deformation;
recalculating normals on disconnected TMD triangles creates facets. The
matched lighting comparison is
`build/model-viewer/entity-mesh/obj-rebecca-refined/rebecca-normal-game-comparison.png`.
The normal change is barely visible under this main-hall light, but the revised
file loaded and completed another 60-frame run.
`tools/sculpt_human_shoulders_obj.py` rounds the outer torso shoulder contour
in the verified OBJ's Z (width) and Y (height) axes, adds four triangles per
refined triangle, and carries the original smooth normals through the
deformation. The rig's left/right arm offsets establish Z as width. Blender
imports viewport (X,Y,Z) as (X,-Z,Y), so its Y and Z are width and height.
The earlier torso files used mismatched axes; all 21 are regenerated with the
rig-confirmed mapping. `tools/sculpt_human_shoulders.py` uses it as well.
Nine additional torso replacements now cover the Chris NPC, Kenneth, Forest,
Richard, Enrico, devoured Kenneth, dying Barry, and both alternate Chris
outfits. Jill's two alternate outfit EMDs (`em1031`, `em1033`) now have the same
rig-confirmed shoulder sculpt. This makes 23 human torso replacements in total.
All 23 passed the
static verifier and separate 60-frame Wine viewer runs with the matching
replacement loaded. The corrected Chris front and side comparisons are in
`build/model-viewer/entity-mesh/shoulders-rig-axis-candidates/`; Enrico and
Kenneth were also inspected in the game camera. The Jill alternate costume
comparison is in
`build/model-viewer/entity-mesh/jill-alternate-shoulders-candidates/jill-alt-comparison.png`.
The silhouette improvement is
modest. Gameplay weapon, damage, and corpse poses still require checking.
`tools/propagate_entity_mesh_replacements.py` compares the complete refined
OBJ content (apart from its source-path comment) and source hash across all
963 drawable parts. It found 12 identical-geometry targets for existing head
and torso replacements. Those copies are installed under the matching EMD
names, including `em1021`, `em102b`, `em102c`, `em102d`, and `em102e`.
All 12 passed the static verifier, and seven affected viewer selectors
(`2`, `3`, `37`, `47`, `48`, `49`, `50`) completed separate 60-frame Wine runs.
The matched camera 4 capture for `em102e` is
`build/model-viewer/entity-mesh/propagated-candidates/em102e-game-comparison.png`.
The copies preserve the original rigid joints; gameplay animation and contact
poses still need checking.
`tools/sculpt_tyrant_face_obj.py` adds restrained brow, cheek, nose bridge and
chin relief to Tyrant's separate head joint (joint 2), then adds four triangles
per refined triangle. The four matching EMDs (`em100c`, `em1010`, `em110c`,
`em1110`) have identical head source geometry and each receives its own
source-hash-checked replacement. All four passed 60-frame Wine viewer runs.
The matched close camera 4 comparison is
`build/model-viewer/entity-mesh/tyrant-face-candidates/tyrant-face-comparison.png`.
The facial change is subtle at normal game distance; attacks and head movement
in gameplay remain untested.
Tyrant has no separate neck joint: head joint 2 attaches directly to torso
joint 1. A camera 4 close-up shows the lower jaw nearly touching the ribbed
collar (`build/model-viewer/entity-mesh/tyrant-torso-candidates/tyrant-neck-before.png`).
The `sculpt_tyrant_neck_obj.py` trial kept the lower head attachment fixed and
lifted the face 95 viewport units; the matched comparison is
`build/model-viewer/entity-mesh/tyrant-neck-candidates/tyrant-neck-face-comparison.png`.
It lengthened the face silhouette but exposed no convincing neck, so it was
not installed. The existing head's lowest vertices form a narrow forward
strip rather than a complete neck cylinder; a proper fix needs new neck
geometry and suitable UVs, then head-turn and attack-pose inspection. A
`sculpt_tyrant_torso_obj.py` chest-relief trial also made too little visible
difference to install.
`tools/build_tyrant_neck_mesh.py` tests a new head-rigged neck surface while
retaining the installed facial sculpt. It passes the static replacement
verifier, and the same-engine viewer confirms that the extra triangles load.
At a plausible position the geometry is occluded by the jaw and collar
(`build/model-viewer/entity-mesh/tyrant-neck-geometry-candidates/tyrant-neck-forward-comparison.png`).
A deliberately far-forward diagnostic position exposed the grey surface in
front of the mouth (`tyrant-neck-debug-far.png` in the same directory). Thus
the missing-looking neck is mainly an overlap/silhouette problem; simply
adding a tube to the head is not an acceptable fix. Neither variant was
installed.
The later `--placement slim --lift-head 80` version raises the existing head
geometry while adding a narrower head-rigged neck behind the jaw. The first
low-Y forward trial looked like a flat grey chin block; a high-Y trial
produced a cylinder above the skull. Those remain uninstalled. The slim
version is installed for all four Tyrant EMDs. Matched front captures are
`build/model-viewer/entity-mesh/tyrant-neck-slim-candidates/neck-before.png`
and `neck-after.png`; `neck-oblique-after.png` and `neck-pose40.png` check
another angle and animation frame. All four selectors loaded their replacement
head and completed 120 unfrozen viewer frames; the installed set passes the
629-file static verifier. Attack poses and the raised head's gameplay contact
still need checking in game.
An additional `--placement slim --lift-head 120` trial from the pre-neck
face mesh was captured at
`build/model-viewer/entity-mesh/tyrant-neck-lift120-candidates/front.png`.
It exposes a flat grey patch under the chin and still leaves the collar
crowding the neck, so it was not installed. A useful next Tyrant pass must
reshape the torso collar and blend the neck into the jaw while checking
head turns and attack poses.
`tools/sculpt_tyrant_collar_obj.py` then tested a central collar drop of
120 and 65 viewport units while keeping the shoulder edges fixed. The first
viewer capture was invalid because the combined candidate directory omitted
the previously uncovered torso file; its image was pixel-identical to the
installed model. After the torso file was linked, the 120-unit version
(`build/model-viewer/entity-mesh/tyrant-collar-candidates/front-loaded.png`)
exposed a straight pale neck tube. The shallower 65-unit cut and a tapered
head-rigged neck trial
(`build/model-viewer/entity-mesh/tyrant-neck-tapered-candidates/front.png`)
still showed a rectangular chin-to-collar gap. Their 705-file candidate set
passed the static verifier, and the game log confirms the torso replacement
loaded. Neither is installed. A believable neck needs the jaw, neck surface,
and collar silhouette sculpted together; another simple head lift or collar
drop will not suffice.
The later `--placement bridge` trial moved the head-rigged tube forward under
the flat jaw rim and darkened its vertex colours to suggest neck muscles.
With the original collar it stayed almost completely hidden; combined with
the 65-unit collar cut it still appeared as a separate grey block. Captures
are in `build/model-viewer/entity-mesh/tyrant-neck-bridge-candidates/`.
It is not installed; changing the head's lower topology is necessary to
make the transition continuous.
`tools/sculpt_tyrant_forearm_obj.py` rounds and tapers Tyrant's normal forearm
(joint 4) while leaving both joint attachment rings fixed. Four replacements
cover `em100c`, `em1010`, `em110c`, and `em1110`. The matched camera 4 crop is
`build/model-viewer/entity-mesh/tyrant-forearm-candidates/tyrant-forearm-close-comparison.png`.
All four passed source verification and separate 60-frame Wine viewer runs.
The improvement is modest at game distance; contact, attack, and death poses
still require gameplay inspection.
`tools/sculpt_tyrant_calves_obj.py` tested additional rear calf and shin relief
on lower-leg joints 10 and 13. Even at `--strength 1.5`, the matched camera 4
comparison (`build/model-viewer/entity-mesh/tyrant-calf-candidates/tyrant-calves-strong-comparison.png`)
showed too little improvement to install.
`tools/sculpt_spider_abdomen_obj.py` rounds the abdomen (joint 19) of Web
Spinner and Black Tiger in both scenarios, leaving its attachment and all
animated leg joints in place. Four source-hash-checked replacements passed
separate 60-frame Wine viewer runs. The stronger inspected version uses
`--strength 1.8`; its rear camera comparison is
`build/model-viewer/entity-mesh/spider-abdomen-strong-candidates/spider-back-comparison.png`.
Its front view changes less because the legs hide much of the abdomen. Spider
walk and attack poses still require gameplay inspection.
`tools/sculpt_spider_legs_obj.py` tapers the eight distal leg segments of Web
Spinner and Black Tiger in both scenarios. It narrows only the outer half of
each segment around its measured tip centre; the hinges and leg-tip distance
stay fixed. All 32 source-hash-checked replacements passed verification and
each of the four EMDs completed a 60-frame animated Wine viewer run with its
eight leg parts and abdomen replacement loaded. Matched main-hall camera 4
captures are `build/model-viewer/entity-mesh/spider-legs-before-centered.png`
and `build/model-viewer/entity-mesh/spider-leg-candidates/spider-after-centered.png`.
The taper removes the blunt tube ends in this view. Ground contact and attack
poses still need a gameplay check.
A stronger distal taper trial (1.4 and 1.8 strength) in
`build/model-viewer/entity-mesh/spider-sharp-tip-candidates/` made the visible
end cap jagged rather than smoothly pointed in the frozen engine view. The
installed 1.0 taper remains the better version.
`tools/add_spider_tarsal_claws.py` builds on that taper with a paired pointed
tarsal claw emerging from each of the eight distal leg tips. The 32 updated
meshes cover Web Spinner and Black Tiger in both scenarios; their original
installed versions are backed up under
`build/model-viewer/entity-mesh/spider-claw-candidates/baseline-distal/`.
Use that backup as `--installed-dir` when regenerating; the tool rejects an
already augmented leg so the claws cannot be appended twice.
The frozen main-hall candidate captures in that directory show a distinct
claw silhouette and a matching dark tip colour. All four EMDs loaded their
eight updated leg parts and completed 120 unfrozen viewer frames. Walking,
floor contact, and attack poses still need gameplay inspection.
`tools/sculpt_spider_upper_legs_obj.py` narrows the broad inner shaft of each
of those eight legs while keeping the body socket and knee rings fixed. All
32 additional replacements passed source verification. Each spider EMD then
loaded its 16 leg replacements and abdomen replacement during a 60-frame
animated Wine viewer run. The matched camera 4 follow-up is
`build/model-viewer/entity-mesh/spider-upper-leg-candidates/spider-upper-after-centered.png`;
it shows a slimmer inner shaft, although gameplay leg articulation remains
untested.
`tools/sculpt_spider_carapace_obj.py` raises and rounds the giant spiders'
central cephalothorax (joint 0) while leaving the abdomen seam, mouth end,
and animated leg sockets fixed. All four Web Spinner / Black Tiger variants
have this replacement installed. Matched close camera 4 images are
`build/model-viewer/entity-mesh/spider-carapace-candidates/before-framed.png`
and `after-framed.png`; the fuller crown is most visible from the front.
The oblique comparison in the same directory shows the rear join unchanged.
All four selectors completed 120 unfrozen viewer frames with 18 authored
parts loaded, and all 633 installed meshes passed static verification.
Attack and floor-contact poses still need gameplay inspection.
`tools/sculpt_chimera_head_obj.py` adds shape to Chimera's separate brow and
mandibles (joint 3) in both scenarios. The original UVs and rigid animation
remain. Both source-hash-checked meshes passed 60-frame Wine viewer runs; the
matched game-camera comparison is
`build/model-viewer/entity-mesh/chimera-head-candidates/chimera-head-comparison.png`.
The change is subtle under the main-hall lighting. Attack poses still require
gameplay inspection.
`tools/sculpt_chimera_upper_arms_obj.py` rounds the middle of upper-arm joints
4 and 7 for both scenario Chimeras. The ends remain fixed at the shoulder and
elbow. Matched front and oblique viewer comparisons are in
`build/model-viewer/entity-mesh/chimera-upper-arm-candidates/`; the front
silhouette and side volume are visibly fuller. All four files passed
source-hash/topology verification, and both EMDs completed 60-frame Wine
viewer runs. Climbing and attack animation still need gameplay inspection.
`tools/sculpt_chimera_forearms_obj.py` adds restrained muscle volume to joint
5 and 8 forearms on both scenario EMDs, leaving the elbow and wrist ends
unchanged. Matched front and oblique comparisons are in
`build/model-viewer/entity-mesh/chimera-forearm-candidates/`. All four files
passed source-hash/topology verification and both EMDs completed 60-frame
Wine viewer runs. Claw contact and attack motion need gameplay inspection.
`tools/sculpt_chimera_lower_legs_obj.py` adds calf contour to joints 12 and
15 and narrows/extends the forward foot tips on joints 13 and 16 for both
scenario Chimeras. Knees, ankles, and foot Y coordinates remain fixed. The
matched side captures are
`build/model-viewer/entity-mesh/chimera-lower-leg-candidates/before-side.png`
and `after-side.png`; a frame-40 pose in the same directory shows no new open
ankle join. All eight replacements are installed, the 701-file set passes
static verification, and both models completed 120 unfrozen viewer frames.
Climbing and floor contact in gameplay still need checking.
A paired Chimera quadriceps relief trial in
`build/model-viewer/entity-mesh/chimera-thigh-candidates/` remained visually
indistinguishable in matched frozen main-hall frames, even at 1.5 strength.
Neither strength variant was installed.
`tools/sculpt_crow_beak_obj.py` hooks and tapers the crow's separate beak;
`tools/sculpt_crow_wingtips_obj.py` extends and droops its two outer wing
feather joints while preserving their hinges. Both crow EMDs have all three
parts installed. The matched game-camera comparison is
`build/model-viewer/entity-mesh/crow-wingtip-candidates/crow-features-comparison.png`.
Both models loaded all three parts and completed 60-frame Wine viewer runs.
The viewer pose is airborne; flight and attack animations in gameplay remain
to be checked.
`tools/sculpt_wasp_abdomen_obj.py` rounds the wasp's striped abdomen at its
middle while holding the attachment and stinger end in place. Both scenario
EMDs passed 60-frame Wine viewer runs with the new part loaded. The matched
main-hall close-up is
`build/model-viewer/entity-mesh/wasp-abdomen-candidates/wasp-abdomen-comparison.png`.
Flying and sting contact poses still require gameplay inspection.
A four-wing membrane bow trial in
`build/model-viewer/entity-mesh/wasp-wing-candidates/` passed source
verification but its matched frozen studio captures showed almost no visible
change. A stronger downward tip bend folded the wings over the body in the
viewer. Neither trial is installed.
`tools/sculpt_neptune_snout_obj.py` tapers and slightly lengthens Neptune's
upper head and animated lower jaw in their shared rest-space X, using the
jaw's +165 offset to keep their tips aligned. Both scenario EMDs have both
parts installed and passed 60-frame Wine viewer runs. Matched side and front
comparisons are in `build/model-viewer/entity-mesh/neptune-snout-candidates/`.
The effect is modest at normal camera distance; bite contact in gameplay still
needs testing.
`tools/sculpt_neptune_dorsal_fin_obj.py`,
`tools/sculpt_neptune_tail_fin_obj.py`, and
`tools/sculpt_neptune_pectoral_fins_obj.py` add small sweeps and longer tips to
the dorsal, tail, and paired pectoral fins while retaining their joint bases.
All four fin parts are installed for both scenario EMDs. The eight files pass
source and topology verification; each scenario loaded all four fins and
completed a 60-frame Wine viewer run. Side and front comparisons are in
`build/model-viewer/entity-mesh/neptune-fin-candidates/` and
`build/model-viewer/entity-mesh/neptune-pectoral-candidates/`. Aquarium
swimming and attack contact still need gameplay inspection.
`tools/sculpt_neptune_tail_body_obj.py` reduces the repeated midsection
bulges in Neptune's four rear body parts (joints 8–11), leaving the rigid
section ends nearly fixed. Both scenario EMDs now have all eight parts
installed. Their source hashes and topology pass verification, and both
models loaded their four new sections in separate 60-frame viewer runs.
The matched side comparison is
`build/model-viewer/entity-mesh/neptune-tail-body-candidates/neptune-tail-comparison.png`.
Both EMDs also completed 180-frame viewer runs with animation unfrozen.
The side captures held at animation ticks 40 and 80 are in the same candidate
directory; neither shows a new exposed gap. The improvement is modest;
texture seams and actual aquarium gameplay still need work.
`tools/sculpt_neptune_front_body_obj.py` applies a smaller version of the
same taper to front body joints 2 and 5 in both scenario EMDs. Four new
replacements passed source verification and both variants completed 180
unfrozen viewer frames with the new parts loaded. The side rest-pose and
animation-tick-80 comparisons are in
`build/model-viewer/entity-mesh/neptune-front-body-candidates/`. The underside
transition is smoother, but the large texture rings remain; actual swimming
and bite animations still need gameplay inspection.
`tools/sculpt_neptune_mid_body_obj.py` applies a similar taper to the central
body part (joint 7) in both scenario EMDs. The 1.5-strength candidate reduces
the middle bulge while leaving the joint ends unchanged. Its frozen side
comparison is in `build/model-viewer/entity-mesh/neptune-mid-body-candidates/`.
Both EMDs loaded the part and completed 180 unfrozen viewer frames, and all
13 drawable Neptune parts now have authored replacements. A later pose showed
no new exposed gap, though its baseline and candidate captures did not freeze
at identical animation phases. Aquarium gameplay remains untested.
`tools/smooth_neptune_body_normals_obj.py` tested an alternative normal-only
seam pass (`build/model-viewer/entity-mesh/neptune-body-normal-candidates/neptune-normal-comparison.png`).
It left the segment rings essentially unchanged, so it was not installed.
The trial paired Chimera chest relief in
`build/model-viewer/entity-mesh/chimera-chest-candidates/` was not installed.
Its matched camera comparison shows too little visible contour change to
justify an extra replacement, despite changing the geometry. The existing
texture accounts for most of the chest detail in that view.
`tools/sculpt_adder_snout_obj.py` tapers and slightly lengthens the adder's
upper head and animated lower jaw in shared rest-space, accounting for the
jaw's -25 X offset. Both scenario EMDs have both parts installed. The matched
camera comparison is
`build/model-viewer/entity-mesh/adder-snout-candidates/adder-comparison.png`;
the change is very subtle at game distance. Both models completed viewer runs,
but bite contact and body motion still require gameplay inspection.
`tools/sculpt_adder_body_obj.py` is an uninstalled trial for articulated
body joints 2–10 in both scenario EMDs. Its 719-file combined candidate set
passes source-hash/topology verification, but the main-hall viewer framing
used so far does not show the whole adder reliably. The body profile and
motion have not yet been visually accepted for installation.
`tools/sculpt_yawn_snout_obj.py` narrows and slightly extends Yawn's upper
head and animated lower jaw in shared rest-space, accounting for the jaw's
+425 X offset. Both encounters in both scenarios have both parts installed;
all four models completed viewer runs. Matched front and side close-ups are in
`build/model-viewer/entity-mesh/yawn-snout-candidates/`. The visual change is
modest, and attack and bite motion still require gameplay inspection.
`tools/sculpt_chris_closeup_arm_obj.py` rounds the mid forearm of the separate
Chris close-up arm rigs while leaving the elbow and wrist attachments fixed.
Both arms in both scenarios have the sculpt installed and completed viewer
runs. The matched comparison is
`build/model-viewer/entity-mesh/chris-closeup-arm-candidates/chris-arm-comparison.png`.
Finger and typing animation still require gameplay inspection.
`tools/sculpt_typing_hand_obj.py` shapes the separate palm and fourteen
articulated finger segments in each of the four typing-hand EMDs. It gives the
middle finger sections mild knuckle volume and rounds terminal tips inward,
keeping the joint ends and fingertip reach fixed. The 60 new replacements are
installed. Matched frozen camera 4 captures using viewer FOV 512 are
`build/model-viewer/entity-mesh/typing-hand-candidates/zoom512-before.png`
and `zoom512-after.png`; the change is subtle at ordinary game distance.
All four rigs loaded 16 authored parts and completed 120 unfrozen viewer
frames. The installed 693-file set passed static verification. Keyboard
contact in the actual cutscene still needs inspection.
`tools/sculpt_spider_web_obj.py` gives each of the six spider-web panels a
shallow depth curve while leaving its outer boundary in place. Both scenario
EMDs have all six panels installed and completed viewer runs. The oblique
comparison is
`build/model-viewer/entity-mesh/spider-web-candidates/spiderweb-oblique-comparison.png`.
This gives the overlapping panels some depth; in-room placement and web
interaction still need gameplay inspection.
`tools/sculpt_plant_stalk_obj.py` adds shallow organic fullness to the
segmented Plant 42 and hanging monster-plant stalks while holding each segment's
end rings fixed. The free end of Plant 42 also has a mild taper. All four
scenario EMDs loaded every part and completed viewer runs. Matched comparisons
are in `build/model-viewer/entity-mesh/plant-stalk-candidates/`. These changes
are subtle in the available camera views; actual plant attack animation and
room placement still require gameplay inspection.
`tools/sculpt_enrico_face_obj.py` adds restrained brow, nose, cheek, and jaw
relief to Enrico's head. The matched front and three-quarter game-camera
comparisons are in `build/model-viewer/entity-mesh/enrico-face-candidates/`;
the profile change is visible without a new neck seam. The replacement loaded
and completed a 60-frame viewer run.
`tools/sculpt_richard_face_obj.py` adds small brow, nose, cheek, lip, and jaw
relief to Richard's separate head. A matched front comparison in
`build/model-viewer/entity-mesh/richard-face-candidates/` uses the EMD's rest
pose so the face can be seen; its normal lying pose was also inspected. The
replacement passed source-hash/topology verification, loaded, and completed a
60-frame viewer run. Cutscene expressions still need gameplay inspection.
`tools/sculpt_barry_dying_face_obj.py` carries Barry's existing face sculpt to
his dying cutscene EMD, whose head has identical positions and normals but a
different UV layout. The matched comparison is in
`build/model-viewer/entity-mesh/barry-dying-face-candidates/`. Its own source
hash and UVs are preserved; verification and a 60-frame viewer run passed.
Lower-face trials for Kenneth's two corpse heads and Forest are in
`build/model-viewer/entity-mesh/corpse-face-candidates/`. The matched camera
captures showed no useful change on ordinary Kenneth and only very small
changes on devoured Kenneth and Forest. Their wound textures and existing head
silhouettes dominate at play distance, so these three trial files were not
installed. A stronger Forest test stretched the wound and mouth texture
without improving the overall face shape.
`tools/sculpt_chris_face_obj.py` adds similarly restrained anatomy to Chris's
player, room NPC, and two alternate-outfit heads. All four EMDs loaded the
replacement and completed 60-frame viewer runs. Matched front and three-quarter
comparisons are in `build/model-viewer/entity-mesh/chris-face-sculpt-candidates/`.
The HD texture remains the main source of fine facial detail.
`tools/sculpt_jill_alternate_face_obj.py` brings the two alternate-outfit
heads closer to Jill's regular face sculpt. Both EMDs completed 60-frame viewer
runs. The matched camera comparison is
`build/model-viewer/entity-mesh/jill-alt-face-candidates/jill-alt-face-comparison.png`.
These facial sculpts still need gameplay animation and expression checks.
`tools/sculpt_human_toes_obj.py` rounds the distal toe columns of the separate
foot joints (5 and 8), leaving the heel and ankle in place. Forty-four foot
replacements cover 22 human EMDs. An earlier rig reading incorrectly called
these hands; a viewer displacement probe of joint 4 (the lower leg) and joint
11 (the hand) corrected the mapping. Kenneth's devoured-state EMD has no
usable foot geometry and is excluded. Matched Chris and Jill camera crops are
in `build/model-viewer/entity-mesh/human-fingertip-candidates/`; the change is
small at normal game distance. Every affected EMD completed a 60-frame viewer
run with both feet loaded. Forest's first process exited with signal 9 after
loading, then a separate identical run completed normally. Gameplay walking
and floor contact still need inspection.
`tools/sculpt_human_forearms_obj.py` uses the corrected human rig map: joints
10 and 13 are the forearms, with joints 11 and 14 as their hands. It smooths
coincident geometry across UV seams and gives the exposed middle of each arm a
small muscle contour while holding open edges and elbow/wrist ends fixed. The
46 replacements cover 23 human EMDs and passed source-hash/mesh verification.
Matched Chris and Jill camera 4 captures are in
`build/model-viewer/entity-mesh/human-forearm-candidates/`. The silhouette
change is modest at gameplay distance. Chris, Jill, Barry, and the `em1029`
variant completed 60-frame Wine viewer runs with both new parts loaded;
weapon grips and damage poses still need gameplay inspection.
`tools/sculpt_human_upper_arms_obj.py` follows the tilted centrelines of
human upper-arm joints 9 and 12, adding modest deltoid/biceps volume between
unchanged shoulder and elbow ends. Forty-six replacements cover 23 human
EMDs. Matched close camera 4 captures for Chris and Jill, plus inspected
Barry and Rebecca captures, are in
`build/model-viewer/entity-mesh/human-upper-arm-candidates/`. All 46 passed
source-hash verification. Chris and Jill completed separate 60-frame runs
with player animation 0; Barry, Rebecca, and `em1029` completed 60-frame NPC
viewer runs. Raised-arm weapon and damage poses still need gameplay checks.
`tools/sculpt_human_fingers_obj.py` now addresses the actual hand joints (11
and 14). It tapers the distal finger columns and leaves the wrist geometry in
place. Forty-five source-hash-checked replacements cover both hands of 22
human EMDs and one hand of `em1028`; that model's other hand has an unusual
574-unit local Y span and needs its own sculpt. Matched Chris and Jill camera
4 captures are in `build/model-viewer/entity-mesh/human-finger-candidates/`.
The detail is subtle at game-camera distance. Chris, Jill, Barry, `em1028`,
and `em1029` completed 60-frame animated Wine viewer runs with the installed
files. The initial Chris and `em1029` runs timed out under concurrent system
load; their separate retries completed normally. Gameplay weapon grips and
hand contacts remain untested.
The trial whole-leg Laplacian pass in
`build/model-viewer/entity-mesh/human-leg-candidates/` was rejected. The
matched Chris camera 4 captures (`chris-before.png`, `chris-after.png`, and
`chris-after-soft.png`) show that even the weaker pass narrows his trouser
legs and erases cloth folds. No leg replacements from that trial are installed.
`tools/sculpt_human_leg_contours_obj.py` instead adds a small radial contour to
the middle of each thigh and lower leg without averaging neighbouring vertices.
The hip, knee, and ankle ends are unchanged, retaining the existing cloth
folds and joint seams. Ninety-one replacements cover 23 human EMDs. Kenneth's
devoured-state `em1029` has one unusually short thigh, which remains excluded
for a separate pose-specific sculpt. Matched main-hall camera 4 comparisons
for Chris, Jill, Barry, Rebecca, an alternate Chris outfit, and Kenneth are in
`build/model-viewer/entity-mesh/human-leg-contour-candidates/`. Every installed
part passed source-hash/topology checks and all 23 EMDs loaded their new legs
in 60-frame Wine viewer runs. The change is restrained at game-camera
distance; walking, floor contact, and cutscene poses still require gameplay
inspection.
The Cerberus chest/rib relief trial in
`build/model-viewer/entity-mesh/cerberus-body-candidates/` also remains
uninstalled. Matched front and side game-camera captures show negligible
silhouette change; the existing HD skin supplies the visible rib definition.
The Tyrant distal-claw taper/curl trial in
`build/model-viewer/entity-mesh/tyrant-claw-candidates/` is also uninstalled.
The matched main-hall camera 4 captures show that the three already-long
blades change too little at gameplay distance to justify the extra mesh.
`tools/sculpt_hunter_toes_obj.py` tested a longer, narrower Hunter toe tip.
Its matched camera 4 images in
`build/model-viewer/entity-mesh/hunter-toe-candidates/` likewise showed almost
no improvement over the already long talons; the trial was not installed.
`tools/sculpt_tyrant_toes_obj.py` extends and tapers the five large toe tips
on both feet of all four Tyrant EMDs (`em100c`, `em1010`, `em110c`, `em1110`).
The ankles and sole height stay fixed. The stronger candidate's matched
camera comparison is in `build/model-viewer/entity-mesh/tyrant-toe-strong-candidates/`.
All eight replacements passed source-hash/topology verification, and all four
EMDs loaded both feet in 60-frame Wine viewer runs. Walking and floor contact
still need gameplay inspection. A radial upper-arm trial in
`build/model-viewer/entity-mesh/tyrant-upper-arm-candidates/` was not installed
because its corrected game-camera comparison showed little visible benefit.
`tools/sculpt_hunter_snout_obj.py` lengthens and tapers the Hunter's paired
upper muzzle and animated jaw in shared rest-space, accounting for the jaw's
+189 X offset. Both scenario EMDs loaded both parts and completed viewer runs.
Matched front and side camera comparisons are in
`build/model-viewer/entity-mesh/hunter-snout-candidates/`. Bite contact remains
untested in gameplay.
`tools/sculpt_hunter_thighs_obj.py` adds middle-thigh muscle volume to joints
10 and 13 of both scenario Hunters, leaving the hip and crouched knee ends
fixed. The stronger candidate's matched front and side comparisons are in
`build/model-viewer/entity-mesh/hunter-thigh-strong-candidates/`. All four
files passed source-hash/topology verification; both EMDs loaded both thighs
and completed 60-frame Wine viewer runs. Leaping, landing, and attack poses
still need gameplay inspection.
`tools/sculpt_hunter_upper_arms_obj.py` tested a stronger deltoid and narrower
elbow shaft on upper-arm joints 4 and 7. The paired camera 4 capture at
`build/model-viewer/entity-mesh/hunter-upper-arm-candidates/hunter-upper-arm-comparison.png`
showed too little visible change, so neither arm was installed.
A dorsal-ridge trial on Hunter's torso in
`build/model-viewer/entity-mesh/hunter-back-candidates/` also passed source
verification but barely altered its side silhouette in a matched frozen
camera. Neither strength variant was installed.
The single unworked joint 2 on most standing human EMDs is the pelvis/pants.
`tools/sculpt_human_pelvis_obj.py` tested cloth bulges and a recessed fly on
Chris; the close camera 4 comparison is
`build/model-viewer/entity-mesh/human-pelvis-candidates/chris-pelvis-comparison.png`.
It mostly widened the existing baggy pants and was not installed. Repeating
that deformation across characters would increase coverage without improving
their appearance.
`tools/sculpt_zombie_shoulders_obj.py` drops the outer torso shoulder line on
the white- and green-coat zombies in both scenarios, leaving its width and arm
overlap in place. A temporary colour-coded mesh confirmed that high local Y
maps to the visible shoulder before the sculpt was authored. All four torso
parts passed viewer runs. The matched front comparison is
`build/model-viewer/entity-mesh/zombie-shoulder-candidates/zombie-drop-comparison.png`.
The contour change is modest; raised-arm animations remain to be checked.
`tools/sculpt_zombie_chest_wound_obj.py` builds on that shoulder shape for
the white-coat zombie (`em1000`, `em1100`), recessing the texture's open chest
wound and lifting its ragged edge. It replaces the two earlier torso meshes;
the authored-part count does not change. The matched front and oblique camera
comparisons are in `build/model-viewer/entity-mesh/zombie-chest-wound-candidates/`.
Both replacements pass source verification, load in 10 unfrozen viewer frames,
and render for 60 frames with the pose frozen at tick 8. The installed baseline
white zombie also times out before an unfrozen 30-frame viewer run exits;
therefore this does not establish safe grab/bite motion. Gameplay testing is
still required. The previous shoulder-only meshes are backed up under
`zombie-chest-wound-candidates/baseline-torsos/`.
The green-coat variants (`em1011`, `em1111`) now use a narrower version of
that cavity at 0.8 strength. Front and oblique comparisons are in
`build/model-viewer/entity-mesh/zombie-green-wound-candidates/`; a 1.2-strength
trial visibly warped the wound texture and was rejected. Both installed green
torsos pass verification, load through 10 unfrozen animation ticks, and
complete 60 held-pose viewer frames. Their earlier shoulder-only meshes are
backed up under `zombie-green-wound-candidates/baseline-torsos/`. Longer
grab/bite motion still requires gameplay inspection.
`tools/sculpt_naked_zombie_shoulders_obj.py` applies a similar restrained
shoulder slope to the exposed torsos of `em1001` and `em1101`. The matched
medium and close camera 4 captures are in
`build/model-viewer/entity-mesh/naked-zombie-torso-candidates/`. Both
source-hash-checked replacements loaded and completed 60-frame Wine viewer
runs. The change is small at normal distance; raised-arm gameplay poses still
need inspection.
`tools/sculpt_naked_zombie_ribs_obj.py` builds on those two torsos, raising
three rib bands and recessing the abdominal opening without changing the
shoulder or waist attachment. The 1.0-strength files replace the earlier
shoulder-only versions for `em1001` and `em1101`; the stronger 1.5 trial was
rejected. The matched close camera comparison is
`build/model-viewer/entity-mesh/naked-zombie-rib-candidates/naked-ribs-comparison.png`.
Both EMDs passed source verification and 120 unfrozen viewer frames. A later
pose capture in the same directory shows no new joint gap. Grab and bite
contact in gameplay remain untested. The shoulder-only files are backed up
under `naked-zombie-rib-candidates/baseline-torsos/`.
`tools/sculpt_zombie_fingers_obj.py` tapers and lengthens the separate hands
of the white- and green-coat zombies in both scenarios. Naked zombies have
degenerate hand joints and need a different treatment. Eight replacements
passed source-hash/topology verification; all four EMDs loaded both hands and
completed 60-frame Wine viewer runs. Matched front and oblique camera
comparisons are in `build/model-viewer/entity-mesh/zombie-finger-strong-candidates/`.
The effect is visible in the dangling fingers without changing the wrists;
grab and bite contact need gameplay inspection. A trial radial upper-arm
sculpt in `build/model-viewer/entity-mesh/zombie-upper-arm-candidates/` was
not installed because its matched camera view showed almost no useful shape
change.
`tools/sculpt_naked_zombie_fingers_obj.py` shapes the hand ends of forearm
joints 4 and 7 for the naked zombies (`em1001` and `em1101`); their separate
hand joints have almost no geometry. Four replacements passed the source-hash
and topology verifier. Both variants exited cleanly after 60 viewer frames.
The matched camera 4 comparison is in
`build/model-viewer/entity-mesh/naked-zombie-finger-candidates/`. The change
slightly tapers and lengthens the silhouette; the original connected finger
topology remains, and grab poses still need gameplay inspection.
`tools/sculpt_zombie_leg_contours_obj.py` tested the human trouser contour
deformation on the clothed zombies. The matched leg framing in
`build/model-viewer/entity-mesh/zombie-leg-candidates/` showed almost no
visible improvement, so these four trial files were not installed.
The Cerberus (`em1002.emd`) has a paired upper muzzle and animated lower-jaw
sculpt from `tools/sculpt_cerberus_muzzle.py`. The head grows from 3,936 /
1,312 to 7,872 vertices / 5,248 triangles; the jaw grows from 1,104 / 368
to 2,208 / 1,472. The matched main-hall camera 4 comparison is
`build/model-viewer/entity-mesh/obj-cerberus-refined/cerberus-muzzle-game-comparison.png`.
Both parts loaded and passed a 60-frame run. The Jill-scenario `em1102.emd`
head and jaw have identical refined geometry and hashes; the same files are
installed under those names and `RE1_MODEL_VIEW=j:6` passed a separate
60-frame run. The visible silhouette change is modest; attack and bite
contacts remain untested in gameplay.
`tools/sculpt_cerberus_haunches_obj.py` rounds the paired rear-thigh muscle
bellies between unchanged hip and lower-leg ends. Both scenario EMDs loaded
their two new parts and completed 60-frame Wine viewer runs. Matched side and
front captures are in `build/model-viewer/entity-mesh/cerberus-haunch-candidates/`;
the side view shows a fuller haunch without a joint gap. Running and jumping
poses still need gameplay inspection. A trial front/rear toe extension in
`build/model-viewer/entity-mesh/cerberus-paw-candidates/` remains uninstalled:
its side and front camera views showed too little improvement at game scale.
`tools/sculpt_cerberus_shins_obj.py` tested radial tendon and calf volume on
front shin joints 6 and 8 for both scenario EMDs. The same-engine full-body
and closer captures are under
`build/model-viewer/entity-mesh/cerberus-shin-candidates/`; the game log
confirms both files loaded. The change is nearly invisible at game distance
and the close camera still shows essentially the original straight-leg
silhouette, so the four candidates are not installed.
`tools/sculpt_cerberus_front_legs_obj.py` rounds the paired upper front-leg
muscles with unchanged ends at the shoulder and elbow. The four new meshes
cover both scenario EMDs. Matched close viewer frames are in
`build/model-viewer/entity-mesh/cerberus-front-leg-candidates/`; both scenario
models loaded the new meshes and completed 120 unfrozen frames. Running and
attack poses still need gameplay inspection.
Sixteen cloned Hunters with the replacements submitted 152,624 visible
triangles per frame in a two-frame collector check: 87,088 used the heap
spillover path and zero were dropped. This is a render-capacity check, not a
gameplay frame-rate result.
Each element is currently limited to 65,535 vertices by the viewport's
16-bit indices; the renderer's frame triangle collector can grow beyond its
old fixed 65,536-triangle array. Gameplay motion and frame rate still require
testing before using a replacement generally.

With the installed Latino HD pack, a per-EMD texture trace found 83 of 112
atlas-page loads covered before the additional seven verified aliases in
`tools/data/re1_gog_usa_hd_skins.csv`, and 94 of 112 afterward. All pages for
49 of the 67 EMDs now resolve to HD art. The before/after per-file reports are
`build/model-viewer/entity-mesh/hd-texture-audit.json` and
`hd-texture-coverage-after.json`. The nine remaining distinct missing atlases
need new art or a verified source; the pack's visually unrelated WebPs must
not be assigned just because their colour statistics are similar.
Set `RE1_HD_TEXTURE_AUDIT=1` for the per-page CRC and HD hit/miss trace on
stderr. `RE1_MODEL_VIEW_EMD_BEGIN` marks where the selected model's pages begin.

For animation checks, `RE1_MODEL_VIEW_PLAYER_ANIM=1` plays the CHAR EMD's
20-frame sequence on Chris or Jill. Two 120-frame viewer runs completed with
the 16x mesh setting, and `chris-walk-a.png` / `chris-walk-b.png` in the audit
directory show different joint poses. The viewer does not exercise collision,
weapon transitions, or a crowded room's frame rate.
Set `RE1_MODEL_VIEW_FREEZE_AFTER=8` to hold the pose after eight calls to
`Joint_move`, which makes before/after screenshots comparable at one frame.
Set `RE1_MODEL_VIEW_STILL=1` to keep the studio viewer at its starting angle
instead of rotating the model. Combine it with the pose freeze for matched
full-model comparisons. The main hall scene keeps its selected room camera.
For a crowded-model renderer check, `RE1_MODEL_VIEW_CLONES=16` submits the
same animated model sixteen times in one frame without changing its asset.
`RE1_TMD_TRI_AUDIT=1` prints collected, overflow, and dropped triangle counts.
With EMD index 17 (Yawn) at 16x refinement, a two-frame Wine check collected
121,408 triangles per frame, 55,872 beyond the old fixed 65,536-entry array,
with zero drops. The collector now allocates extra space only for those crowded
frames. This tests render capacity, not gameplay speed with sixteen Yawns.

## Play-distance shape audit

Two October 2026 candidates were inspected but not installed. A Cerberus
skull/cheek relief pass passed the `.r1mesh` verifier, yet its matched main-hall
front captures (`build/model-viewer/entity-mesh/cerberus-before-full.png` and
`build/model-viewer/entity-mesh/cerberus-head-candidates/cerberus-head-after-front.png`)
show almost no improvement at game distance. A spider pedipalp curl is hidden
by the front legs in the same camera; its candidate captures are under
`build/model-viewer/entity-mesh/spider-pedipalp-candidates/`. Their prototype
scripts were removed, and neither set of candidate files is in the installed
replacement directory.

`tools/sculpt_hunter_torso_obj.py` tested pectoral, sternum, rib and abdomen
relief on Hunter joint 1 in both scenario EMDs. The source-hash-matched torso
loaded in the game viewer, but the matched full-body camera 4 images under
`build/model-viewer/entity-mesh/hunter-torso-candidates/` differ only in a
small patch of the chest. The existing texture already gives the visible
anatomy; these two trial meshes are not installed.

The largest uncovered geometry belongs to Yawn's segmented body in all four
EMDs (`em100d`, `em1012`, `em110d`, `em1112`). The main-hall side capture at
`build/model-viewer/entity-mesh/yawn-before-side.png` shows that the body is
already rounded by the 16x pass but its rigid section joins remain visible.
The adjacent segments already overlap by roughly 450 source units. On the
camera-side flank, their exposed surfaces use substantially different texture
coordinates at the crossover: seven joins from joints 5–12 differ by about
0.26–0.41 in U. A one-section radial staggering study is saved in
`build/model-viewer/entity-mesh/yawn-seam-candidates/`; small offsets were
invisible, while a larger offset produced a lump beside the unchanged join.
That candidate and its script were rejected. Further local surface relief
cannot resolve this texture transition. A UV-only candidate matching one child
flank's U to its parent at the crossover is in
`build/model-viewer/entity-mesh/yawn-uv-candidates/`. The broad remap sampled
white space from other UV islands; restricting it to the scale strip removed
that artifact but left a new dark patch beside the join. It was also rejected
and is not installed. A useful next test needs a continuous material across
the entire overlapping surface, or a unified animated body mesh, then multiple
poses to confirm that section articulation remains intact.
`tools/sculpt_yawn_body_segments_obj.py` tested a smooth radial roundover on
body joints 3–12 of all four EMDs. Its 40 candidate files passed the static
verifier and each EMD completed a 60-frame viewer run. The matched, fully
framed side images in `build/model-viewer/entity-mesh/yawn-body-candidates/`
showed repeated bulges that made the snake's outline less continuous. The
files were removed from the installed replacements; this candidate is retained
only for comparison.
`tools/sculpt_yawn_body_taper_obj.py` takes the opposite approach: each rigid
section is tapered inward about its measured local centreline, with its two
ends unchanged. The front sections receive the strongest taper because their
middle radius rises roughly 30% above the end radius; the rear sections
receive less. All 40 body meshes are installed across the four Yawn EMDs.
Matched main-hall side captures, including animation tick 40, are in
`build/model-viewer/entity-mesh/yawn-body-taper-candidates/`. The outline is
straighter and the tick-40 pose shows no new open gap. All four EMDs loaded the
ten new body sections and completed 120 unfrozen viewer frames. The atlas
pattern still jumps at the section joins; the taper does not solve that
material seam or turn the rigid parts into a skinned body. Gameplay attacks
and room-scale performance still need testing.
`tools/sculpt_yawn_tail_obj.py` continues the inward taper across the final
two articulated tail sections in all four Yawn EMDs. It leaves the joint-13
ends and the joint-14 root fixed, narrowing the swollen middle and free tip.
Matched still-angle studio captures are
`build/model-viewer/entity-mesh/yawn-tail-candidates/tail-before3.png` and
`tail-after3.png`. The silhouette change is modest but visible. All eight
meshes passed the static verifier in the 629-file installed set, and all four
Yawn selectors completed 120 unfrozen main-hall viewer frames with them.
The texture jumps between sections and rigid tail articulation remain.

## Character lighting study

Set `RE1_MODEL_LIGHTING=cinematic` to try an opt-in model lighting preset. It
keeps the room's original ambient and three RDT lights, then adds a warm
upper-left key and a cooler camera-side fill when lighting TMD vertices. It
affects TMD models in gameplay and in the viewer; it does not change the
pre-rendered room image. The original lighting is used when the variable is
unset. Compare `build/model-viewer/chris-lighting-comparison.png`: original HD
face lighting on the left, cinematic preset on the right. This is a lighting
study, not a reproduction of the 2002 remake's renderer. It still lights per
vertex and does not cast dynamic shadows onto the background.

`RE1_MODEL_LIGHTING=remake-study` is a stronger comparison preset with a
broader key, fill, and subtle edge light. The default renderer still uses the
original lighting. In the same main-hall camera, it makes Chris's face and
forearms more readable; its effect on the zombie and Hunter is much smaller.
Matched screenshots are in `build/model-viewer/lighting-build/` as
`chris-original.png` / `chris-remake-study.png`, with equivalent pairs for
`zombie` and `hunter`. This option affects TMD vertex lighting across the
game. It does not add per-pixel materials, cast shadows, or change the room
background, and its camera-space light direction will not suit every camera.
Room-specific lights and shadows aligned to each background are the next
meaningful lighting step toward a remake-style presentation.
