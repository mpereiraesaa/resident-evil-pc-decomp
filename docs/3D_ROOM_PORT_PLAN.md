# 3D room port plan

## Goal

Add an opt-in path for rendering a room as a real 3D scene while retaining the
PC release's room scripts, fixed-camera cuts, player movement, enemy logic,
doors, pickups, sound zones, and collision behavior. The original pre-rendered
room remains the default and fallback. A shoulder camera is a later project;
the first room must work with its authored RDT cameras.

## What the current data provides

The room definition table (RDT) contains useful gameplay and camera data:

- Camera position, target, roll, and field of view for each fixed shot.
- Camera-switch trigger quads, evaluated from the player's room-space X/Z.
- Room collision boundaries, NPC walk zones, footstep zones, and room lights.
- Room scripts and the tables for interactive objects.
- Background mask/overlay images and their screen-space sprite placement.

The room backdrop itself is loaded separately from a compressed PAK for each
camera and drawn as a 320x240 image. The room collision records describe where
gameplay blocks movement; they are not a render mesh and do not describe wall
height, ceiling shape, materials, or UVs. The installed USA data also supplies
TMD/TIM models for actors, items, and selected room objects, but the base rooms
do not contain a complete walkable 3D environment that can simply be enabled.

## Existing rendering capabilities

The Windows Marni backend is already D3D11. It renders 3D TMD models, composes
their transforms with `g_RoomCameraData`, projects triangles, applies lights,
and substitutes HD textures for matching PSX texture pages. The common Marni
API also has `DrawTriangles3D`, which depth-tests and writes opaque triangles;
the D3D11 and OpenGL backends both implement it. Room cameras are loaded from
the RDT by `Room_SetupCamera`; `check_camera_switch` and the SCD scripts
already control the original camera cuts.

The current room image is queued through the 2D/background sprite path. The
existing TMD path is built around PS1 model resources, not a general static-
level asset loader. The frame pipeline draws the background before the TMD
triangles, so the static room pass can be inserted after background setup and
before actors to let the shared depth buffer occlude them. The missing pieces
are authored geometry, a static-mesh loader and projection path, and a way to
skip the pre-rendered quad only when a valid mesh is active.

## Recommended first slice

1. Start with `USA/Stage1/ROOM1180.RDT`, the mansion storeroom under the east
   stairs. Its RDT has two fixed cameras, making it a bounded first slice. The
   two original views can be unpacked with `tools/pak_view.py` from
   `Stage1/RC1180.pak` and `Stage1/RC1181.pak`.
   The RDT is 149,860 bytes; its camera records use raw room-space positions
   `(2682, -1836, 1872)` → `(4950, -2628, 9018)` and
   `(2502, -1800, 9900)` → `(4950, -2610, 2088)`, both with FOV `207` and roll
   zero. The positions confirm that the two views look toward opposite ends of
   the same room-space, but they are not room geometry by themselves.
2. The camera-switch table has one active quad in each direction: camera 0 → 1
   uses `(678,4927), (515,5595), (9285,5292), (9464,4043)`; camera 1 → 0 uses
   `(0,300), (0,10600), (9600,10600), (9600,300)`. The engine tests these quads
   against the player's X/Z; camera transitions therefore remain fully owned
   by the original room logic.
3. The collision-boundary header is at RDT offset `0x170`, with quadrant split
   `(4572,4338)` and 16 records grouped `3/4/4/5`. Their aggregate X/Z extent
   is `0..10050` by `0..10744`; 14 are rectangular records and 2 are circular.
   Those records constrain gameplay, not wall height or the room silhouette.
   The init script at `0x268` defines the exit as a `door_set` zone
   `(5600,8200)` sized `1200x1300`, to room `0x0B`; arrival
   `(9500,0,14300)` belongs to that destination room. The script also activates
   one item model and room-object model slot 0 with flags `0x11` (not model
   id `0x11`). Keep these
   interactions and original TMD props in the game path; map their actual world
   placements against the two reference images before final mesh authoring.
4. Author a small room shell in Blender from the reference images, using game-room
   coordinates. Start with floor, walls, door openings, and a few large
   landmarks; do not attempt to infer the mesh from collision rectangles.
5. The first loader slice accepts a deliberately small OBJ/MTL subset (`v`,
   `vt`, `mtllib`, `usemtl`, `f`, `newmtl`, `Kd`, and `map_Kd`) from a
   mod/override directory. PNG materials use RGBA8; missing/invalid images
   fall back to flat blockout colors, and absent/invalid room geometry keeps
   the original RDT/PAK path. This is an interchange prototype, not a
   production mesh pipeline.
6. Project the static shell from the RDT camera's `from`/`to` and FOV using the
   same orthonormal basis as the collision-debug overlay. Reuse the game's
   world-space actors and props so their alignment can be checked against the
   shell; the direct camera basis avoids actor-specific Y offsets.
7. Keep original scripts and collision active for this first slice. Add only
   the minimum new collision surface data if the RDT boundaries cannot prevent
   walking through the authored shell.
8. A mod root opts a room into the renderer by containing
   `Stage<S>/room<S><RR><V>.obj` (for example, `Stage1/room1180.obj` for Chris
   in ROOM118). `<S>` is the one-based room file stage, `<RR>` is the two-digit
   room id, and `<V>` is the character variant. The loader follows the
   engine's RDT file-stage and Chris/Jill selectors, releases the previous
   room's material textures at a room transition, and leaves the original
   background active when the requested mesh is absent or invalid.

## Acceptance checks

- With the option off, the existing room renders and behaves as before.
- With the option on, the same RDT camera cuts align the 3D shell, player,
  enemies, and existing room objects.
- Player movement, interaction points, room scripts, sounds, and door
  transitions still use the original room logic.
- Walls correctly occlude characters and objects; transparent effects and
  screen-space UI remain readable.
- Invalid or absent room assets produce a logged fallback, not a crash.

The first milestone should not include a shoulder camera. That mode would need
continuous camera control and separate validation of camera-relative movement,
camera collision, enemy visibility/aiming, and the room's original camera-zone
assumptions.

## Current prototype and next implementation input

`src/game/Room3D.cpp` provides the opt-in static mesh path for data-selected
room files, and
`tests/fixtures/room3d/Stage1/room1180.obj` is a validation blockout: a closed
room shell, a simple door leaf, prop silhouettes, and a generated 256x128
low-resolution texture atlas. Its floor height and major landmarks have been
adjusted against both unpacked 320x240 camera images using an offline projection
check. The floor sits at `y=0`, matching the plane used by the existing RDT
collision overlay. This only validates the camera-space composition of the
fixture; it does not prove in-game actor alignment, faithfully reconstruct the
room art, or replace any game asset.

The room renderer clips mesh triangles against its near plane before projection.
This matters for a camera placed inside a room: dropping a whole triangle when
one corner is behind the camera removes large floor and wall sections. An
offline comparison and the integration test show floor triangles in both
original views; runtime camera, actor, and collision alignment still needs an
in-game check.

Run the headless 32-bit integration test from the repository root with:

```sh
g++ -m32 -std=c++17 -ffunction-sections -fdata-sections -Isrc -I. \
  tests/room3d_render_test.cpp -Wl,--gc-sections -o /tmp/room3d_render_test
/tmp/room3d_render_test
```

The same test can also be built for the actual Windows 32-bit target and run
under Wine with the project's MinGW compiler:

```sh
i686-w64-mingw32-g++ -std=c++17 -O0 -ffunction-sections -fdata-sections \
  -I. -Isrc -Isrc/game tests/room3d_render_test.cpp \
  -Wl,--gc-sections -static -static-libgcc -static-libstdc++ \
  -o /tmp/room3d_render_test.exe
WINEDEBUG=-all wine /tmp/room3d_render_test.exe
```

This PE32/Wine run exits successfully. Keep the cross-test on MinGW: a separate
Clang PE32 build faults in its C++ runtime while constructing an MTL filename
string, while the project's MinGW build and Wine run pass.

It loads the committed OBJ/MTL/PNG fixture through the production loader,
supplies both ROOM1180 RDT cameras, invokes the room renderer for each view,
and checks that the atlas uploads, source floor vertices use `y=0`, floor
triangles survive near clipping, and finite triangles reach a stub Marni
backend. This covers parser and projection integration, not the real D3D11/Wine
device or in-game actor alignment.

The next real milestone is to replace the rough shell with a faithful storeroom
mesh and original-style low-resolution materials based on the unpacked camera
backgrounds and room data, then check actor alignment and camera cuts in-game.
Keep RDT scripts, movement, and collision authoritative while doing that. A
shoulder camera remains a separate later effort.

### Initial material path and remaining texture work

The blockout loader now accepts OBJ `vt` coordinates and face references,
`mtllib`/`newmtl`, diffuse `Kd`, and `map_Kd` PNG files. The decoder expands
grayscale, RGB, indexed, grayscale-alpha, and RGBA PNGs to RGBA8; uploaded
textures are passed to the D3D11/OpenGL triangle path with perspective-correct
UVs. Missing or invalid PNGs log a diagnostic and keep the material's flat
fallback color. The interchange subset does not yet support MTL texture
options, filenames containing spaces, normal maps, alpha-tested geometry, or
mipmaps; material names are limited to 63 bytes and texture filenames to 255
bytes. Material paths must stay relative to the OBJ directory.

The RDT and PAK provide cameras, masks, and the pre-rendered views, not a set of
seamless wall/floor texture sheets or a complete room mesh. The two camera
images therefore serve as reconstruction references; the shell needs authored
low-resolution materials and UVs that remain fixed in room space. Camera-space
projection of a whole background would only reproduce the baked image and
would not be a suitable texture solution for the later shoulder-camera phase.

### Local runtime smoke test

To enable the blockout in the Windows cross-build, add these values to the
`config.ini` beside `build/windows-cross/residentevil.exe` (keep any other mod
roots after the fixture if they are still needed):

```ini
[Assets]
Path=../../assets
Version=USA
Mods=../../tests/fixtures/room3d

[Debug]
EnableDebug=1
```

Start the game, press F1, confirm `ROOM CHANGE`, select mansion stage 1F and
room `ROOM 118` (`MANSION STOREROOM`), then press Enter. The debug room change
uses the game's ordinary room-load, script, camera, and collision paths. The
fixture should replace only the backdrop; F8 can show the original collision
overlay. From the storeroom, move through its existing camera trigger to check
the second fixed view.

The local Wine smoke test used `SDL_AUDIODRIVER=dummy RE1_DEBUGLOG=1
WINEDEBUG=-all` in an isolated folder with the fixture mod root. It initialized
D3D11, entered the game through the title and character screens, and used the
F1 debug room selector to load `ROOM 118` through the original room transition.
The runtime log records `[room3d] loaded ... (90 triangles)`, and the rendered
scene shows the fixture's textured shell and props together with Chris. Walking
through the room's original camera trigger changes to the opposite fixed view;
the original RDT and gameplay path remain active. Both views were captured
locally under `build/room3d-validation-run/screenshots/`.

This verifies that the opt-in mesh path runs in the actual Windows PE32 build
under Wine and follows the room's two camera cuts. It does not validate a
faithful reconstruction, exact actor alignment against the original backdrops,
or wall collision against the new mesh; those remain before a production room
port. Direct comparison with the unpacked original 320x240 camera images
confirms the current shell and props are still a coarse blockout. Its RDT
collision data remains authoritative. The loader now selects separate
room/stage/character mesh paths and has a headless transition test, so the next
fidelity work can author and validate room assets incrementally without another
room-specific code branch.

### ROOM118 geometry reconstruction pass

`python3 tools/build_room118_mesh.py` reproduces the reconstruction OBJ (initially 450 triangles).
The generator contains the source measurements and distinguishes recovered
placements from visually estimated decorations. It distributes no background
pixels; materials still use the synthetic atlas and simple flat colors.

- The walkable interior is bounded by blocking rectangle faces at X=1854/7344
  and Z=1944/8948. The old 0..10050 by 0..10744 shell incorrectly used the
  *outer* collision bounds and made the room too large.
- The original moving chest lid belongs to omodel slot 0. The command at
  RDT+0x2D4 places it at `(6320,-1410,1770)` with yaw `0xC00`; its TMD vertices
  span local X=0..1440, Y=-90..51, Z=-1080..1080. The reconstructed body sits
  below that lid at X=5240..7400, Z=1770..3210, with no duplicate static lid.
- The barrel and crates occupy the opposite end from the door. The typewriter
  stands on a pedestal table near `(2500,8500)`. Those decorative dimensions,
  the 6000-unit ceiling and the 3300-unit door height are visual estimates.
- RDT camera 1 is outside the interior wall at Z=9900. The original RC1181
  image omits the nearby wall and typewriter. `room1180.cameras` hides the
  OBJ group `door_end` in that fixed view, retaining the side walls and full
  geometry in the asset. This cutaway is inferred from the original images.
  A future free-camera mode needs its own visibility policy.

The optional `.cameras` file sits beside the OBJ and accepts `hide <camera>
<group>` records, with zero-based camera indices 0..31 and OBJ group names
shorter than 64 characters. Unlisted groups remain visible. These overrides
apply only to static room meshes, not actors. The room pass is also suppressed
during the original door scene.

The Windows x86 cross-build was run under Wine with the 450-triangle mesh.
Both fixed views were inspected after an ordinary debug room transition and
walking through the camera trigger. Camera 1 retains its right wall without
the empty vertical strip caused by the earlier global clipping experiment;
the chest body aligns with the original animated lid. Captures are local at
`build/room3d-fidelity-run/screenshots/room118-cam{0,1}.png`.
The headless 32-bit renderer regression covers view-specific group exclusion,
floor clipping, room/character asset selection and door-scene suppression;
the platform boundary check also passes. Materials, lighting and decorative
shapes remain approximations, so this is not yet the faithful finished room.

A subsequent surface-layout pass subdivides the floor and ceiling into 4x4
atlas panels, aligns floor grain with the long Z axis, and adds the low wall
skirting visible in the references. That pass produced 570 triangles. The
32-bit renderer regression passes, and both views were inspected in Wine after
leaving ROOM118 and returning to force an asset reload. Captures are
`room118-material-cam0.png` and `room118-material-cam1.png` in the same local
screenshots directory. This corrects texture scale/orientation only: the
synthetic atlas still has overly regular seams, and the blue lamp pool and
surface shading remain absent. These are outstanding fidelity work, not a
claim that the original materials have been recovered.


The renderer now accepts the OBJ vertex-color extension `v x y z r g b`.
RGB components must be finite and in [0,1]; omitted colors default to white,
and the traditional single optional homogeneous coordinate remains accepted.
Vertex RGB multiplies MTL diffuse color and is interpolated when clipping.
Parser regressions cover missing/partial colors, invalid ranges and NaN.

The generator authors a static blue light pool below the ROOM118 lamp and
ambient bounce toward the door. Walls use subdivided neutral plaster faces;
props and floor receive the same spatial color field. These are visual
estimates from the backgrounds, not recovered original lights or a shadow
solver. Plaster texture detail and cast/contact shadows remain unfinished.

The vertex-color build was compiled for Windows x86 and inspected in Wine in
both ROOM118 fixed views, including the original camera trigger. Local evidence:
`build/room3d-fidelity-run/screenshots/room118-light-cam0.png` and
`room118-light-cam1.png`. The blue pool appears beneath the lamp, with darker
ceiling and furniture. Walls are currently smooth because the previous coarse
placeholder texture was removed; this remains an intermediate lighting pass.
The 32-bit renderer regression and platform-boundary check pass.


The next furniture pass replaces the chest body's horizontal wood placeholder
with a muted metal finish, adds frame rails, rivets and an open carrying handle,
and gives the typewriter individual keys, spacebar, platen and carriage lever.
These details are authored approximations of the background silhouette, not
recovered original meshes. The game still owns the original animated chest lid.
The furniture pass was inspected in both Wine camera views; local captures are
`room118-props-cam0.png` and `room118-props-cam1.png`. Grounded box undersides
are omitted to avoid coplanar floor overlap. The headless renderer test passes.
The floor-underside removal alone did not eliminate the dark wedges along the
skirting in camera 0. The generator now also omits skirting back faces coplanar
with the wall. This latter correction passes the renderer regression but still
requires a fresh in-game asset reload and screenshot before claiming the
artifact is fixed. Current running game predates that last mesh edit.


### Linear-depth interpolation correction

Reloading the mesh with hidden skirting backs still reproduced camera 0's black
wedges. The actual cause was shared shader depth interpolation: callers supply
linear view Z normalized to [0,1], whereas hardware interpolates NDC depth
linearly in screen space. That is incorrect for this linear-depth encoding.
For example, halfway on screen between depths 1000 and 9000, perspective
interpolation gives 1800, while affine interpolation gives 5000. Differently
subdivided wall and skirting faces therefore gave inconsistent occlusion.

Both backends now pass the existing normalized depth as a perspective-correct
varying and write it through `SV_Depth` / `gl_FragDepth`. The CPU depth mapping
and shared actor/shadow/room depth range remain unchanged. The 2D D3D vertex
shader supplies zero depth; 2D draws continue with depth testing disabled.
Writing fragment depth can reduce early-depth optimizations; broader performance
and gameplay regression checks remain necessary before publishing this change.

The Windows x86 build ran under Wine, including title menus, a room transition,
and both ROOM118 views. The original wedge reproduction is eliminated in
`build/room3d-fidelity-run/screenshots/room118-depth-cam0.png`; camera 1 was
also inspected (`room118-depth-cam1.png`). GLSL vertex/fragment linking passed
`glslangValidator -l` on the extracted production shaders. Native backend C++
compilation remains unverified here: the Linux build directory has no generated
Makefile, and the direct 32-bit compile lacks SDL2/_real_SDL_config.h. Only the
x86_64 SDL configuration header is installed. This is not a native runtime pass.


The depth correction also has an automated headless GPU regression:

```sh
python3 tests/check_depth_shader.py
```

It extracts the current production GL shaders, compiles and links them in an
EGL OpenGL 3.3 context, renders an inclined plane with one and eight panels in
both drawing orders against an intervening flat surface, and reads back color
and depth. Expected depth comes from analytic ray/plane intersection. It also
checks a depth-disabled 2D overlay. The host-native harness does not require the
32-bit SDL development package and does not claim to test the game's native ABI.
Dependencies are a C++17 compiler, pkg-config and EGL/OpenGL development libraries.

The current production shaders pass. Running with `--affine-control` deliberately
restores the old fragment-depth behavior and must fail; this negative control
was run and reproduced the incorrect inclined-surface depth. This adds actual GL
shader execution evidence beyond the earlier compile/link-only validation.
Keep this renderer correction and its regression as a separate review unit from
room-art revisions; it changes shared model, shadow and room rendering.


### Plaster material pass

`room1180_plaster.png` is a newly generated neutral plaster albedo (1254x1254),
authored with the image-generation tool on 2026-10-11. It is not extracted from
Capcom assets and must not be described as an original recovered texture. The
prompt requested fine, low-contrast gray plaster, subtle mottling, no cracks,
objects, baked lighting or blue tint. The generated file is copied verbatim;
the MTL and vertex lighting provide the scene's color and shading.

Wall subdivisions now retain continuous UVs over each wall face instead of
repeating a whole tile per lighting cell. The headless renderer test decodes
both textures, distinguishes their handles and verifies both are released when
changing rooms. The fixture PNGs have explicit .gitignore exceptions, preventing
a clean checkout from silently losing the previously ignored synthetic atlas.

Both fixed views were inspected after a Wine room reload. Local captures:
`build/room3d-fidelity-run/screenshots/room118-plaster-cam0.png` and
`room118-plaster-cam1.png`. Grain is visible without the old large checker-like
wall pattern, and the authored lamp lighting remains continuous. The surface
still needs comparison at original output resolution; this is an approximation,
not a completed fidelity claim. Floor, ceiling, door, furniture wear and contact
shadows remain outstanding, along with all other room reconstructions.

### Item-box interaction validation

A live ROOM118 interaction exposed an integration defect: the static room pass
ran after the menu background fills and appeared inside the item-box panel.
`Room3D_IsActive` now suppresses the room for `MSF_MENU_ACTIVE` and
`MSF_SCREEN_STANDALONE`, as well as door transitions. Menu close restores room
rendering using its cached mesh/textures. The renderer regression checks both
screen flags independently, absence of draw calls while hidden, resumed drawing,
and unchanged texture allocation count across UI transitions.

The Windows x86 build and platform-boundary check pass. In Wine, Chris approached
the chest, opened its original animated lid, entered the item-box UI and closed
it with the cancel key. The menu now has its intended opaque background and the
3D room returns with the lid closed. No item transfer or save operation was
performed. Local evidence: `room118-chest-opening.png`,
`room118-chest-menu-fixed.png`, and `room118-chest-return.png` under
`build/room3d-fidelity-run/screenshots/`. The opening capture predates the menu
suppression fix; the menu and return captures use the corrected build.

### ROOM118 character variants

The supplied USA `ROOM1180.RDT` and `ROOM1181.RDT` are byte-identical (149860
bytes, SHA-256 `d2635a5aebf5b9eb97cc68a1966b0838c442f799841ca64fd83cf1bc82d7c52c`).
This supports sharing this room's static geometry and camera cutaways across
Chris and Jill; it is not evidence that other rooms share their layouts.
The generator now emits both `room1180.obj` and `room1181.obj` by default,
with explicit camera sidecars. Both meshes reference the same material library
and textures. `--output` retains the single-output export behavior.
The headless renderer test loads Jill's actual fixture, checks the requested
variant path, draws it and verifies that textures from the preceding variant
were released. No engine-wide fallback between character variants was added.

Jill was selected through a new game and ROOM118 entered via the debug room
selector in Wine. The runtime log confirms `room1181.obj` loaded (3572 triangles).
Both camera views and the original camera trigger were inspected, then the
item-box menu opened successfully with no room geometry behind its panels.
Local captures: `room118-jill-cam0.png`, `room118-jill-cam1.png`,
`room118-jill-itembox.png` in the fidelity-run screenshots directory.
No inventory transfers or saved-game writes were performed. This validates
this room's variant integration, not fidelity completion or other Jill rooms.

### ROOM10B measured survey

`tools/survey_room3d.py` exports the stored USA PC RDT camera records and
collision bounds to JSON and an X/Z SVG plan. For example:

```sh
python3 tools/survey_room3d.py assets/USA/Stage1/ROOM10B0.RDT --out build/room3d-survey
```

ROOM10B has four cameras and eleven distinct collision boundaries. Its Chris
and Jill files differ, but their extracted cameras, boundaries and quadrant
counts match. This does not establish equivalence of scripts or objects.
The parser consumes the first four boundary counts, matching
`Room_SetupCollisionCallbacks`, preserves duplicate record/quadrant membership,
and rejects truncated camera and collision records. Manual validation against
ROOM10B's two variants and ROOM118 succeeded; four truncated inputs were rejected.

The four original backgrounds show a corridor and stairwell. Camera heights
range from Y=-1116 to Y=-10494. Collision bounds alone do not establish the
stairs' elevation, ceiling or visible wall surfaces. No ROOM10B mesh has been
added yet: stair heights and transitions must be established before constructing
and validating the room. The survey outputs are local build artifacts, not
redistributed game assets.
