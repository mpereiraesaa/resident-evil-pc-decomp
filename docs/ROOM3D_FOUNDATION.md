# Offline room reconstruction foundations

These modules provide independently usable foundations for the room batcher.
They do not change the game renderer or reconstruct a room on their own. Later
changes add room surveys, fitting, appearance and reporting.

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

Publication follows dependency order in small changes. This foundation is
separate from runtime integration and the full batcher; those remain subsequent
steps until their source and validation are included. Generated room exports,
original game assets, downloaded model weights and caches are not source files.
