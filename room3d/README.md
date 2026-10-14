# Room 3D corrections

Portable manual corrections for the reconstructed room meshes, plus the review
state of every room. Nothing here is a game asset: the files hold object
identities, transforms, parametric additions and content hashes only. Players
supply their own purchased game files; the automatic export and the replayed
corrections are generated locally from them.

- `overrides/ROOM1180.json` — one override file per RDT, written by
  `tools/room3d_overrides.py` (`init`, `add`, `replay`, `check`, `compare`).
  `python tools/batch_room3d.py ... --overrides room3d/overrides` replays every
  file into `<out>/corrected/<room>/` after the automatic export.
- `status.json` — review status (`unreviewed`, `needs_work`,
  `visually_approved`) and, separately, gameplay validation (`not_validated`,
  `validated`, `broken`), maintained with `tools/room3d_rooms.py status`.
  `tools/room3d_rooms.py catalog` lists unique layouts versus scenario variants.

See "Portable overrides and replay" in `docs/ROOM3D_FOUNDATION.md`.
