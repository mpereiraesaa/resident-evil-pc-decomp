#!/usr/bin/env python3
"""Import the source-compatible parts of the Latino/HD mod as an asset mod.

The published pack targets the Japanese/Sourcenext layout. This tool extracts
only its voice WAVs and modern movie files, then writes the region-neutral
layout consumed by config.ini [Assets] Mods. It deliberately does not install
the replacement executable, DLL hooks, ASI plugin, saves, or hash-addressed HD
WebP files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


USA_MOVIE_ALIASES = {
    "ed4.mp4": "eu4.mp4",
    "ed5.mp4": "eu5.mp4",
    "oj.mp4": "ou.mp4",
    "pj.mp4": "pu.mp4",
    "staf_b.mp4": "staf_r.mp4",
    "stfc_b.mp4": "stfc_r.mp4",
    "stfj_b.mp4": "stfj_r.mp4",
    "stfz_b.mp4": "stfz_r.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Latino voices and MP4s into a native asset override"
    )
    parser.add_argument("archive", type=Path, help="source .7z mod archive")
    parser.add_argument("destination", type=Path, help="output mod root")
    parser.add_argument(
        "--region",
        choices=("USA", "JPN"),
        default="USA",
        help="movie naming used by the selected base tree (default: USA)",
    )
    parser.add_argument(
        "--voices-only",
        action="store_true",
        help="skip MP4 movies (the archive is solid, so this may not save time)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and show the extraction plan without writing files",
    )
    return parser.parse_args()


def require_inputs(archive: Path) -> str:
    if not archive.is_file():
        raise RuntimeError(f"archive not found: {archive}")
    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7z is required to read the solid archive")
    return seven_zip


def copy_tree_files(source: Path, destination: Path) -> int:
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        shutil.copy2(item, destination / item.name)
        count += 1
    return count


def copy_movies(source: Path, destination: Path, region: str) -> int:
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.suffix.lower() != ".mp4":
            continue
        output_name = item.name
        if region == "USA":
            output_name = USA_MOVIE_ALIASES.get(item.name.lower(), item.name)
        shutil.copy2(item, destination / output_name)
        count += 1
    return count


def main() -> int:
    args = parse_args()
    try:
        seven_zip = require_inputs(args.archive)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selections = ["JPN/voice/*"]
    if not args.voices_only:
        selections.append("movie/*")

    print(f"archive:     {args.archive.resolve()}")
    print(f"destination: {args.destination.resolve()}")
    print(f"region:      {args.region}")
    print(f"content:     voices{' + movies' if not args.voices_only else ''}")
    print("excluded:    executable, DLL/ASI hooks, saves, sound tree, HD WebP tree")
    if args.dry_run:
        return 0

    # Extract into a temporary directory first. A solid archive must be scanned
    # through, but no unsupported payload is ever copied into the game tree.
    with tempfile.TemporaryDirectory(prefix="re1-latino-") as temporary:
        temp_root = Path(temporary)
        command = [seven_zip, "x", "-y", f"-o{temp_root}", str(args.archive)]
        command.extend(selections)
        subprocess.run(command, check=True)

        voice_source = temp_root / "JPN" / "voice"
        if not voice_source.is_dir():
            raise RuntimeError("archive has no JPN/voice directory")
        voice_count = copy_tree_files(voice_source, args.destination / "voice")

        movie_count = 0
        if not args.voices_only:
            movie_source = temp_root / "movie"
            if not movie_source.is_dir():
                raise RuntimeError("archive has no movie directory")
            movie_count = copy_movies(movie_source, args.destination / "movie",
                                      args.region)

    print(f"imported:    {voice_count} voice files, {movie_count} movie files")
    print(f"config:      add {args.destination} to [Assets] Mods")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
