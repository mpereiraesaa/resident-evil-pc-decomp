#!/usr/bin/env python3
"""Import compatible Latino/HD pack content as a native asset mod.

The published pack targets the Japanese/Sourcenext layout. This tool extracts
its voice WAVs, modern movie files and confidently matched USA GOG room
backgrounds, then writes the layout consumed by config.ini [Assets] Mods. It
does not install the replacement executable, hook DLLs, ASI plugin or saves.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from compile_translation import compile_payload


HD_BACKGROUND_MAP = (
    Path(__file__).parent / "data" / "re1_gog_usa_hd_backgrounds.csv"
)
HD_MASK_MAP = Path(__file__).parent / "data" / "re1_gog_usa_hd_masks.csv"


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
        description="Import Latino voices, MP4s and USA room backgrounds"
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
        help="skip MP4 movies and HD backgrounds",
    )
    parser.add_argument(
        "--no-hd",
        "--no-hd-backgrounds",
        dest="no_hd",
        action="store_true",
        help="skip native HD room backgrounds, masks and the libwebp runtime",
    )
    parser.add_argument(
        "--no-translation",
        action="store_true",
        help="skip the embedded Spanish font and text banks",
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


def copy_hd_files(source: Path, destination: Path, mapping_file: Path,
                  kind: str) -> int:
    if not mapping_file.is_file():
        raise RuntimeError(f"HD {kind} map not found: {mapping_file}")

    count = 0
    with mapping_file.open(newline="", encoding="ascii") as mapping:
        for row in csv.DictReader(mapping):
            output = Path(row["path"])
            if output.is_absolute() or ".." in output.parts:
                raise RuntimeError(f"unsafe HD output path in map: {output}")
            input_file = source / row["webp"]
            if not input_file.is_file():
                raise RuntimeError(
                    f"archive has no mapped HD {kind} file: {row['webp']}"
                )
            output_file = destination / output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_file, output_file)
            count += 1
    return count


def extract_embedded_7z(source: Path, destination: Path, seven_zip: str) -> None:
    """Extract the valid 7z stream embedded in the replacement ddraw.dll."""
    data = source.read_bytes()
    signature = b"7z\xbc\xaf\x27\x1c"
    candidates = []
    position = data.find(signature)
    while position >= 0:
        if position + 32 <= len(data):
            next_offset, next_size = struct.unpack_from("<QQ", data, position + 12)
            total_size = 32 + next_offset + next_size
            if 32 <= total_size <= len(data) - position:
                candidates.append((position, total_size))
        position = data.find(signature, position + 1)
    if not candidates:
        raise RuntimeError("ddraw.dll has no valid embedded 7z translation")

    # The resource stream is the largest valid candidate. This also ignores a
    # shorter false-positive signature present in the DLL code/data section.
    start, size = max(candidates, key=lambda candidate: candidate[1])
    archive = destination.parent / "translation.7z"
    archive.write_bytes(data[start:start + size])
    selections = (
        "encoding.xml", "misc.xml", "desc.xml", "item_simple.xml",
        "text/Room_*.xml", "DATA/PBFONT0.tim",
    )
    command = [seven_zip, "x", "-y", f"-o{destination}", str(archive)]
    command.extend(selections)
    subprocess.run(command, check=True)


def import_translation(ddraw: Path, destination: Path,
                       temporary: Path, seven_zip: str) -> tuple[int, int]:
    payload = temporary / "translation-payload"
    extract_embedded_7z(ddraw, payload, seven_zip)
    font = payload / "DATA" / "PBFONT0.tim"
    if not font.is_file():
        raise RuntimeError("embedded translation has no DATA/PBFONT0.tim")
    font_destination = destination / "data" / "fontus.tim"
    font_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(font, font_destination)
    return compile_payload(payload, destination / "translation")


def main() -> int:
    args = parse_args()
    try:
        seven_zip = require_inputs(args.archive)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import_hd = (not args.voices_only and not args.no_hd and
                 args.region == "USA")
    import_text = not args.voices_only and not args.no_translation
    selections = ["JPN/voice/*"]
    if not args.voices_only:
        selections.append("movie/*")
    if import_hd:
        selections.extend(("hires/bgd/*", "hires/mask/*", "libwebp.dll"))
    if import_text:
        selections.append("ddraw.dll")

    print(f"archive:     {args.archive.resolve()}")
    print(f"destination: {args.destination.resolve()}")
    print(f"region:      {args.region}")
    content = "voices"
    if not args.voices_only:
        content += " + movies"
    if import_hd:
        content += " + HD room art"
    if import_text:
        content += " + Spanish text/font"
    print(f"content:     {content}")
    print("excluded:    executable, hook DLLs/ASI, saves, sound tree, HD misc/skins")
    if args.region == "JPN" and not args.voices_only and not args.no_hd:
        print("note:        direct HD room map currently targets USA GOG assets")
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

        hd_count = 0
        mask_count = 0
        if import_hd:
            hd_source = temp_root / "hires" / "bgd"
            if not hd_source.is_dir():
                raise RuntimeError("archive has no hires/bgd directory")
            hd_count = copy_hd_files(hd_source, args.destination,
                                     HD_BACKGROUND_MAP, "background")
            mask_source = temp_root / "hires" / "mask"
            if not mask_source.is_dir():
                raise RuntimeError("archive has no hires/mask directory")
            mask_count = copy_hd_files(mask_source, args.destination,
                                       HD_MASK_MAP, "mask")
            webp_runtime = temp_root / "libwebp.dll"
            if not webp_runtime.is_file():
                raise RuntimeError("archive has no 32-bit libwebp.dll")
            shutil.copy2(webp_runtime, args.destination / "libwebp.dll")

        text_bank_count = 0
        text_entry_count = 0
        if import_text:
            ddraw = temp_root / "ddraw.dll"
            if not ddraw.is_file():
                raise RuntimeError("archive has no ddraw.dll translation payload")
            text_bank_count, text_entry_count = import_translation(
                ddraw, args.destination, temp_root, seven_zip
            )

    print(f"imported:    {voice_count} voice files, {movie_count} movie files, "
          f"{hd_count} HD backgrounds, {mask_count} HD masks, "
          f"{text_bank_count} text banks ({text_entry_count} entries)")
    print(f"config:      add {args.destination} to [Assets] Mods")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
