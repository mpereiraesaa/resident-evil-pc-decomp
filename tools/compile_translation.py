#!/usr/bin/env python3
"""Compile Classic REbirth RE1 text XML into small native message banks."""

from __future__ import annotations

import argparse
import html
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


MAGIC = b"RE1TXT1\0"
TEXT_PATTERN = re.compile(r"<Text>(.*?)</Text>", re.DOTALL)
TAG_PATTERN = re.compile(r"\{([a-z]+)(?: ([^}]*))?\}")


def load_encoding(path: Path) -> dict[str, int]:
    encoding: dict[str, int] = {}
    for entry in ET.parse(path).getroot().iter("Entry"):
        char = entry.get("Char")
        value = entry.get("Encode")
        if char is None or value is None:
            continue
        # The table intentionally has a few aliases. Preserve the first mapping
        # for duplicate display characters and still accept the later aliases.
        encoding.setdefault(char, int(value, 0))
    # The shipped Spanish room text uses the masculine ordinal twice, while
    # its font table only defines the visually equivalent degree glyph.
    if "°" in encoding:
        encoding.setdefault("º", encoding["°"])
    return encoding


def load_texts(path: Path) -> list[str]:
    # Two published room files contain a missing closing brace. Classic
    # REbirth accepts them, so repair that exact typo before scanning. A regex
    # reader also preserves raw control bytes that make those XML files invalid
    # according to a strict XML parser.
    source = path.read_text(encoding="utf-8-sig", errors="surrogateescape")
    source = source.replace("{scroll 0{scroll 1}", "{scroll 0}{scroll 1}")
    return [html.unescape(value) for value in TEXT_PATTERN.findall(source)]


def byte_args(tag: str, args: str | None, count: int) -> list[int]:
    values = [] if args is None else args.split()
    if len(values) != count:
        raise ValueError(f"{{{tag}}} expects {count} numeric argument(s)")
    result = [int(value, 0) for value in values]
    if any(value < 0 or value > 255 for value in result):
        raise ValueError(f"{{{tag}}} argument outside byte range")
    return result


def encode_text(text: str, encoding: dict[str, int], name: bool = False) -> bytes:
    output = bytearray()
    terminated = False
    position = 0
    while position < len(text):
        if text.startswith("\\n", position):
            output.append(0x02)
            position += 2
            continue

        if text[position] == "{":
            match = TAG_PATTERN.match(text, position)
            if match is None:
                raise ValueError(f"invalid control tag near {text[position:position + 32]!r}")
            tag, args = match.groups()
            if tag == "clear":
                output.extend((0x03, *byte_args(tag, args, 1)))
            elif tag == "scroll":
                output.extend((0x04, *byte_args(tag, args, 1)))
            elif tag == "color":
                output.extend((0x05, *byte_args(tag, args, 1)))
            elif tag == "string":
                output.extend((0x06, *byte_args(tag, args, 1)))
            elif tag == "branch":
                output.extend((0x08, *byte_args(tag, args, 3)))
            elif tag == "timed":
                output.extend((0x01, *byte_args(tag, args, 1)))
                terminated = True
            else:
                raise ValueError(f"unsupported control tag {{{tag}}}")
            position = match.end()
            continue

        char = text[position]
        if ord(char) < 0x20:
            # Some original translation files carry already encoded control
            # bytes. Keep them byte-for-byte instead of rejecting the XML.
            output.append(ord(char))
        elif char in encoding:
            output.append(encoding[char])
        else:
            raise ValueError(f"character {char!r} has no encoding entry")
        position += 1

    if name:
        output.append(0x07)
    elif not terminated:
        output.extend((0x01, 0x00))
    return bytes(output)


def write_bank(path: Path, entries: list[bytes]) -> None:
    header_size = len(MAGIC) + 4 + 4 * (len(entries) + 1)
    offsets = [header_size]
    for entry in entries:
        offsets.append(offsets[-1] + len(entry))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<I", len(entries)))
        output.write(struct.pack(f"<{len(offsets)}I", *offsets))
        for entry in entries:
            output.write(entry)


def compile_one(source: Path, destination: Path,
                encoding: dict[str, int], names: bool = False) -> int:
    texts = load_texts(source)
    if not texts:
        raise ValueError(f"no <Text> entries in {source}")
    entries = []
    for index, text in enumerate(texts):
        try:
            entries.append(encode_text(text, encoding, names))
        except ValueError as exc:
            raise ValueError(f"{source}: Text[{index}]: {exc}") from exc
    write_bank(destination, entries)
    return len(entries)


def compile_payload(source: Path, destination: Path) -> tuple[int, int]:
    encoding_file = source / "encoding.xml"
    if not encoding_file.is_file():
        raise RuntimeError("embedded translation has no encoding.xml")
    encoding = load_encoding(encoding_file)

    fixed = (
        ("misc.xml", "global.bin", False),
        ("desc.xml", "descriptions.bin", False),
        ("item_simple.xml", "item_names.bin", True),
    )
    entry_count = 0
    bank_count = 0
    for source_name, output_name, names in fixed:
        input_file = source / source_name
        if not input_file.is_file():
            raise RuntimeError(f"embedded translation has no {source_name}")
        entry_count += compile_one(input_file, destination / output_name,
                                   encoding, names)
        bank_count += 1

    room_source = source / "text"
    for input_file in sorted(room_source.glob("Room_[0-9A-Fa-f]*_[01].xml")):
        output_name = input_file.stem.lower() + ".bin"
        entry_count += compile_one(input_file, destination / "room" / output_name,
                                   encoding)
        bank_count += 1
    if bank_count == len(fixed):
        raise RuntimeError("embedded translation has no room message XML")
    return bank_count, entry_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile Classic REbirth RE1 XML translation banks"
    )
    parser.add_argument("source", type=Path, help="extracted translation payload")
    parser.add_argument("destination", type=Path, help="output translation folder")
    args = parser.parse_args()
    banks, entries = compile_payload(args.source, args.destination)
    print(f"compiled: {banks} banks, {entries} text entries")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, ET.ParseError) as exc:
        raise SystemExit(f"error: {exc}")
