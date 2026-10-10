#!/usr/bin/env python3
"""Build the small LGPL FFmpeg decoder used by the Win32/Wine FMV backend."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
SOURCE = BUILD / "ffmpeg-win32-src"
PREFIX = BUILD / "ffmpeg-win32-prefix"
VERSION = "9.0.2"
ARCHIVE = SOURCE / f"ffmpeg-{VERSION}.tar.xz"
URL = f"https://ffmpeg.org/releases/ffmpeg-{VERSION}.tar.xz"
SHA256 = "8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e"
STAMP = PREFIX / f".ffmpeg-{VERSION}-win32-lp64-disabled"


def run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    if STAMP.is_file():
        print(f"FFmpeg {VERSION} Win32 decoder already built")
        return 0

    SOURCE.mkdir(parents=True, exist_ok=True)
    PREFIX.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        print(f"Downloading FFmpeg {VERSION} source from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"FFmpeg source SHA-256 mismatch: {digest}")

    if not (SOURCE / "configure").is_file():
        with tarfile.open(ARCHIVE, "r:xz") as tar:
            # Reject absolute and parent-relative paths before extracting.
            for member in tar.getmembers():
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise SystemExit(f"unsafe FFmpeg archive member: {member.name}")
            tar.extractall(SOURCE, filter="data")
        nested = SOURCE / f"ffmpeg-{VERSION}"
        if nested.is_dir():
            for item in nested.iterdir():
                shutil.move(str(item), SOURCE / item.name)
            nested.rmdir()

    if not (SOURCE / "config.mak").is_file():
        run([
            "./configure",
            f"--prefix={PREFIX}",
            "--target-os=mingw32",
            "--arch=x86",
            "--cross-prefix=i686-w64-mingw32-",
            "--enable-cross-compile",
            "--disable-shared",
            "--enable-static",
            "--disable-programs",
            "--disable-doc",
            "--disable-network",
            "--disable-autodetect",
            "--disable-everything",
            "--enable-avcodec",
            "--enable-avformat",
            "--enable-avutil",
            "--enable-swresample",
            "--enable-swscale",
            "--enable-protocol=file",
            "--enable-demuxer=mov",
            "--enable-decoder=h264,aac",
            "--enable-parser=h264,aac",
            "--disable-debug",
            "--disable-w32threads",
            "--disable-x86asm",
        ], cwd=SOURCE)

    jobs = max(1, min(4, os.cpu_count() or 1))
    run(["make", f"-j{jobs}"], cwd=SOURCE)
    run(["make", "install"], cwd=SOURCE)
    expected = ["libavformat.a", "libavcodec.a", "libavutil.a",
                "libswresample.a", "libswscale.a"]
    missing = [name for name in expected if not (PREFIX / "lib" / name).is_file()]
    if missing:
        raise SystemExit("FFmpeg install missing: " + ", ".join(missing))
    STAMP.write_text(f"FFmpeg {VERSION}; target i686-w64-mingw32; LGPL 2.1+\n")
    print(f"Built FFmpeg {VERSION} Win32 static decoder in {PREFIX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
