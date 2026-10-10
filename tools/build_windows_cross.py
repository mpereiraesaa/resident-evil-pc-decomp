#!/usr/bin/env python3
"""Build the Win32 D3D11 game from Linux with Clang and MinGW-w64.

Game.vcxproj remains the Windows source list.  Reading it here prevents this
developer build from silently drifting away from the MSVC release build.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Game.vcxproj"
BUILD = ROOT / "build" / "windows-cross"
OBJ = BUILD / "obj"
EXE = BUILD / "residentevil.exe"
FFMPEG_PREFIX = ROOT / "build" / "ffmpeg-win32-prefix"
TARGET = "i686-w64-windows-gnu"


def sources() -> list[Path]:
    tree = ET.parse(PROJECT)
    ns = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
    return [ROOT / node.attrib["Include"].replace("\\", "/")
            for node in tree.findall(".//m:ClCompile", ns)
            if "Include" in node.attrib]


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("-j", type=int, default=os.cpu_count() or 1,
                        help="reserved for compatibility; compilation is incremental")
    args = parser.parse_args()

    run([sys.executable, str(ROOT / "tools" / "build_ffmpeg_win32.py")])

    if args.clean and BUILD.exists():
        shutil.rmtree(BUILD)
    OBJ.mkdir(parents=True, exist_ok=True)

    compiler = shutil.which("clang++")
    if compiler is None:
        raise SystemExit("clang++ is required")

    common = [
        compiler, f"--target={TARGET}", "-std=c++17", "-O2",
        "-fomit-frame-pointer", "-ffunction-sections", "-fdata-sections",
        "-fms-extensions", "-Wno-everything", "-Isrc",
        f"-I{FFMPEG_PREFIX / 'include'}", "-DRE1_USE_FFMPEG",
        "-DWIN32", "-D_WINDOWS", "-DNDEBUG",
        "-DDIRECT3D_VERSION=0x0b00", "-D_CRT_SECURE_NO_WARNINGS",
    ]

    objects: list[Path] = []
    for source in sources():
        relative = source.relative_to(ROOT).as_posix().replace("/", "_")
        obj = OBJ / (relative + ".o")
        objects.append(obj)
        dependencies = [source, PROJECT]
        if source.name == "video.cpp" and source.parent.name == "win32":
            dependencies.append(source.parent / "video_ffmpeg.inc")
        if obj.exists() and obj.stat().st_mtime >= max(p.stat().st_mtime for p in dependencies):
            continue
        run(common + ["-c", str(source), "-o", str(obj)])

    response = BUILD / "objects.rsp"
    response.write_text("\n".join(f'"{obj}"' for obj in objects) + "\n")
    libraries = [
        "-Wl,--start-group",
        "-lavformat", "-lavcodec", "-lswscale", "-lswresample", "-lavutil",
        "-Wl,--end-group",
        "-lm", "-latomic", "-lpthread", "-lbcrypt",
        "d3d11", "dxgi", "d3dcompiler_47", "dxguid", "xinput",
        "winmm", "imm32", "xaudio2_8", "mfplat", "mfreadwrite",
        "mfuuid", "ole32", "dbghelp", "user32", "gdi32", "advapi32",
        "shell32", "uuid",
    ]
    run([
        compiler, f"--target={TARGET}", f"-L{FFMPEG_PREFIX / 'lib'}",
        f"@{response}", "-o", str(EXE),
        "-static", "-static-libgcc", "-static-libstdc++", "-mwindows",
        "-Wl,--gc-sections",
        *(arg if arg.startswith("-") else f"-l{arg}" for arg in libraries),
    ])
    smaa_dir = BUILD / "smaa"
    smaa_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "src" / "third_party" / "smaa" / "SMAA.hlsl",
                 smaa_dir / "SMAA.hlsl")
    print(f"Built {EXE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
