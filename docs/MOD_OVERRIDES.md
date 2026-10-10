# Asset mod overrides

`config.ini` can compose user-owned content without replacing the base GOG
files or loading hook DLLs. Add semicolon-separated mod roots to `[Assets]`:

```ini
[Assets]
Path=../../assets
Version=USA
Mods=../../mods/latino;../../mods/my-textures
```

Relative paths start at the game executable. The first mod listed has highest
priority. Resolution then checks later mods, the active content-mode overlay
(for example `DC/`), and finally the selected `USA/` or `JPN/` tree.

Each mod mirrors the path below the region folder. Typical files are:

```text
mods/latino/voice/V001_00.WAV
mods/latino/movie/PU.mp4
mods/latino/Stage1/RC1000.webp
mods/latino/Objspr/OSP00000.webp
mods/latino/libwebp.dll
mods/my-textures/data/title.pix
```

The video player automatically prefers an `.mp4` override beside a legacy
`.avi` name. File and directory case is resolved on Linux/Wine as well as on
Windows.

## Latino/HD pack importer

The Latino/HD pack is arranged for the Japanese/Sourcenext release. Its 563
voice filenames correspond one-for-one with the USA GOG voice tree, while a
few Japanese movie names differ. Import the compatible payload with:

```bash
python3 tools/import_latino_pack.py \
  "/path/to/RE1 Voces Latinoamericano Y Graficos HD.7z" \
  mods/latino --region USA
```

For `build/windows-cross/residentevil.exe`, enable it with:

```ini
Mods=../../mods/latino
```

The importer copies the voices and MP4s, maps the Japanese movie names to their
USA counterparts, and rewrites compatible hash-named backgrounds to stable
`StageN/RCxxxx.webp` names. The committed map contains 598 high-confidence
matches against the 676 USA GOG room PAKs. Unmapped cameras keep their original
background instead of risking art from another room. Occlusion pages use the
same process: 377 of 436 USA mask PAKs have high-confidence 1024x1024 RGBA
replacements. Their logical UV size remains 256x256, so existing sprite depth
and room geometry continue to select the correct part of the higher-resolution
texture. The imported 32-bit `libwebp.dll` decodes these files in the Win32
build; Linux uses its system libwebp when available.

The custom executable, `ddraw.dll`, `dinput8.dll`, `bio1hd.asi`, saves and the
replacement sound tree remain excluded. Those hooks target another executable
and the source build now handles supported assets directly. Use `--no-hd` to
import only voices and movies, or `--voices-only` for voices alone. Other HD
texture classes (`misc`, `skin`, `skin2`) and Spanish menu/game text embedded by
the replacement executable/DLL remain separate native ports.
