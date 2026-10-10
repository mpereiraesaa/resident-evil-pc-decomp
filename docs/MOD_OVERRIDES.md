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
mods/latino/hires/misc/97DE3E43.webp
mods/latino/hires/skin/028F82B6.webp
mods/latino/hires/skin2/018087D8.webp
mods/latino/libwebp.dll
mods/latino/data/fontus.tim
mods/latino/translation/global.bin
mods/latino/translation/room/room_100_0.bin
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

The importer also preserves the pack's hash-addressed `misc`, `skin` and
`skin2` pages. The source build computes the same CRC32 identifiers as the HD
plugin from the logical DirectDraw texture: RGB555 channels use the original
three-bit expansion, pixels are signed as BGRA, and black is the transparent
colour key. A matching WebP replaces the D3D11 SRV while its logical width,
height and UV coordinates stay unchanged. For example, USA GOG's
`Data/STATUS.TIM` resolves to `hires/misc/97DE3E43.webp`, the Spanish HD menu
sheet. Missing hashes fall back to the original TIM without changing startup
or requiring `bio1hd.asi`.

The Spanish translation is stored as a 7z resource inside the pack's
`ddraw.dll`. The importer locates that resource, copies its 256x256 extended
font as the mod's `data/fontus.tim`, and compiles 2,664 translated strings into
275 compact binary banks. The proportional 14x14 source glyphs are repacked
into the USA renderer's 8x14 cells, and its three CLUT rows are reduced to the
single grayscale row the renderer expects. Colour remains supplied by the
engine's existing text tints. The banks cover global messages, item names,
descriptions, and both character variants of 272 rooms. The original XML stays
outside the runtime tree and the hook DLL itself is never installed. The
native game-side bank loader is the component that consumes these files.

The custom executable, `ddraw.dll`, `dinput8.dll`, `bio1hd.asi`, saves and the
replacement sound tree remain excluded. Those hooks target another executable
and the source build handles supported assets directly. Use `--no-hd` to skip
room art, `--no-translation` to skip Spanish text, or `--voices-only` for
voices alone.
