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

The importer copies the voices and MP4s and maps the Japanese movie names to
their USA counterparts. It excludes the custom executable, `ddraw.dll`,
`dinput8.dll`, `bio1hd.asi`, saves and the replacement sound tree. Those hooks
target another executable and are unnecessary for source-native overrides.

The pack's `hires/*.webp` files use names derived from the ASI plugin's texture
hashes. They require a native hash lookup and WebP upload path before this build
can use them; the importer intentionally leaves them out. Spanish menu and game
text embedded by the replacement executable/DLL also needs a separate native
text-data port.
