"""Stage independently named room assets for Room3D.cpp; never install into a live game."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

ROOM = re.compile(r'room([1-7])[0-9a-f]{2}[01]$')


def dependency(root, name):
    if not name or any(c.isspace() for c in name) or '\\' in name:
        raise ValueError('Only space-free generated asset paths are supported')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or non-local dependency: {name}')
    return path


def room_files(source, name):
    match = ROOM.fullmatch(name)
    if not match:
        raise ValueError('Expected room name such as room1010')
    mesh = source / 'assisted.obj'
    text = mesh.read_text()
    vertices = uvs = triangles = 0
    libraries = {}
    materials = set()
    plan = {}
    lines = []
    for line in text.splitlines():
        words = line.split('#', 1)[0].split()
        if not words:
            lines.append(line)
            continue
        tag = words[0]
        vertices += tag == 'v'
        uvs += tag == 'vt'
        if tag == 'f':
            triangles += len(words) - 3
        if tag == 'mtllib':
            if len(words) != 2:
                raise ValueError('Expected one material library per mtllib line')
            original = words[1]
            if original not in libraries:
                libraries[original] = f'{name}/material{len(libraries)}.mtl'
            line = 'mtllib ' + libraries[original]
        lines.append(line)
    if vertices > 100000 or uvs > 100000 or triangles > 50000:
        raise ValueError('Mesh exceeds Room3D.cpp vertex/UV/triangle limits')
    payload = ('\n'.join(lines) + '\n').encode()
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError('OBJ exceeds Room3D.cpp 4 MiB limit')
    plan[name + '.obj'] = payload
    textures = {}
    for original, target in libraries.items():
        rows = []
        material = dependency(source, original)
        for line in material.read_text().splitlines():
            words = line.split('#', 1)[0].split()
            if words and words[0] == 'newmtl':
                if len(words) != 2 or len(words[1].encode()) >= 64:
                    raise ValueError('Material name exceeds runtime limits')
                materials.add(words[1])
            if words and words[0] == 'map_Kd':
                if len(words) != 2:
                    raise ValueError('Only a plain map_Kd PNG filename is supported')
                texture = dependency(source, words[1])
                if texture not in textures:
                    destination = f'{name}/texture{len(textures):03d}.png'
                    content = texture.read_bytes()
                    with Image.open(texture) as image:
                        if image.format != 'PNG' or image.width * image.height > 4096 * 4096:
                            raise ValueError('Texture exceeds runtime format/pixel limits')
                        image.load()
                    if len(content) > 16 * 1024 * 1024:
                        raise ValueError('Texture exceeds runtime 16 MiB limit')
                    textures[texture] = destination
                    plan[destination] = content
                # Runtime and offline preview resolve map_Kd from the OBJ directory.
                line = 'map_Kd ' + textures[texture]
            rows.append(line)
        payload = ('\n'.join(rows) + '\n').encode()
        if len(payload) > 1024 * 1024:
            raise ValueError('Material library exceeds runtime 1 MiB limit')
        plan[target] = payload
    if len(materials) > 128:
        raise ValueError('Material count exceeds runtime limit')
    sidecar = mesh.with_suffix('.cameras')
    if sidecar.exists():
        plan[name + '.cameras'] = sidecar.read_bytes()
    stage = 'Stage' + match.group(1)
    return {f'{stage}/{path}': payload for path, payload in plan.items()}, {
        'room': name, 'source': str(source.resolve()), 'vertices': vertices,
        'triangles': triangles, 'textures': len(textures), 'materials': len(materials)}


def package(batch, out, rooms=None, overrides=None):
    if out.exists():
        raise ValueError('Choose a new package directory; existing packages are preserved')
    if out.resolve().is_relative_to(batch.resolve()):
        raise ValueError('Package must be outside the batch output')
    overrides = overrides or {}
    available = {}
    for checkpoint in (batch / 'checkpoints').glob('room*.json'):
        record = json.loads(checkpoint.read_text())
        result = record.get('result', {})
        if result.get('status') == 'image_assisted':
            available[checkpoint.stem] = batch / checkpoint.stem
    selected = sorted(set(rooms) if rooms else available.keys())
    missing = set(selected) - available.keys()
    if missing or set(overrides) - set(selected):
        raise ValueError('Requested rooms or overrides are not completed selected exports')
    if not selected:
        raise ValueError('No completed image-assisted rooms available')
    for name in selected:
        if out.resolve().is_relative_to(overrides.get(name, available[name]).resolve()):
            raise ValueError('Package must not be inside a source room')
    results = []
    out.mkdir(parents=True)
    for name in selected:
        try:
            source = overrides.get(name, available[name])
            files, report = room_files(source, name)
            for relative, payload in files.items():
                target = out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            report['files'] = {relative: hashlib.sha256(payload).hexdigest() for relative, payload in files.items()}
            results.append(dict(report, status='packaged'))
        except (OSError, ValueError) as error:
            results.append({'room': name, 'status': 'error', 'error': str(error)})
    manifest = {'results': results, 'qualification': 'Staged asset bundle; not installed or play-tested'}
    (out / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rooms', nargs='+')
    parser.add_argument('--override', action='append', default=[], metavar='ROOM=FOLDER',
                        help='Use a manually retouched variant of a completed room')
    args = parser.parse_args()
    overrides = {}
    for item in args.override:
        name, separator, folder = item.partition('=')
        if not separator or not folder:
            parser.error('--override requires ROOM=FOLDER')
        overrides[name.lower()] = Path(folder)
    report = package(args.batch, args.out, [s.lower() for s in args.rooms] if args.rooms else None, overrides)
    for result in report['results']:
        print(result['room'], result['status'], result.get('error', ''))
    return int(any(result['status'] == 'error' for result in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
