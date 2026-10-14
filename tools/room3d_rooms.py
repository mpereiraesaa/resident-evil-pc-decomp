"""Catalog unique room geometry versus scenario variants with review and gameplay status.

RDT files come in scenario pairs (ROOM1180 / ROOM1181). Rooms that share the
measured layout and background images can share one override file; rooms that
differ need their own corrections. Review status is tracked separately from
gameplay validation, because a visually approved room is not a tested room.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

from room3d_overrides import geometry_sha256, room_identity
from survey_room3d import survey

SCHEMA = 1
REVIEW_STATES = ('unreviewed', 'needs_work', 'visually_approved')
GAMEPLAY_STATES = ('not_validated', 'validated', 'broken')
LABELS = {'unreviewed': 'Sin revisar', 'needs_work': 'Necesita trabajo', 'visually_approved': 'Aprobada visualmente',
          'not_validated': 'Sin validar en juego', 'validated': 'Validada en juego', 'broken': 'Rota en juego'}


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def background_hashes(rdt, cameras):
    """Hash the original background files the batcher would project, by camera index."""
    stage = int(rdt.stem[4], 16)
    background_stage = stage - 5 if stage > 5 else stage
    folder = next((p for p in rdt.parent.parent.iterdir() if p.name.lower() == f'stage{background_stage}'), None)
    if folder is None:
        return {}
    files = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    prefix = f'rc{background_stage}{rdt.stem[5:-1].lower()}'
    result = {}
    for camera in cameras:
        path = files.get(prefix + f"{camera['index']:x}" + '.pak')
        if path is not None:
            result[str(camera['index'])] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def inventory(assets):
    """Per-RDT identity and content fingerprints, read directly from the game files."""
    records = []
    for rdt in sorted(p for p in Path(assets).rglob('*') if p.suffix.lower() == '.rdt'):
        record = room_identity(rdt.stem)
        if rdt.read_bytes() == bytes(4):
            records.append({**record, 'empty_slot': True})
            continue
        room = survey(rdt)
        records.append({**record, 'empty_slot': False, 'rdt_sha256': room['sha256'],
                        'geometry_sha256': geometry_sha256(room), 'cameras': len(room['cameras']),
                        'backgrounds': background_hashes(rdt, room['cameras'])})
    return records


def group_variants(records):
    """Annotate each record with the scenario variants that share its layout or images."""
    by_room, by_layout = {}, {}
    for record in records:
        by_room.setdefault((record['stage'], record['room']), []).append(record)
        if not record['empty_slot']:
            by_layout.setdefault((record['geometry_sha256'], json.dumps(record['backgrounds'], sort_keys=True)),
                                 []).append(record['rdt'])
    for record in records:
        if not record['empty_slot']:
            key = (record['geometry_sha256'], json.dumps(record['backgrounds'], sort_keys=True))
            record['shared_layout_with'] = [name for name in by_layout[key] if name != record['rdt']]
    for members in by_room.values():
        for record in members:
            record['variant_group'] = f"{record['stage']}-{record['room']:02X}"
            if record['empty_slot']:
                continue
            others = [m for m in members if m is not record and not m['empty_slot']]
            record['same_geometry'] = [m['rdt'] for m in others if m['geometry_sha256'] == record['geometry_sha256']]
            record['same_backgrounds'] = [m['rdt'] for m in others if m['backgrounds'] == record['backgrounds']]
            record['same_rdt'] = [m['rdt'] for m in others if m['rdt_sha256'] == record['rdt_sha256']]
    return records


def load_status(path):
    status = read_json(path, {'schema': SCHEMA, 'rooms': {}})
    if status.get('schema') != SCHEMA or not isinstance(status.get('rooms'), dict):
        raise ValueError('Unsupported status file')
    for name, entry in status['rooms'].items():
        room_identity(name)
        if entry.get('review', 'unreviewed') not in REVIEW_STATES or entry.get('gameplay', 'not_validated') not in GAMEPLAY_STATES:
            raise ValueError(f'{name}: unknown review or gameplay status')
    return status


def set_status(path, room, review=None, gameplay=None, note=None):
    """Update one room's tracked status; unknown states are rejected before writing."""
    import datetime

    status = load_status(path)
    name = room_identity(room)['rdt']
    entry = status['rooms'].setdefault(name, {'review': 'unreviewed', 'gameplay': 'not_validated'})
    if review is not None:
        if review not in REVIEW_STATES:
            raise ValueError('Review status must be one of ' + ', '.join(REVIEW_STATES))
        entry['review'] = review
    if gameplay is not None:
        if gameplay not in GAMEPLAY_STATES:
            raise ValueError('Gameplay status must be one of ' + ', '.join(GAMEPLAY_STATES))
        entry['gameplay'] = gameplay
    if note is not None:
        entry['note'] = note
    entry['updated'] = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    Path(path).write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    return entry


def propagate(assets, status_path):
    """Give unreviewed rooms the review status of their reviewed same-layout siblings.

    Only rooms whose layout and background files are identical to a reviewed
    room inherit, and only when every reviewed sibling agrees. The note names
    the source so an inherited status is never mistaken for a separate review;
    gameplay validation is never inherited.
    """
    import datetime

    status = load_status(status_path)
    records = {r['rdt']: r for r in group_variants(inventory(assets))}
    changed = {}
    for name, record in sorted(records.items()):
        current = status['rooms'].get(name, {})
        if record['empty_slot'] or (current.get('review', 'unreviewed') != 'unreviewed'
                                    and 'inherited_from' not in current):
            continue
        reviewed = {other: status['rooms'][other] for other in record['shared_layout_with']
                    if status['rooms'].get(other, {}).get('review', 'unreviewed') != 'unreviewed'
                    and 'inherited_from' not in status['rooms'][other]}
        if not reviewed or len({entry['review'] for entry in reviewed.values()}) != 1:
            continue
        source, entry = sorted(reviewed.items())[0]
        if current.get('inherited_from') == source and current.get('review') == entry['review'] \
                and current.get('note', '').endswith(entry.get('note', '')):
            continue
        changed[name] = {'review': entry['review'], 'gameplay': 'not_validated', 'inherited_from': source,
                         'note': f"Inherited from {source} (identical layout and background files); not reviewed separately. "
                                 + entry.get('note', ''),
                         'updated': datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}
    status['rooms'].update(changed)
    Path(status_path).write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    return changed


def build(assets, out, overrides=None, status_path=None, batch=None):
    """Write rooms.json and an HTML table; never inside the assets or batch output."""
    out = Path(out)
    if out.resolve().is_relative_to(Path(assets).resolve()) or (batch and out.resolve().is_relative_to(Path(batch).resolve())):
        raise ValueError('Catalog must be outside input assets and batch output')
    records = group_variants(inventory(assets))
    status = load_status(status_path) if status_path else {'schema': SCHEMA, 'rooms': {}}
    manifest = {item['room'].upper(): item for item in read_json(Path(batch) / 'manifest.json', {}).get('results', [])} if batch else {}
    counts = {'rdt_files': len(records), 'empty_slots': sum(r['empty_slot'] for r in records),
              'unique_geometry': len({r['geometry_sha256'] for r in records if not r['empty_slot']}),
              'variants_with_same_geometry': sum(bool(r.get('same_geometry')) for r in records),
              'variants_with_same_geometry_and_backgrounds': sum(
                  bool(set(r.get('same_geometry', [])) & set(r.get('same_backgrounds', []))) for r in records),
              'unique_layout_and_backgrounds': len({(r['geometry_sha256'], json.dumps(r['backgrounds'], sort_keys=True))
                                                    for r in records if not r['empty_slot']}),
              'overrides': 0, 'review': dict.fromkeys(REVIEW_STATES, 0), 'gameplay': dict.fromkeys(GAMEPLAY_STATES, 0),
              'inherited_review': 0}
    rows = []
    for record in records:
        name = record['rdt']
        entry = status['rooms'].get(name, {})
        record['review'] = entry.get('review', 'unreviewed')
        record['gameplay'] = entry.get('gameplay', 'not_validated')
        record['note'] = entry.get('note', '')
        if not record['empty_slot']:
            counts['review'][record['review']] += 1
            counts['gameplay'][record['gameplay']] += 1
            counts['inherited_review'] += 'inherited_from' in entry
            if 'inherited_from' in entry:
                record['inherited_from'] = entry['inherited_from']
        override_path = Path(overrides) / (name + '.json') if overrides else None
        if override_path and not override_path.exists() and not record['empty_slot']:
            shared = [other for other in record['shared_layout_with'] if (Path(overrides) / (other + '.json')).exists()]
            if len(shared) == 1:
                override_path = Path(overrides) / (shared[0] + '.json')
                record['overrides_shared_from'] = shared[0]
        if override_path and override_path.exists():
            data = read_json(override_path, {})
            record['overrides'] = {'steps': len(data.get('history', [])), 'updated': data.get('updated', data.get('created')),
                                   'appearance': data.get('appearance', {}).get('mode')}
            counts['overrides'] += 1
        if name in manifest:
            record['export'] = manifest[name].get('status')
            corrections = manifest[name].get('corrections')
            if corrections:
                record['corrections'] = corrections.get('status')
        if batch:
            replay = read_json(Path(batch) / 'corrected' / name.lower() / 'replay.json', None)
            if replay:
                record['corrections'] = replay.get('status')
        links, notes = [], []
        if batch and (Path(batch) / name.lower() / 'index.html').exists():
            links.append(f'<a href="{html.escape(os.path.relpath(Path(batch) / name.lower() / "index.html", out), quote=True)}">Automática</a>')
        if batch and (Path(batch) / 'corrected' / name.lower() / 'index.html').exists():
            links.append(f'<a href="{html.escape(os.path.relpath(Path(batch) / "corrected" / name.lower() / "index.html", out), quote=True)}">Corregida</a>')
        if record['empty_slot']:
            notes.append('Espacio vacío')
        else:
            shared = set(record['same_geometry']) & set(record['same_backgrounds'])
            if shared:
                notes.append('Misma geometría e imágenes que ' + ', '.join(sorted(shared)))
            elif record['same_geometry']:
                notes.append('Misma geometría, distintas imágenes que ' + ', '.join(sorted(record['same_geometry'])))
            elif record['variant_group'] and len([r for r in records if r['variant_group'] == record['variant_group']]) > 1:
                notes.append('Geometría distinta de su variante')
            cross = [other for other in record['shared_layout_with'] if other not in record['same_geometry']]
            if cross:
                notes.append('Misma disposición e imágenes que ' + ', '.join(sorted(cross)))
            if 'overrides' in record:
                notes.append(f"Correcciones: {record['overrides']['steps']} pasos"
                             + (f" (compartidas de {record['overrides_shared_from']})" if 'overrides_shared_from' in record else ''))
            if record.get('export'):
                notes.append('Exportación: ' + record['export'])
            if record.get('corrections'):
                notes.append('Réplica: ' + record['corrections'])
            if entry.get('inherited_from'):
                notes.append('Estado heredado de ' + entry['inherited_from'])
            if record['note']:
                notes.append(record['note'])
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{record['variant_group']}</td>"
                    f"<td>{LABELS[record['review']] if not record['empty_slot'] else ''}</td>"
                    f"<td>{LABELS[record['gameplay']] if not record['empty_slot'] else ''}</td>"
                    f"<td>{' · '.join(links)}</td><td>{html.escape('. '.join(notes))}</td></tr>")
    state = {'schema': SCHEMA, 'counts': counts, 'rooms': records,
             'qualification': 'Shared geometry means identical measured layout; it is not a fidelity or playability claim'}
    out.mkdir(parents=True, exist_ok=True)
    page = ('<!doctype html><html lang="es"><meta charset="utf-8"><title>Habitaciones y variantes</title>'
            '<style>body{background:#171b22;color:#eee;font:16px system-ui;margin:24px}a{color:#8cf}'
            'table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #455;padding:8px;text-align:left}</style>'
            '<h1>Habitaciones y variantes de escenario</h1>'
            '<p>La revisión visual y la validación en juego se registran por separado. '
            'Las variantes con la misma geometría e imágenes pueden compartir un archivo de correcciones; '
            'la réplica comprueba las huellas antes de aplicarlas.</p>'
            '<pre>' + html.escape(json.dumps(counts, indent=2)) + '</pre>'
            '<table><tr><th>RDT</th><th>Grupo</th><th>Revisión</th><th>Juego</th><th>Enlaces</th><th>Notas</th></tr>'
            + ''.join(rows) + '</table></html>')
    for filename, contents in (('rooms.json', json.dumps(state, indent=2) + '\n'), ('index.html', page)):
        temporary = out / (filename + '.tmp')
        temporary.write_text(contents)
        temporary.replace(out / filename)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    catalog = commands.add_parser('catalog', help='Write rooms.json and index.html')
    catalog.add_argument('--assets', type=Path, required=True, help='USA asset directory')
    catalog.add_argument('--out', type=Path, required=True)
    catalog.add_argument('--overrides', type=Path, help='Directory of portable override files')
    catalog.add_argument('--status', type=Path, help='Tracked status.json')
    catalog.add_argument('--batch', type=Path, help='Batch output root, for export and replay status')
    status = commands.add_parser('status', help='Set the review or gameplay status of a room')
    status.add_argument('--status', type=Path, required=True)
    status.add_argument('room')
    status.add_argument('--review', choices=REVIEW_STATES)
    status.add_argument('--gameplay', choices=GAMEPLAY_STATES)
    status.add_argument('--note')
    inherit = commands.add_parser('propagate', help='Copy review status to unreviewed same-layout siblings')
    inherit.add_argument('--assets', type=Path, required=True)
    inherit.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'catalog':
        state = build(args.assets, args.out, args.overrides, args.status, args.batch)
        print(json.dumps(state['counts'], indent=2))
        print(args.out / 'index.html')
    elif args.command == 'propagate':
        changed = propagate(args.assets, args.status)
        print(json.dumps({name: entry['inherited_from'] for name, entry in changed.items()}, indent=2))
    else:
        print(json.dumps(set_status(args.status, args.room, args.review, args.gameplay, args.note), indent=2))


if __name__ == '__main__':
    main()
