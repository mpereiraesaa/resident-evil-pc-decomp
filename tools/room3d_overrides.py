"""Portable, versioned manual room corrections replayed onto fresh automatic exports.

An override file identifies a room by RDT name and content fingerprints, names
each corrected object by kind, script anchor and bounds instead of by the
generated group index, and keeps an ordered history of transform and addition
steps. Replay always starts from a fresh automatic export, so corrections never
accumulate twice. An incompatible source or an object that cannot be resolved
uniquely is reported as a conflict; a correction is never silently retargeted.
Files contain no machine-local paths and no original game assets.
"""

import argparse
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from room3d_additions import append_objects
from room3d_checkpoint import digest
from room3d_preview import render
from room3d_retouch import apply_edits, catalog, materialize, prepare, read_mesh, transforms

SCHEMA = 1
TOOL = 'room3d_overrides'
DEFAULT_TOLERANCE = 50.0
NAMED_PREFIXES = ('boundary_', 'added_')
NAMED_GROUPS = ('floor',)


class Conflict(ValueError):
    """A correction cannot be applied to this export without guessing."""

    def __init__(self, kind, message, **details):
        super().__init__(message)
        self.record = {'conflict': kind, 'message': message, **details}


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def room_identity(name):
    """Stage, room number and scenario player from an RDT stem such as ROOM1180."""
    stem = Path(name).stem.upper()
    if not re.fullmatch(r'ROOM[1-7][0-9A-F]{2}[01]', stem):
        raise ValueError('Expected an RDT stem such as ROOM1180, got ' + repr(name))
    return {'rdt': stem, 'stage': int(stem[4], 16), 'room': int(stem[5:7], 16), 'player': int(stem[7])}


def geometry_sha256(survey):
    """Fingerprint of the measured camera and collision layout, independent of scripts."""
    return canonical_sha256({'cameras': survey['cameras'], 'boundaries': survey['boundaries'],
                             'quadrant_split': survey.get('quadrant_split')})


def tools_sha256():
    folder = Path(__file__).parent
    return canonical_sha256({path.name: digest(path) for path in sorted(folder.glob('*.py'))})


def source_fingerprints(room):
    """Content identity of the export's inputs: RDT, layout and background images."""
    room = Path(room)
    replay = room / 'replay.json'
    if replay.exists():
        return read_json(replay)['source']
    survey = read_json(room / 'survey.json')
    result = read_json(room / 'result.json') if (room / 'result.json').exists() else {}
    backgrounds = {str(c['index']): c['background_sha256'] for c in result.get('cameras', [])
                   if 'background_sha256' in c}
    if len(backgrounds) != len(survey['cameras']):
        raise ValueError('Every camera needs a recorded background hash; export the room with the batcher')
    return {**room_identity(survey['source']), 'rdt_sha256': survey['sha256'],
            'geometry_sha256': geometry_sha256(survey), 'backgrounds': backgrounds}


def generator_record(room):
    """Frozen generator identity: code digest, boolean recipe settings and the baseline mesh."""
    room = Path(room)
    recipe = {}
    if (room / 'assisted.json').exists():
        recipe = read_json(room / 'assisted.json').get('fitted_recipe', {})
    return {'tool': TOOL, 'schema': SCHEMA, 'tools_sha256': tools_sha256(),
            'settings': {key: value for key, value in sorted(recipe.items()) if isinstance(value, bool)},
            'assisted_obj_sha256': digest(room / 'assisted.obj')}


def group_kinds(room, mesh):
    """Semantic kind per OBJ group, following the batcher's naming, plus script anchors."""
    room = Path(room)
    recipe = {}
    if (room / 'assisted.json').exists():
        recipe = read_json(room / 'assisted.json').get('fitted_recipe', {})
    kinds, anchors = {}, {}
    for i, item in enumerate(recipe.get('primitives', [])):
        kinds[f'primitive_{i}'] = item['kind']
        if item.get('source_offset') is not None:
            anchors[f'primitive_{i}'] = int(item['source_offset'])
    kinds.update({s['group']: s['material'] for s in recipe.get('structural_surfaces', [])
                  if s['material'] in ('wall', 'floor', 'ceiling')})
    kinds['floor'] = 'floor'
    for context_name in ('retouch-appearance.json',):
        if (room / context_name).exists():
            kinds.update({item['group']: item['kind'] for item in read_json(room / context_name)['added_groups']})
    if (room / 'replay.json').exists():
        replayed = read_json(room / 'replay.json')
        kinds.update(replayed.get('kinds', {}))
        anchors.update({group: int(value) for group, value in replayed.get('anchors', {}).items()})
    for group in {face[0] for face in mesh[2]}:
        kinds.setdefault(group, 'wall' if group.startswith('boundary_') else 'unknown')
    return kinds, anchors


def identities(mesh, kinds, anchors):
    bounds = catalog(mesh)
    result = {}
    for group, box in bounds.items():
        record = {'kind': kinds.get(group, 'unknown'), 'min': box['min'], 'max': box['max']}
        if group in anchors:
            record['source_offset'] = anchors[group]
        result[group] = record
    return result


def object_identities(room):
    mesh = read_mesh(Path(room) / 'assisted.obj')
    kinds, anchors = group_kinds(room, mesh)
    return identities(mesh, kinds, anchors)


def bounds_delta(a, b):
    return float(max(np.abs(np.array(a['min']) - b['min']).max(), np.abs(np.array(a['max']) - b['max']).max()))


def named_group(group):
    return group in NAMED_GROUPS or group.startswith(NAMED_PREFIXES)


def resolve(group, identity, current, tolerance=DEFAULT_TOLERANCE, anchors=True):
    """Find the one current group that is the recorded object, or raise a Conflict.

    Script-anchored objects resolve by anchor; structural groups by their stable
    name plus unchanged bounds; everything else by kind and bounds within the
    tolerance. Zero or several candidates are conflicts, never a guess. Script
    offsets only identify objects within the same RDT, so ``anchors=False``
    makes anchored objects resolve by kind and bounds like the others.
    """
    kind = identity['kind']
    same_kind = {name: item for name, item in current.items() if item['kind'] == kind}
    if 'source_offset' in identity and anchors:
        anchored = [name for name, item in same_kind.items()
                    if item.get('source_offset') == identity['source_offset']]
        if len(anchored) > 1:
            raise Conflict('ambiguous_anchor', f'{group}: several {kind} objects share script offset '
                           f'{identity["source_offset"]}', group=group, candidates=anchored)
        if not anchored:
            raise Conflict('missing_object', f'{group}: no {kind} with script offset {identity["source_offset"]}',
                           group=group)
        delta = bounds_delta(identity, current[anchored[0]])
        return anchored[0], 'script_anchor', delta
    if named_group(group):
        if group not in current:
            raise Conflict('missing_object', f'{group}: structural group is absent', group=group)
        delta = bounds_delta(identity, current[group])
        if delta > tolerance:
            raise Conflict('changed_object', f'{group}: bounds moved by {delta:.1f} units', group=group,
                           delta=delta)
        return group, 'stable_name', delta
    matches = sorted((bounds_delta(identity, item), name) for name, item in same_kind.items())
    matches = [(delta, name) for delta, name in matches if delta <= tolerance]
    if len(matches) > 1:
        raise Conflict('ambiguous_object', f'{group}: {len(matches)} {kind} objects within {tolerance} units',
                       group=group, candidates=[name for _, name in matches])
    if not matches:
        raise Conflict('missing_object', f'{group}: no {kind} within {tolerance} units of the recorded bounds',
                       group=group)
    delta, name = matches[0]
    return name, 'kind_and_bounds', delta


def history_sha256(history):
    return canonical_sha256(history)


def initialize(room, note=None):
    """Start an override file for the room exported in ``room`` (an automatic batch output)."""
    source = source_fingerprints(room)
    record = {'schema': SCHEMA, 'tool': TOOL, 'created': now(),
              'room': {key: source[key] for key in ('rdt', 'stage', 'room', 'player')},
              'source': source, 'baseline': generator_record(room), 'history': [],
              'appearance': {'mode': 'original_uvs_preserved'},
              'qualification': 'Manual visual correction; gameplay collisions and scripts are unchanged'}
    if note:
        record['note'] = note
    return record


def validate(overrides):
    if not isinstance(overrides, dict) or overrides.get('schema') != SCHEMA or overrides.get('tool') != TOOL:
        raise ValueError('Unsupported override file')
    for key in ('room', 'source', 'baseline', 'history', 'appearance'):
        if key not in overrides:
            raise ValueError('Override file is missing ' + key)
    if overrides['appearance'].get('mode') not in ('original_uvs_preserved', 'reprojected'):
        raise ValueError('Unknown appearance mode')
    text = json.dumps(overrides)
    if re.search(r'(?<![A-Za-z0-9])(/home/|/tmp/|/mnt/|[A-Za-z]:\\\\)', text):
        raise ValueError('Override files must not contain machine-local paths')
    for index, step in enumerate(overrides['history'], 1):
        if step.get('step') != index or not isinstance(step.get('edits'), dict) \
                or not isinstance(step.get('additions'), list) or not isinstance(step.get('objects'), dict):
            raise ValueError(f'History step {index} is malformed')
        if set(step['edits']) - set(step['objects']):
            raise ValueError(f'History step {index} edits objects without recorded identity')
    return overrides


def base_state(overrides, room):
    """Check that ``room`` is the state the next step must be recorded against."""
    room = Path(room)
    replay = room / 'replay.json'
    if replay.exists():
        report = read_json(replay)
        if report.get('status') != 'applied' or report.get('history_sha256') != history_sha256(overrides['history']):
            raise ValueError('Record new steps against a replay of the current override history, '
                             'or against the automatic export when the history is empty')
        return 'replayed'
    if overrides['history']:
        raise ValueError('The history is not empty; record further steps against its replayed output, '
                         'not against the automatic export, so transforms do not accumulate twice')
    source = source_fingerprints(room)
    if source['rdt_sha256'] != overrides['source']['rdt_sha256']:
        raise ValueError('This export was generated from a different RDT than the override file')
    return 'automatic'


def add_step(overrides, room, edits=None, additions=None, note=None, tolerance=DEFAULT_TOLERANCE):
    """Append one correction step recorded against the current replayed state."""
    validate(overrides)
    edits = edits or {}
    additions = additions or []
    base = base_state(overrides, room)
    mesh = read_mesh(Path(room) / 'assisted.obj')
    kinds, anchors = group_kinds(room, mesh)
    mesh, generated = append_objects(mesh, json.loads(json.dumps(additions)))
    kinds.update({item['group']: item['kind'] for item in generated})
    transforms(edits, catalog(mesh))
    if not edits and not additions:
        raise ValueError('A step needs edits or additions')
    current = identities(mesh, kinds, anchors)
    objects = {group: current[group] for group in edits}
    for group, identity in objects.items():
        if 'source_offset' not in identity and not named_group(group):
            others = [name for name, item in current.items()
                      if name != group and item['kind'] == identity['kind']
                      and bounds_delta(identity, item) <= tolerance]
            if others:
                raise Conflict('ambiguous_object', f'{group}: cannot be told apart from {others} within '
                               f'{tolerance} units; replay could not resolve it', group=group, candidates=others)
    step = {'step': len(overrides['history']) + 1, 'recorded': now(), 'base': base,
            'base_sha256': digest(Path(room) / 'assisted.obj'),
            'objects': objects, 'edits': json.loads(json.dumps(edits)),
            'additions': json.loads(json.dumps(additions))}
    if note:
        step['note'] = note
    overrides['history'].append(step)
    overrides['updated'] = step['recorded']
    return step


def check_source(overrides, room):
    """Classify the export's inputs against the recorded source; raise on mismatch."""
    fresh = source_fingerprints(room)
    recorded = overrides['source']
    if fresh['rdt_sha256'] == recorded['rdt_sha256']:
        status = 'exact'
    elif (fresh['geometry_sha256'] == recorded['geometry_sha256']
          and (fresh['stage'], fresh['room']) == (recorded['stage'], recorded['room'])):
        status = 'scenario_variant'
    elif fresh['geometry_sha256'] == recorded['geometry_sha256'] and fresh['backgrounds'] == recorded['backgrounds']:
        # Stages 6 and 7 revisit stage 1 and 2 rooms with the same layout and images.
        status = 'layout_variant'
    else:
        raise Conflict('source_mismatch', f"{fresh['rdt']} does not share the recorded room layout of "
                       f"{recorded['rdt']}", recorded=recorded['rdt'], fresh=fresh['rdt'])
    return {'status': status, 'rdt': fresh['rdt'], 'recorded_rdt': recorded['rdt'],
            'backgrounds_match': fresh['backgrounds'] == recorded['backgrounds'],
            'baseline_mesh_match': digest(Path(room) / 'assisted.obj') == overrides['baseline']['assisted_obj_sha256'],
            'tools_match': tools_sha256() == overrides['baseline']['tools_sha256']}


def shifted_transform(transform, identity, resolved):
    """Keep an explicit pivot attached to the object when its bounds relocated."""
    if 'pivot' not in transform:
        return transform
    old_center = (np.array(identity['min']) + identity['max']) / 2
    new_center = (np.array(resolved['min']) + resolved['max']) / 2
    return {**transform, 'pivot': (np.array(transform['pivot'], float) + new_center - old_center).tolist()}


def replay(overrides, room, out, reproject=None, segmentation=None, depth=None,
           tolerance=DEFAULT_TOLERANCE, allow_partial=False):
    """Rebuild the corrected room from a fresh automatic export into a new directory."""
    validate(overrides)
    room, out = Path(room), Path(out)
    source = check_source(overrides, room)
    mesh, dependencies, survey, context = prepare(room, out)
    kinds, anchors = group_kinds(room, mesh)
    state = mesh
    generated_all, resolutions, conflicts = [], [], []
    for step in overrides['history']:
        state, generated = append_objects(state, json.loads(json.dumps(step['additions'])))
        generated_all.extend(generated)
        kinds.update({item['group']: item['kind'] for item in generated})
        current = identities(state, kinds, anchors)
        mapped = {}
        for group, transform in step['edits'].items():
            try:
                name, method, delta = resolve(group, step['objects'][group], current, tolerance,
                                              anchors=source['status'] == 'exact')
                if name in mapped:
                    raise Conflict('duplicate_target', f'{group}: {name} already receives an edit in step '
                                   f'{step["step"]}', group=group, candidates=[name])
            except Conflict as conflict:
                record = {**conflict.record, 'step': step['step']}
                conflicts.append(record)
                if not allow_partial:
                    report = {'schema': SCHEMA, 'status': 'conflict', 'source': source,
                              'resolutions': resolutions, 'conflicts': conflicts}
                    raise Conflict('replay_conflict', conflict.record['message'], report=report) from conflict
                continue
            mapped[name] = shifted_transform(transform, step['objects'][group], current[name])
            resolutions.append({'step': step['step'], 'group': group, 'resolved': name, 'method': method,
                                'bounds_delta': round(delta, 3)})
        state = apply_edits(state, mapped)
    if reproject is None:
        reproject = overrides['appearance']['mode'] == 'reprojected'
    history_digest = history_sha256(overrides['history'])
    record = {'source': 'automatic export; see replay.json', 'edits': 'replayed; see replay.json',
              'overrides_history_sha256': history_digest}
    materialize(room, out, state, generated_all, dependencies, context, survey, record,
                reproject, segmentation, depth)
    report = {'schema': SCHEMA, 'status': 'partial' if conflicts else 'applied', 'replayed': now(),
              'room': overrides['room'], 'source': {**overrides['source'], **{
                  'rdt': source['rdt'], 'rdt_sha256': source_fingerprints(room)['rdt_sha256'],
                  'backgrounds': source_fingerprints(room)['backgrounds']}},
              'source_check': source, 'history_sha256': history_digest, 'steps': len(overrides['history']),
              'resolutions': resolutions, 'conflicts': conflicts,
              'kinds': {group: kind for group, kind in kinds.items() if group.startswith(('primitive_', 'added_'))},
              'anchors': anchors, 'appearance': 'reprojected' if reproject else 'original_uvs_preserved',
              'assisted_obj_sha256': digest(out / 'assisted.obj'),
              'qualification': 'Replayed manual corrections; fidelity and gameplay remain unverified'}
    write_json(out / 'replay.json', report)
    write_json(out / 'overrides.json', overrides)
    return report


def compare(a, b, tolerance=1e-6):
    """Geometry and rendered-view agreement between two room directories."""
    a, b = Path(a), Path(b)
    mesh_a, mesh_b = read_mesh(a / 'assisted.obj'), read_mesh(b / 'assisted.obj')
    bounds_a, bounds_b = catalog(mesh_a), catalog(mesh_b)
    groups = sorted(set(bounds_a) | set(bounds_b))
    deltas = {group: (bounds_delta(bounds_a[group], bounds_b[group])
                      if group in bounds_a and group in bounds_b else None) for group in groups}
    survey = read_json(a / 'survey.json')
    views = []
    for camera in survey['cameras']:
        image_a, mask_a = render(a / 'assisted.obj', camera)
        image_b, mask_b = render(b / 'assisted.obj', camera)
        difference = np.abs(np.asarray(image_a, dtype=float) - np.asarray(image_b, dtype=float))
        views.append({'camera': camera['index'], 'mean_abs_difference': round(float(difference.mean()), 4),
                      'max_abs_difference': float(difference.max()),
                      'coverage_difference': round(float(np.abs(mask_a.mean() - mask_b.mean())), 6)})
    identical = (len(mesh_a[0]) == len(mesh_b[0]) and all(d is not None and d <= tolerance for d in deltas.values())
                 and np.allclose(np.array([v[:3] for v in mesh_a[0]]), np.array([v[:3] for v in mesh_b[0]]),
                                 atol=tolerance))
    return {'identical_geometry': bool(identical), 'identical_bytes': digest(a / 'assisted.obj') == digest(b / 'assisted.obj'),
            'vertex_counts': [len(mesh_a[0]), len(mesh_b[0])], 'group_bounds_delta': deltas, 'views': views,
            'qualification': 'Agreement between two exports; not a fidelity measure against the original room'}


def shared_override(overrides_root, room):
    """The one override file whose recorded layout and images match this export, if any."""
    try:
        fresh = source_fingerprints(room)
    except (OSError, ValueError, KeyError):
        return None
    matches = []
    for path in sorted(Path(overrides_root).glob('ROOM*.json')):
        try:
            source = read_json(path)['source']
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (source.get('geometry_sha256') == fresh['geometry_sha256']
                and source.get('backgrounds') == fresh['backgrounds']):
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


def replay_for_batch(overrides_root, batch_out, room_name, segmentation=None, depth=None):
    """Batch hook: replay ``<overrides_root>/<ROOM>.json`` into ``<batch_out>/corrected/<room>``.

    Without a file of its own, a room uses the single override whose recorded
    layout and background images match it exactly (a scenario or stage variant).
    """
    overrides_root, batch_out = Path(overrides_root), Path(batch_out)
    room = batch_out / room_name.lower()
    path = overrides_root / (room_name.upper() + '.json')
    shared = None
    if not path.exists():
        path = shared_override(overrides_root, room)
        if path is None:
            return {'status': 'no_overrides'}
        shared = path.stem
    out = batch_out / 'corrected' / room_name.lower()
    if out.exists():
        if not (out / 'replay.json').exists() and any(out.iterdir()):
            return {'status': 'error', 'error': f'{out} exists and is not a previous replay output'}
        shutil.rmtree(out)
    try:
        overrides = validate(read_json(path))
        report = replay(overrides, room, out,
                        segmentation=segmentation / room_name.lower() if segmentation else None,
                        depth=depth / room_name.lower() if depth else None)
    except Conflict as conflict:
        out.mkdir(parents=True, exist_ok=True)
        report = conflict.record.get('report', {'status': 'conflict', 'conflicts': [conflict.record]})
        write_json(out / 'replay.json', report)
        return {'status': 'conflict', 'report': f'corrected/{room_name.lower()}/replay.json',
                'conflicts': report.get('conflicts', [])}
    except (ValueError, OSError) as error:
        return {'status': 'error', 'error': str(error)}
    return {'status': report['status'], 'report': f'corrected/{room_name.lower()}/replay.json',
            'page': f'corrected/{room_name.lower()}/index.html', 'steps': report['steps'],
            'conflicts': report['conflicts'], 'source_check': report['source_check']['status'],
            **({'shared_from': shared} if shared else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create an override file for an automatic room export')
    init.add_argument('--room', type=Path, required=True, help='Automatic batch output directory of the room')
    init.add_argument('--out', type=Path, required=True, help='Override file to create, e.g. room3d/overrides/ROOM1180.json')
    init.add_argument('--note')
    add = commands.add_parser('add', help='Record a correction step from editor edits and/or additions')
    add.add_argument('--overrides', type=Path, required=True)
    add.add_argument('--room', type=Path, required=True,
                     help='The automatic export (first step) or the replay of the current history')
    add.add_argument('--edits', type=Path, help='JSON mapping group names to transforms, as the editor exports')
    add.add_argument('--additions', type=Path, help='JSON list of parametric objects to insert')
    add.add_argument('--note')
    add.add_argument('--reproject', action='store_true', help='Record that replays should rebake appearance')
    add.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE)
    play = commands.add_parser('replay', help='Rebuild the corrected room from a fresh automatic export')
    play.add_argument('--overrides', type=Path, required=True)
    play.add_argument('--room', type=Path, required=True, help='Fresh automatic export of the room')
    play.add_argument('--out', type=Path, required=True, help='New directory for the corrected room')
    play.add_argument('--reproject', action='store_true', help='Force appearance reprojection')
    play.add_argument('--segmentation', type=Path, help='Existing SAM directory for this room')
    play.add_argument('--depth', type=Path, help='Existing DA3 directory for this room')
    play.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE)
    play.add_argument('--allow-partial', action='store_true',
                      help='Apply resolvable edits and report the rest instead of failing')
    check = commands.add_parser('check', help='Report compatibility of an override file with an export')
    check.add_argument('--overrides', type=Path, required=True)
    check.add_argument('--room', type=Path, required=True)
    check.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE)
    diff = commands.add_parser('compare', help='Compare geometry and rendered views of two room directories')
    diff.add_argument('a', type=Path)
    diff.add_argument('b', type=Path)
    args = parser.parse_args()
    if args.command == 'init':
        if args.out.exists():
            parser.error('Override file already exists; use add')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, initialize(args.room, args.note))
        print(args.out)
    elif args.command == 'add':
        if not args.edits and not args.additions:
            parser.error('--edits and/or --additions are required')
        overrides = validate(read_json(args.overrides))
        step = add_step(overrides, args.room, read_json(args.edits) if args.edits else None,
                        read_json(args.additions) if args.additions else None, args.note, args.tolerance)
        if args.reproject:
            overrides['appearance']['mode'] = 'reprojected'
        write_json(args.overrides, overrides)
        print(f"{args.overrides}: step {step['step']} recorded against the {step['base']} export")
    elif args.command == 'replay':
        overrides = validate(read_json(args.overrides))
        try:
            report = replay(overrides, args.room, args.out, True if args.reproject else None,
                            args.segmentation, args.depth, args.tolerance, args.allow_partial)
        except Conflict as conflict:
            print(json.dumps(conflict.record.get('report', conflict.record), indent=2))
            raise SystemExit(2) from conflict
        print(json.dumps({key: report[key] for key in ('status', 'source_check', 'resolutions', 'conflicts')}, indent=2))
        print(args.out / 'index.html')
        raise SystemExit(0 if report['status'] == 'applied' else 2)
    elif args.command == 'check':
        overrides = validate(read_json(args.overrides))
        report = {'source_check': check_source(overrides, args.room), 'resolutions': [], 'conflicts': []}
        current = object_identities(args.room)
        for step in overrides['history']:
            for group, identity in step['objects'].items():
                if group.startswith('added_'):
                    continue
                try:
                    name, method, delta = resolve(group, identity, current, args.tolerance,
                                                  anchors=report['source_check']['status'] == 'exact')
                    report['resolutions'].append({'step': step['step'], 'group': group, 'resolved': name,
                                                  'method': method, 'bounds_delta': round(delta, 3)})
                except Conflict as conflict:
                    report['conflicts'].append({**conflict.record, 'step': step['step']})
        report['status'] = 'conflict' if report['conflicts'] else 'compatible'
        report['qualification'] = ('Objects of later steps are checked against the automatic export; '
                                   'a replay resolves them against the state after earlier steps')
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if not report['conflicts'] else 2)
    elif args.command == 'compare':
        print(json.dumps(compare(args.a, args.b), indent=2))


if __name__ == '__main__':
    main()
