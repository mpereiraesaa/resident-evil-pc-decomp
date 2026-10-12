"""Store explicit visual findings outside exports and bind them to a checkpoint."""

import argparse
import json
from pathlib import Path

from room3d_checkpoint import digest, restore, signature

STATUSES = ('needs_further_geometry_work', 'approximate_visual_pass')


def audit_path(audits, room):
    return audits / f'{room.name}-{signature(str(room.resolve()))[:12]}.json'


def record_audit(room, audits, status, cameras, findings, reviewer):
    if audits.resolve().is_relative_to(room.parent.resolve()):
        raise ValueError('Keep visual audits outside the input run')
    if (status not in STATUSES or not findings or not reviewer.strip()
            or any(not isinstance(s, str) or not s.strip() for s in findings)):
        raise ValueError('Provide a supported status, reviewer and explicit findings')
    checkpoint = room.parent/'checkpoints'/(room.name+'.json')
    before = digest(checkpoint)
    record = json.loads(checkpoint.read_text())
    if record['result'].get('status') != 'image_assisted' or not restore(checkpoint, record['key'], room):
        raise ValueError('Visual review requires an unchanged completed image-assisted export')
    available = {c['index'] for c in json.loads((room/'survey.json').read_text())['cameras']}
    if (not cameras or len(set(cameras)) != len(cameras)
            or any(type(c) is not int for c in cameras) or not set(cameras) <= available):
        raise ValueError('Review existing, distinct camera indices')
    views = []
    for camera in cameras:
        names = [f'camera{camera}-original.png', f'camera{camera}-assisted.png']
        if any(name not in record['artifacts'] for name in names):
            raise ValueError('Reviewed original/rendered views must belong to the completed checkpoint')
        views.append({'camera': camera, 'artifacts': {n: record['artifacts'][n] for n in names}})
    if digest(checkpoint) != before:
        raise ValueError('Export checkpoint changed during review registration')
    result = {'schema': 1, 'source': str(room.resolve()), 'checkpoint_sha256': before,
              'checkpoint_key': record['key'], 'status': status, 'reviewer': reviewer,
              'viewed_cameras': cameras, 'reviewed_views': views, 'findings': findings,
              'qualification': 'Explicit visual review of listed source views only; hidden geometry, fidelity percentage and navigation unverified'}
    target = audit_path(audits, room)
    if target.exists():
        raise ValueError('Audit already exists; choose a new audit directory to retain the old review')
    audits.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2)+'\n')
    return target


def load_audit(audits, room):
    path = audit_path(audits, room)
    if not path.exists():
        return None
    try:
        audit = json.loads(path.read_text())
        if not isinstance(audit, dict):
            raise TypeError('Expected a visual audit record')
        checkpoint = room.parent/'checkpoints'/(room.name+'.json')
        if (audit.get('schema') != 1 or audit.get('source') != str(room.resolve())
                or audit.get('status') not in STATUSES
                or audit.get('checkpoint_sha256') != digest(checkpoint)
                or not restore(checkpoint, audit.get('checkpoint_key'), room)):
            raise ValueError('Visual audit does not match the current complete export')
        if (not isinstance(audit.get('reviewed_views'), list) or not audit['reviewed_views']
                or not isinstance(audit.get('findings'), list) or not audit['findings']
                or any(not isinstance(s, str) or not s.strip() for s in audit['findings'])
                or not isinstance(audit.get('reviewer'), str) or not audit['reviewer'].strip()):
            raise ValueError('Visual audit is missing viewed images or explicit findings')
        record = json.loads(checkpoint.read_text())
        cameras = [view['camera'] for view in audit['reviewed_views']]
        available = {c['index'] for c in json.loads((room/'survey.json').read_text())['cameras']}
        if (cameras != audit.get('viewed_cameras') or len(set(cameras)) != len(cameras)
                or any(type(c) is not int for c in cameras) or not set(cameras) <= available):
            raise ValueError('Visual audit camera metadata is inconsistent')
        for view in audit['reviewed_views']:
            names = {f"camera{view['camera']}-original.png", f"camera{view['camera']}-assisted.png"}
            if (set(view['artifacts']) != names
                    or any(record['artifacts'].get(n) != h for n, h in view['artifacts'].items())):
                raise ValueError('Reviewed images do not match the current export')
        if digest(checkpoint) != audit['checkpoint_sha256']:
            raise ValueError('Export checkpoint changed during visual audit verification')
        return {**audit, 'binding': 'complete_export_verified'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'status': 'stale_or_invalid_audit', 'findings': [str(error)],
                'binding': 'not_verified', 'qualification': 'No current visual approval'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', type=Path, required=True)
    parser.add_argument('--audits', type=Path, required=True)
    parser.add_argument('--status', choices=STATUSES, required=True)
    parser.add_argument('--camera', type=int, action='append', required=True)
    parser.add_argument('--finding', action='append', required=True)
    parser.add_argument('--reviewer', required=True)
    args = parser.parse_args()
    print(record_audit(args.room, args.audits, args.status, args.camera, args.finding, args.reviewer))


if __name__ == '__main__':
    main()
