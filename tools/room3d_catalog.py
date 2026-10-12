"""Build a room-by-room reconstruction catalog with separate export/review status."""

import argparse
import hashlib
import html
import json
import os
import time
from pathlib import Path

from room3d_checkpoint import restore
from room3d_confidence import failure_confidence
from room3d_editor import build as build_editor


def detail_failures(objects):
    """Group rejected proposals by category; gates are diagnostics, not fidelity scores."""
    categories = {}
    for obj in objects:
        if obj.get('accepted'):
            continue
        category = categories.setdefault(obj.get('kind', 'unknown'),
                                         {'rejected': 0, 'failed_gates': {}, 'without_failed_gate': 0})
        category['rejected'] += 1
        failed = [name for name, value in obj.get('gates', {}).items() if value is False]
        category['without_failed_gate'] += not failed
        for name in failed:
            category['failed_gates'][name] = category['failed_gates'].get(name, 0) + 1
    return categories


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def classify(rdt, batch):
    data = rdt.read_bytes()
    name = rdt.stem.lower()
    if data == bytes(4):
        return {"room": name, "status": "empty_slot"}
    folder = batch / name
    checkpoint_path = batch / 'checkpoints' / (name + '.json')
    checkpoint = read_json(checkpoint_path, {})
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('result', {}), dict):
        return {"room": name, "status": "incomplete_or_changed"}
    result = checkpoint.get('result', {})
    if result.get('source_sha256') != hashlib.sha256(data).hexdigest():
        return {"room": name, "status": "in_progress" if folder.exists() else "pending"}
    if result.get('status') != 'image_assisted':
        return {"room": name, "status": "baseline_only"}
    if restore(checkpoint_path, checkpoint.get('key'), folder) is None:
        return {"room": name, "status": "incomplete_or_changed"}
    artifacts = checkpoint.get('artifacts', {})
    for required in ('assisted.obj', 'survey.json', 'index.html', 'result.json'):
        path = folder / required
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifacts.get(required):
            return {"room": name, "status": "incomplete_or_changed"}
    if any('error' in camera for camera in result.get('cameras', [])):
        return {"room": name, "status": "incomplete_or_changed"}
    details = read_json(folder / 'details/results.json', {}).get('objects', [])
    accepted = [obj for obj in details if obj.get('accepted')]
    visibility = result.get('render_visibility', {'status': 'not_measured'})
    return {"room": name,
            "status": "exported_empty_geometry" if visibility.get('status') == 'empty_geometry' else 'exported',
            "render_visibility": visibility,
            "depth_scale": result.get('depth_scale', {'scale': 'not_recorded'}),
            "cameras": len(result.get('cameras', [])),
            "accepted_details": len(accepted),
            "creative_details": sum(obj.get('path') == 'procedural_interpretation' for obj in accepted),
            "single_view_details": sum(len(obj.get('observations', [])) < 2 for obj in accepted),
            "reserved_view_details": sum(bool(obj.get('heldout_cameras')) for obj in accepted),
            "rejected_details": len(details) - len(accepted),
            "detail_failures": detail_failures(details),
            'failure_model_confidence': failure_confidence(details),
            "review_status": result.get('visual_review', {}).get('status', 'see_room_report'),
            "export_key": checkpoint.get('key')}


def refresh(assets, batch, out, reviews=None):
    if out.resolve().is_relative_to(batch.resolve()) or out.resolve().is_relative_to(assets.resolve()):
        raise ValueError('Catalog must be outside input assets and batch output')
    out.mkdir(parents=True, exist_ok=True)
    records, rows, counts = [], [], {}
    previous = read_json(out / 'catalog.json', {}).get('rooms', [])
    old = {item['room']: item for item in previous}
    manifest = read_json(batch / 'manifest.json', {}).get('results', [])
    failures = {item['room'].lower(): item.get('error', '') for item in manifest if item.get('status') == 'error'}
    labels = {'empty_slot': 'Espacio vacío', 'pending': 'Pendiente', 'in_progress': 'Sin finalizar',
              'baseline_only': 'Solo malla básica', 'incomplete_or_changed': 'Incompleto o modificado',
              'exported': 'Exportada; calidad por revisar', 'error': 'Error'}
    labels['exported_empty_geometry'] = 'Exportada sin geometría visible; requiere corrección'
    for rdt in sorted(p for p in assets.rglob('*') if p.suffix.lower() == '.rdt'):
        record = classify(rdt, batch)
        name, status = record['room'], record['status']
        if status not in ('exported', 'exported_empty_geometry') and name in failures:
            status = record['status'] = 'error'
            record['error'] = failures[name]
        counts[status] = counts.get(status, 0) + 1
        links, notes = [], []
        if status in ('exported', 'exported_empty_geometry'):
            queue = read_json(batch / name / 'review-queue.json', {}).get('jobs', [])
            if reviews is not None and queue:
                reviewed = 0
                for job in queue:
                    review = read_json(reviews / name / f"camera{job['camera']}-review.json", {})
                    if (isinstance(review.get('response'), str)
                            and [entry.get('sha256') for entry in review.get('inputs', [])]
                            == [entry['sha256'] for entry in job['inputs']]):
                        reviewed += 1
                record['review_status'] = f'{reviewed}/{len(queue)} informes; propuestas sin verificar'
                review_page = reviews / name / 'index.html'
                if review_page.is_file():
                    link = os.path.relpath(review_page, out)
                    links.append(f'<a href="{html.escape(link, quote=True)}">Informe visual</a>')
            report = os.path.relpath(batch / name / 'index.html', out)
            links.append(f'<a href="{html.escape(report, quote=True)}">Comparar</a>')
            detail_page = batch / name / 'details/index.html'
            if detail_page.is_file():
                link = os.path.relpath(detail_page, out)
                links.append(f'<a href="{html.escape(link, quote=True)}">Diagnóstico de detalles</a>')
            editor = out / (name + '-editor.html')
            if old.get(name, {}).get('export_key') != record['export_key'] or not editor.exists():
                # Replace only this catalog's generated editor, never source room files.
                temporary = out / (name + '-editor.next.html')
                temporary.unlink(missing_ok=True)
                build_editor(batch / name, temporary)
                temporary.replace(editor)
            links.append(f'<a href="{editor.name}">Retocar</a>')
            notes.append(f"{record['cameras']} cámaras; {record['creative_details']} detalles creativos; "
                         f"{record['single_view_details']} aceptados con una sola vista; "
                         f"{record['rejected_details']} rechazados")
            notes.append('Revisión visual: ' + record['review_status'])
            notes.append('Visibilidad: ' + record['render_visibility']['status'])
            notes.append('Escala de profundidad: ' + record['depth_scale']['scale'])
            for kind, failure in sorted(record['detail_failures'].items()):
                reasons = [f'{gate}: {count}' for gate, count in sorted(failure['failed_gates'].items())]
                if failure['without_failed_gate']:
                    reasons.append(f"sin motivo de gate registrado: {failure['without_failed_gate']}")
                notes.append(f"{kind}: {failure['rejected']} rechazados ({'; '.join(reasons)})")
            for kind, scores in sorted(record['failure_model_confidence']['categories'].items()):
                notes.append(f"{kind}: rangos originales detector={scores['detector_score_range']}; "
                             f"SAM={scores['sam_predicted_iou_range']}; "
                             f"objetos sin puntuaciones completas={scores['objects_without_complete_scores']}")
        if record.get('error'):
            notes.append(record['error'])
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{labels[status]}</td>'
                    f'<td>{" · ".join(links)}</td><td>{html.escape(". ".join(notes))}</td></tr>')
        records.append(record)
    state = {'counts': counts, 'rooms': records, 'qualification': 'Export completion is not fidelity or playability'}
    for filename, contents in (
        ('catalog.json', json.dumps(state, indent=2) + '\n'),
        ('index.html', '<!doctype html><html lang="es"><meta charset="utf-8"><title>Habitaciones 3D</title>'
         '<style>body{background:#171b22;color:#eee;font:16px system-ui;margin:24px}a{color:#8cf}'
         'table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #455;padding:10px;text-align:left}</style>'
         '<h1>Habitaciones 3D</h1><p>Exportada significa que terminó la generación. No garantiza fidelidad ni navegación. '
         'Los detalles creativos y de una sola vista son señales para revisar, no errores confirmados.</p>'
         '<p>Las variantes de escenario aparecen por separado. Actualiza la página para ver nuevos resultados. '
         '«Sin finalizar» indica archivos parciales; no demuestra que el proceso siga activo.</p>'
         '<pre>' + html.escape(json.dumps(counts, indent=2)) + '</pre>'
         '<table><tr><th>Habitación</th><th>Estado</th><th>Acciones</th><th>Revisión</th></tr>'
         + ''.join(rows) + '</table></html>')):
        temporary = out / (filename + '.tmp')
        temporary.write_text(contents)
        temporary.replace(out / filename)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--reviews', type=Path, help='Optional output root of the deferred review runner')
    parser.add_argument('--watch-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.watch_seconds and args.watch_seconds < 10:
        parser.error('--watch-seconds must be at least 10, or 0 for one refresh')
    while True:
        state = refresh(args.assets, args.batch, args.out, args.reviews)
        print(json.dumps(state['counts']), flush=True)
        if not args.watch_seconds:
            return
        time.sleep(args.watch_seconds)


if __name__ == '__main__':
    main()
