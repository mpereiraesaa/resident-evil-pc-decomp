"""Summarize completed detail trials, including failures and source provenance.

Example: python tools/evaluate_room3d_details.py --run build/run-a --run build/run-b
--out build/detail-evaluation. Acceptance and mask IoU are not fidelity scores.
"""

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from room3d_confidence import failure_confidence, observation_confidence


def selected_trial(report):
    """Read an explicitly accepted selection; missing metadata proves nothing."""
    if not isinstance(report, dict) or report.get("accepted") is not True:
        return None
    index, trials = report.get("selected"), report.get("trials", [])
    if type(index) is not int or not isinstance(trials, list) or not 0 <= index < len(trials):
        return None
    trial = trials[index]
    if not isinstance(trial, dict):
        return None
    gates = trial.get("gates")
    if not isinstance(gates, dict) or not gates or not all(passed is True for passed in gates.values()):
        return None
    return trial


def creative_evidence(obj):
    """Report the selected creative geometry separately from its rejected fit."""
    if obj.get("accepted") is not True:
        return None
    refined = obj.get('creative_part_refinement', {})
    if refined.get('accepted') is True:
        gates = refined.get('gates', {})
        if not gates or not all(passed is True for passed in gates.values()):
            return None
        return {'shape': refined['selected_shape'], 'placement': refined['placement'],
                'gates': gates, 'connections': None, 'headroom': None, 'landing': None,
                'depth_before': [c['before'] for c in refined['raw_depth']],
                'depth_after': [c['after'] for c in refined['raw_depth']],
                'qualification': 'Selected post-acceptance creative fixture parts; earlier interpretation trials describe previous geometry. Not fidelity or navigation proof.'}
    interpretation = obj.get("interpretation", {})
    trial = selected_trial(interpretation)
    if trial is None:
        return None
    landing = trial.get("landing_assembly")
    landing_trial = selected_trial(landing) if landing is not None else None
    if landing is not None and landing_trial is None:
        return None
    return {
        "shape": trial.get("shape"),
        "placement": trial.get("placement", []),
        "gates": trial["gates"],
        "connections": trial.get("connections"),
        "headroom": (landing_trial if landing_trial is not None else trial).get("headroom"),
        "depth_before": trial.get("depth_before", []),
        "depth_after": trial.get("depth_after", []),
        "landing": None if landing_trial is None else {
            "depth": landing_trial.get("depth"),
            "gates": landing_trial["gates"],
            "contacts": landing_trial.get("contacts"),
            "training_cameras": landing.get("training_cameras", []),
            "heldout_cameras": landing.get("heldout_cameras", []),
            "heldout_checks": landing.get("heldout", []),
        },
        "qualification": "Selected estimated geometry checks; not original fidelity or gameplay traversal proof",
    }


def object_evidence(obj):
    """Keep acceptance distinct from multiview, physical and visual evidence."""
    cameras = sorted({o["camera"] for o in obj.get("observations", [])})
    scores = obj.get("candidate", obj.get("after", []))
    headroom = obj.get("headroom")
    replaced_by_interpretation = (obj.get("interpretation", {}).get("accepted", False)
                                 or obj.get('creative_part_refinement', {}).get('accepted', False))
    if replaced_by_interpretation:
        # These diagnostics describe the rejected fit, not its creative replacement.
        scores, headroom = [], None
    creative = creative_evidence(obj)
    if replaced_by_interpretation and creative is not None:
        headroom = creative["headroom"]
    visibility_scope = obj.get("visibility_evidence_scope", "fitted_scene")
    visibility_changed = visibility_scope == "before_visual_proxy_cleanup"
    if visibility_changed:
        scores = []
        if creative is not None:
            creative["placement"] = []
            creative["depth_before"] = []
            creative["depth_after"] = []
            creative["visibility_evidence_scope"] = visibility_scope
    guard = obj.get("floor_depth_guard")
    guard_evidence = None
    if guard is not None:
        guard_evidence = {
            "accepted": guard["accepted"], "reason": guard["reason"],
            "visibility_evidence_scope": visibility_scope,
            "depth_reference": obj.get("depth_reference", {"kind": "previous_scene"}),
            "training": [] if visibility_changed else guard.get("training", []),
            "heldout": [] if visibility_changed else guard.get("heldout", []),
            "qualification": "Selected-assembly depth veto, not fidelity. Checks precede any later visual proxy cleanup.",
        }
    return {
        "kind": obj["kind"], "accepted": obj["accepted"],
        'model_confidence': observation_confidence(obj),
        "path": obj.get("path"), "observing_cameras": cameras,
        "shared_between_cameras": len(cameras) >= 2,
        "heldout_cameras": obj.get("heldout_cameras", []),
        "minimum_mask_iou": min((s["silhouette_iou"] for s in scores
                                  if "silhouette_iou" in s), default=None),
        "heldout_scores": [] if replaced_by_interpretation or visibility_changed else obj.get("heldout", []),
        "visibility_evidence_scope": visibility_scope,
        "proportions": None if replaced_by_interpretation else obj.get("proportions"),
        "headroom_clear": headroom.get("clear") if headroom is not None else None,
        "obstructed_samples": headroom.get("obstructed_samples") if headroom else None,
        "creative_geometry": creative,
        "floor_depth_guard": guard_evidence,
        "depth_reference": obj.get("depth_reference"),
        "visual_quality": "Requires image inspection; numerical acceptance is insufficient",
    }


def completed_entries(root):
    """Inspect completed evaluation inputs without trusting a live manifest."""
    entries, excluded = [], []
    for path in sorted((root / "checkpoints").glob("*.json")):
        try:
            record = json.loads(path.read_text())
            result = record["result"]
            if record.get("schema") != 1 or result.get("status") != "image_assisted":
                excluded.append({"room": path.stem, "reason": "not_image_assisted"})
                continue
            if any("error" in c for c in result.get("cameras", [])):
                raise ValueError("camera_error")
            folder = root / path.stem
            # These are the inputs consumed below, not proof that the entire
            # export or its execution-code snapshot has been verified.
            for name in ("details/results.json", "assisted.json"):
                source = folder / name
                if name == "assisted.json" and not source.exists() and name not in record["artifacts"]:
                    continue
                if hashlib.sha256(source.read_bytes()).hexdigest() != record["artifacts"].get(name):
                    raise ValueError("changed_evaluation_input: " + name)
            entries.append({"room": path.stem, "status": "image_assisted",
                            "checkpoint_key": record.get("key")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append({"room": path.stem, "reason": str(error)})
    return entries, excluded


def final_geometry_metrics(folder):
    path = folder/'final-details.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    mesh = folder/'fitted.obj'
    visibility = mesh.with_suffix('.cameras')
    actual_visibility = hashlib.sha256(visibility.read_bytes()).hexdigest() if visibility.exists() else None
    if (record.get('schema') != 1
            or record.get('mesh_sha256') != hashlib.sha256(mesh.read_bytes()).hexdigest()
            or record.get('visibility_sha256') != actual_visibility):
        raise ValueError(f'Final detail metrics differ from current mesh/visibility: {folder}')
    return record


def summarize(root, completed_only=False, audits=None):
    if completed_only:
        entries, excluded = completed_entries(root)
    else:
        entries = json.loads((root / "manifest.json").read_text())["results"]
        excluded = []
    rooms = []
    totals = {}
    for item in entries:
        folder = root / item["room"].lower()
        if item["status"] != "image_assisted":
            raise ValueError(f"Incomplete reconstruction: {folder}")
        report = json.loads((folder / "details/results.json").read_text())
        appearance_path = folder / "assisted.json"
        bake = (json.loads(appearance_path.read_text()).get("bake", {})
                if appearance_path.exists() else {})
        total_vertices = bake.get("vertices", 0)
        appearance = {
            "sampled_vertex_fraction": (bake.get("sampled_vertices", 0) / total_vertices
                                        if total_vertices else None),
            "inferred_vertices": bake.get("inferred_vertices"),
            "grazing_rejections": bake.get("grazing_rejections"),
            "qualification": "Vertex sampling coverage, not textured pixel area or visual fidelity",
        }
        audit_path = folder / "visual-audit/findings.json"
        visual_audit = (json.loads(audit_path.read_text()) if audit_path.exists()
                        else {"status": "not_inspected", "findings": []})
        if audits is not None:
            from room3d_visual_audit import load_audit

            external = load_audit(audits, folder)
            if external is not None:
                visual_audit = external
        counts = Counter()
        rejected = Counter()
        evidence = [object_evidence(obj) for obj in report["objects"]]
        for obj in report["objects"]:
            counts[(obj["kind"], "accepted" if obj["accepted"] else "rejected")] += 1
            category = totals.setdefault(obj["kind"], {"accepted": 0, "rejected": 0,
                                                        "failed_gates": {}})
            category["accepted" if obj["accepted"] else "rejected"] += 1
            if not obj["accepted"]:
                for gate, passed in obj.get("gates", {}).items():
                    if passed is False:
                        rejected[gate] += 1
                        category["failed_gates"][gate] = category["failed_gates"].get(gate, 0) + 1
        rooms.append(
            {
                "room": item["room"],
                "checkpoint_key": item.get("checkpoint_key"),
                "final_geometry_metrics": final_geometry_metrics(folder),
                "appearance": appearance,
                "visual_audit": visual_audit,
                "accepted": report["accepted"],
                "objects": len(report["objects"]),
                "categories": [
                    {"kind": k, "status": s, "count": n}
                    for (k, s), n in sorted(counts.items())
                ],
                "failed_gates": dict(rejected),
                'failure_model_confidence': failure_confidence(report['objects']),
                "categories_without_fit_observations": sorted(
                    {"staircase", "railing", "lamp"} - {o["kind"] for o in evidence}
                ),
                "object_evidence": evidence,
                "accepted_shared_objects": sum(
                    o["accepted"] and o["shared_between_cameras"] for o in evidence
                ),
                "accepted_with_unknown_or_failed_headroom": sum(
                    o["accepted"] and o["kind"] == "staircase"
                    and o["headroom_clear"] is not True for o in evidence
                ),
                "heldout_objects": sum(
                    bool(o.get("heldout_cameras")) for o in report["objects"]
                ),
            }
        )
    provenance = root / "frozen-config.json"
    frozen = json.loads(provenance.read_text()) if provenance.exists() else None
    if frozen:
        for name, expected in frozen["source_sha256"].items():
            snapshot = root / "sources" / name
            if Path(name).name == name and not snapshot.exists():
                # Some frozen manifests use module filenames rather than tools/ paths.
                snapshot = root / 'sources' / 'tools' / name
            path = snapshot if snapshot.exists() else Path(name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source differs: {path}")
    variant_groups = {}
    for room in rooms:
        name = room["room"].lower()
        group = name[:-1] if re.fullmatch(r"room[0-9a-f]{4}", name) else name
        variant_groups.setdefault(group, []).append(name)
    return {
        "root": str(root.resolve()),
        "rooms": rooms,
        "frozen_configuration": frozen,
        "source_execution_snapshot": "frozen_sources_verified" if frozen else "not_recorded",
        "completed_only": completed_only,
        "excluded_checkpoints": excluded,
        "category_totals": totals,
        "variant_groups": variant_groups,
        "variant_qualification": "Scenario variants are grouped by room stem, not independent unseen-room tests or guaranteed identical layouts",
        "qualification": "Acceptance measures conservative mask/depth agreement, not 3D fidelity. Repeated evaluations are not fresh unseen-room tests.",
    }


def write_report(roots, output, completed_only=False, audits=None):
    if any(output.resolve().is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Keep evaluation reports outside input runs")
    output.mkdir(parents=True, exist_ok=True)
    runs = [summarize(root, completed_only=completed_only, audits=audits) for root in roots]
    rows = [
        '<!doctype html><meta charset="utf-8"><title>Evaluación de detalles</title>',
        "<style>body{background:#171a20;color:#eee;font:16px sans-serif;margin:24px}a{color:#8bd}td,th{padding:8px;text-align:left;border-bottom:1px solid #555}img{max-width:100%}</style>",
        "<h1>Detalles: evaluación automática</h1><p>Se incluyen candidatos rechazados y fallos por categoría. Aceptación y coincidencia con máscaras no equivalen a fidelidad 3D. Las partes ocultas siguen siendo estimadas.</p>",
    ]
    for root, run in zip(roots, runs, strict=True):

        def link(path):
            return quote(os.path.relpath(path.resolve(), output.resolve()), safe="/.")

        rows.append(f'<h2>{html.escape(root.name)}</h2>')
        if (root / "index.html").is_file():
            rows.append(f'<p><a href="{link(root / "index.html")}">Comparación del batcher</a></p>')
        rows.append(f'<p>{len(run["rooms"])} entradas; {len(run["variant_groups"])} grupos de habitación. '
                    f'{len(run["excluded_checkpoints"])} checkpoints excluidos. '
                    'Las variantes no cuentan como pruebas independientes. Código ejecutado: '
                    + html.escape(run["source_execution_snapshot"]) + '.</p>')
        rows.append('<h3>Rechazos por categoría</h3><pre>'
                    + html.escape(json.dumps(run["category_totals"], indent=2)) + '</pre>')
        rows.append(
            "<table><tr><th>Habitación</th><th>Aceptados / propuestas</th><th>Objetos con cámara reservada</th><th>Criterios fallidos</th><th>Categorías sin observaciones ajustables</th><th>Aceptados compartidos entre cámaras</th><th>Escaleras aceptadas sin espacio superior confirmado</th><th>Fracción de vértices muestreados (no área visible)</th></tr>"
        )
        for room in run["rooms"]:
            folder = root / room["room"].lower()
            rows.append(
                f'<tr><td><a href="{link(folder / "details/index.html")}">{html.escape(room["room"])}</a></td><td>{room["accepted"]} / {room["objects"]}</td><td>{room["heldout_objects"]}</td><td>{html.escape(str(room["failed_gates"]))}</td><td>{html.escape(", ".join(room["categories_without_fit_observations"]))}</td><td>{room["accepted_shared_objects"]}</td><td>{room["accepted_with_unknown_or_failed_headroom"]}</td><td>{room["appearance"]["sampled_vertex_fraction"]}</td></tr>'
            )
        rows.append("</table>")
        rows.append('<h3>Puntuaciones originales de los modelos</h3><p>'
                    'Detector y SAM por máscara; no son probabilidades calibradas de identidad '
                    'o fidelidad. Los informes antiguos pueden carecer de estos datos.</p>')
        for room in run['rooms']:
            rows.append('<details><summary>' + html.escape(room['room'] + ' — fallos por categoría')
                        + '</summary><pre>' + html.escape(json.dumps(room['failure_model_confidence'], indent=2))
                        + '</pre></details>')
            for obj in room['object_evidence']:
                rows.append('<details><summary>' + html.escape(room['room'] + ' — ' + obj['kind'])
                            + '</summary><pre>' + html.escape(json.dumps(obj['model_confidence'], indent=2))
                            + '</pre></details>')
        rows.append('<h3>Geometría creativa seleccionada</h3><p>Forma, colocación y conexiones '
                    'del candidato final; los datos del ajuste rechazado se omiten. '
                    'Los controles geométricos no son una aprobación visual.</p>')
        for room in run["rooms"]:
            for obj in room["object_evidence"]:
                if obj["creative_geometry"] is not None:
                    rows.append('<details><summary>' + html.escape(room["room"] + ' — ' + obj["kind"])
                                + '</summary><pre>' + html.escape(json.dumps(
                                    obj["creative_geometry"], indent=2)) + '</pre></details>')
        rows.append('<h3>Control de profundidad de la geometría seleccionada</h3><p>'
                    'La referencia del suelo detecta sesgos de escala del modelo. '
                    'Una decisión anterior a la limpieza visual no mide la escena posterior.</p>')
        for room in run["rooms"]:
            for obj in room["object_evidence"]:
                if obj["floor_depth_guard"] is not None:
                    rows.append('<details><summary>' + html.escape(room["room"] + ' — ' + obj["kind"])
                                + '</summary><pre>' + html.escape(json.dumps(
                                    obj["floor_depth_guard"], indent=2)) + '</pre></details>')
        rows.append('<h3>Medición de la malla final</h3><p>Geometría después de la limpieza, '
                    'vinculada al OBJ y sus reglas de visibilidad. No sustituye la revisión visual.</p>')
        for room in run['rooms']:
            if room['final_geometry_metrics'] is not None:
                rows.append('<details><summary>' + html.escape(room['room']) + '</summary><pre>'
                            + html.escape(json.dumps(room['final_geometry_metrics'], indent=2))
                            + '</pre></details>')
        rows.append("<h3>Inspección visual registrada</h3><p>Los resultados numéricos no sustituyen esta revisión. Ausencia de inspección no significa aprobación.</p>")
        for room in run["rooms"]:
            folder = root / room["room"].lower()
            audit = room["visual_audit"]
            rows.append(f'<h4>{html.escape(room["room"])}: {html.escape(audit["status"])}</h4>')
            if 'binding' in audit:
                rows.append('<p>Vinculación: ' + html.escape(audit['binding']) +
                            '. Cámaras revisadas: ' + html.escape(str(audit.get('viewed_cameras', []))) + '</p>')
            comparison = folder / "visual-audit/index.html"
            if comparison.exists():
                rows.append(f'<p><a href="{link(comparison)}">Original y reconstrucción renderizada</a></p>')
            rows.append("<ul>" + "".join(f"<li>{html.escape(finding)}</li>"
                                         for finding in audit.get("findings", [])) + "</ul>")
    (output / "summary.json").write_text(json.dumps(runs, indent=2) + "\n")
    (output / "index.html").write_text("\n".join(rows))
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true",
                        help="Inspect hash-checked completed checkpoints while the batch is running")
    parser.add_argument('--audits', type=Path, help='External checkpoint-bound visual findings')
    args = parser.parse_args()
    write_report(args.run, args.out, completed_only=args.completed_only, audits=args.audits)
    print(args.out / "index.html")
