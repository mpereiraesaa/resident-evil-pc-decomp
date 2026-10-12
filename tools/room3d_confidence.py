"""Report raw model scores by source mask, without treating them as probabilities."""

import math

QUALIFICATION = ('Raw detector and SAM scores, not calibrated object identity, '
                 'geometry fidelity or acceptance probabilities. Missing legacy scores are not inferred.')


def observation_confidence(obj):
    records = []
    for obs in obj.get('observations', []):
        evidence = obs.get('confidence_evidence')
        if evidence is None:
            evidence = [{'mask': obs.get('mask'), 'detector_score': obs.get('detector_score'),
                         'predicted_iou': obs.get('predicted_iou')}]
        for source in evidence:
            record = {'camera': obs['camera'], 'mask': source.get('mask')}
            for field in ('mask_pixels_sha256', 'sam_prediction_source'):
                if field in source:
                    record[field] = source[field]
            for field in ('detector_score', 'predicted_iou'):
                value = source.get(field)
                record[field] = (float(value) if isinstance(value, (int, float))
                                 and not isinstance(value, bool) and math.isfinite(value)
                                 and 0 <= value <= 1 else None)
            record['status'] = ('recorded' if all(record[k] is not None for k in
                                ('detector_score', 'predicted_iou')) else 'missing_or_invalid')
            original = source.get('original_predicted_iou')
            if (isinstance(original, (int, float)) and not isinstance(original, bool)
                    and math.isfinite(original) and 0 <= original <= 1):
                record['original_predicted_iou'] = float(original)
            records.append(record)
    complete = sum(r['status'] == 'recorded' for r in records)
    any_score = any(r[k] is not None for r in records for k in ('detector_score', 'predicted_iou'))
    return {'status': 'recorded' if records and complete == len(records)
            else 'partial' if any_score else 'not_recorded',
            'records': records, 'qualification': QUALIFICATION}


def failure_confidence(objects):
    """Category summaries retain missing counts instead of silently dropping them."""
    categories = {}
    for obj in objects:
        if obj.get('accepted'):
            continue
        bucket = categories.setdefault(obj.get('kind', 'unknown'),
            {'rejected_objects': 0, 'objects_without_complete_scores': 0,
             'recorded_masks': 0, 'missing_or_invalid_masks': 0,
             'detector_score_range': None, 'sam_predicted_iou_range': None})
        scores = observation_confidence(obj)
        bucket['rejected_objects'] += 1
        bucket['objects_without_complete_scores'] += scores['status'] != 'recorded'
        for record in scores['records']:
            bucket['recorded_masks'] += record['status'] == 'recorded'
            bucket['missing_or_invalid_masks'] += record['status'] != 'recorded'
            for field, name in (('detector_score', 'detector_score_range'),
                                ('predicted_iou', 'sam_predicted_iou_range')):
                value = record[field]
                if value is not None:
                    old = bucket[name]
                    bucket[name] = [value, value] if old is None else [min(old[0], value), max(old[1], value)]
    return {'categories': categories, 'qualification': QUALIFICATION}
