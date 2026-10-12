#!/usr/bin/env python3
"""Export USA PC RDT cameras and collision bounds for room reconstruction.

Reads stored offsets directly, before the engine relocates pointers. Outputs
JSON measurements and an X/Z SVG survey, not a recovered environment mesh.
"""
import argparse
import hashlib
import html
import json
import struct
from pathlib import Path


def survey(path):
    data = path.read_bytes()

    def unpack(fmt, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(data):
            raise ValueError(f'{path.name}: truncated record at 0x{offset:X}')
        return struct.unpack_from(fmt, data, offset)

    if len(data) < 0x94:
        raise ValueError(f'{path.name}: truncated RDT header')
    camera_count = data[1]
    cameras = []
    for index in range(camera_count):
        values = unpack('<9i', 0x94 + index*44 + 8)
        cameras.append({'index': index, 'origin': list(values[:3]),
                        'target': list(values[3:6]), 'roll': values[6], 'fov': values[8]})
    offset, = unpack('<I', 0x4C)
    if offset < 0x94 + camera_count*44:
        raise ValueError(f'{path.name}: collision offset overlaps camera/header data')
    split_x, split_z, *counts = unpack('<hh5I', offset)
    # Room_SetupCollisionCallbacks makes group[4] the end pointer before adding
    # count[4]. Only the first four counts describe consumed boundary records.
    total = sum(counts[:4])
    if total > (len(data) - offset - 24)//12:
        raise ValueError(f'{path.name}: collision counts exceed the file')
    boundaries = []
    identities = {}
    index = 0
    for quadrant, count in enumerate(counts[:4]):
        for _ in range(count):
            values = unpack('<6H', offset + 24 + index*12)
            if values not in identities:
                identities[values] = len(boundaries)
                xmax, zmax, xmin, zmin, kind, flags = values
                boundaries.append({'index': len(boundaries), 'xmin': xmin, 'zmin': zmin,
                                   'xmax': xmax, 'zmax': zmax, 'type': kind, 'flags': flags,
                                   'quadrants': [], 'record_indices': []})
            entry = boundaries[identities[values]]
            if quadrant not in entry['quadrants']:
                entry['quadrants'].append(quadrant)
            entry['record_indices'].append(index)
            index += 1
    return {'source': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                'bytes': len(data), 'coordinate_system': 'X/Z room units; negative Y is up',
                'cameras': cameras, 'collision_offset': offset,
                'quadrant_split': [split_x,split_z], 'quadrant_counts': counts,
                'boundaries': boundaries,
                'limitations': ['Collision bounds are not visible wall surfaces.',
                             'No floor, stair or ceiling heights inferred.',
                             'Scripts, camera-switch zones and dynamic objects are not decoded.']}


def svg_plan(room):
    bounds = room['boundaries']
    coords = [(b[k],b[z]) for b in bounds for k,z in [('xmin','zmin'),('xmax','zmax')]]
    coords += [(p[0],p[2]) for c in room['cameras'] for p in [c['origin'],c['target']]]
    if not coords:
        raise ValueError('No coordinates to plot')
    xmin = min(p[0] for p in coords)-500; zmin = min(p[1] for p in coords)-500
    width = max(p[0] for p in coords)-xmin+500
    height = max(p[1] for p in coords)-zmin+500
    text_size = max(width,height)/65
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin} {zmin} {width} {height}" width="900" height="900">',
             '<title>'+html.escape(room['source'])+' collision and camera survey</title>',
             f'<rect x="{xmin}" y="{zmin}" width="{width}" height="{height}" fill="#111827"/>',
             '<g stroke-width="22" font-family="sans-serif" font-size="'+str(text_size)+'">']
    for b in bounds:
        x,z=b['xmin'],b['zmin']; w,h=b['xmax']-x,b['zmax']-z
        color = '#fbbf24' if b['type'] & 255 == 4 else '#34d399'
        if b['type'] & 255 == 3:
            r=w/2
            shape=f'<circle cx="{x+r}" cy="{z+r}" r="{r}"'
        else:
            shape=f'<rect x="{x}" y="{z}" width="{max(0,w)}" height="{max(0,h)}"'
        lines.append(shape+f' fill="{color}" fill-opacity="0.08" stroke="{color}"/>')
        lines.append(f'<text x="{x+50}" y="{z+text_size}" fill="{color}">B{b["index"]}</text>')
    for camera in room['cameras']:
        x,y,z=camera['origin']; tx,_ty,tz=camera['target']
        lines.append(f'<path d="M{x} {z} L{tx} {tz}" stroke="#f472b6" stroke-dasharray="80 60"/>')
        lines.append(f'<circle cx="{x}" cy="{z}" r="90" fill="#f472b6"/>')
        lines.append(f'<text x="{x+120}" y="{z}" fill="#f9a8d4">C{camera["index"]} Y={y}</text>')
    lines += ['</g></svg>']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rdt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    try:
        room=survey(args.rdt)
        plan=svg_plan(room)
    except (ValueError,OSError) as error:
        parser.exit(1,str(error)+'\n')
    args.out.mkdir(parents=True,exist_ok=True)
    stem=args.rdt.stem.lower()
    (args.out/(stem+'.json')).write_text(json.dumps(room,indent=2)+'\n')
    (args.out/(stem+'.svg')).write_text(plan)
    print(f'{stem}: {len(room["cameras"])} cameras, {len(room["boundaries"])} unique boundaries')


if __name__ == '__main__':
    main()
