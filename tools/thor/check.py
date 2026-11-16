"""Structural checks on a generated .bbmodel against the source project.

    python3 tools/thor/check.py thor/source/tor_hair_reference_v3.bbmodel thor/thor_boss.bbmodel
"""
import json, sys, base64, io
from PIL import Image
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Model
import numpy as np

src, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name, info)


check('top-level keys match', list(src.keys()) == list(out.keys()))
check('meta unchanged', src['meta'] == out['meta'])
G = {g['uuid']: g for g in out['groups']}
E = {e['uuid']: e for e in out['elements']}
seen_e, seen_g = [], []


def walk(n):
    if isinstance(n, str):
        seen_e.append(n)
        return
    seen_g.append(n['uuid'])
    for c in n['children']:
        walk(c)


for n in out['outliner']:
    walk(n)
check('every cube placed exactly once', sorted(seen_e) == sorted(E), f'{len(seen_e)} cubes')
check('every group placed exactly once', sorted(seen_g) == sorted(G), f'{len(seen_g)} groups')
check('unique uuids', len(set(seen_e) | set(seen_g)) == len(seen_e) + len(seen_g))

# legacy bones: every group name + uuid of the source survives
sg = {g['uuid']: g['name'] for g in src['groups']}
missing = [n for u, n in sg.items() if u not in G or G[u]['name'] != n]
check('all source bones kept (same name and uuid)', not missing, f'{len(sg)} source groups, missing {missing[:5]}')
new = sorted(g['name'] for g in out['groups'] if g['uuid'] not in sg)
print('     new bones:', len(new), new)

tex = out['textures'][0]
im = Image.open(io.BytesIO(base64.b64decode(tex['source'].split(',', 1)[1])))
check('texture embedded and square', im.size == (tex['width'], tex['height']), str(im.size))
check('uv space same as source', (tex['uv_width'], tex['uv_height']) == (src['textures'][0]['uv_width'], src['textures'][0]['uv_height']) and out['resolution'] == src['resolution'])
W = out['resolution']['width']
A = np.asarray(im)
bad = empty = 0
for e in out['elements']:
    for f in e['faces'].values():
        u1, v1, u2, v2 = f['uv']
        if min(u1, v1) < 0 or max(u2, v2) > W or u2 <= u1 or v2 <= v1:
            bad += 1
            continue
        k = im.size[0] / W
        reg = A[int(round(v1 * k)):int(round(v2 * k)), int(round(u1 * k)):int(round(u2 * k)), 3]
        if reg.size == 0 or reg.min() < 255:
            empty += 1
check('all face UVs inside the atlas', bad == 0, f'bad={bad}')
check('all face UVs land on opaque texels', empty == 0, f'transparent={empty}')
check('cube keys match source', set(src['elements'][0]) == set(out['elements'][0]))

a, b = Model(data=src), Model(data=out)
(alo, ahi), (blo, bhi) = a.bounds(), b.bounds()
print('     source bounds', np.round(alo, 2), np.round(ahi, 2))
print('     new    bounds', np.round(blo, 2), np.round(bhi, 2))
check('feet on the same ground plane', abs(alo[1] - blo[1]) < 0.01)
check('height within 0.5 unit of source', abs(ahi[1] - bhi[1]) < 0.5, f'{ahi[1]:.2f} -> {bhi[1]:.2f}')
check('footprint not larger than source', blo[0] >= alo[0] - 0.01 and bhi[0] <= ahi[0] + 0.01 and bhi[2] - blo[2] <= ahi[2] - alo[2] + 0.5)
print('     cubes', len(src['elements']), '->', len(out['elements']))
print('RESULT', 'OK' if ok else 'FAILED')
