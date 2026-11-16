"""Compatibility checks for the generated model against the original geometry.

    python3 tools/elira/check.py elira/source/elira.geo.json elira/elira.geo.json elira/elira.png [elira/elira_2048.png]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from geo import GeoModel

src, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
tex = Image.open(sys.argv[3]).convert('RGBA')
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name, info)


gs, go = src['minecraft:geometry'][0], out['minecraft:geometry'][0]
check('format version unchanged', src['format_version'] == out['format_version'], out['format_version'])
check('description unchanged (identifier, texture size, visible bounds)', gs['description'] == go['description'], json.dumps(go['description']))
strip = lambda b: {k: v for k, v in b.items() if k != 'cubes'}
check('same bones in the same order', [b['name'] for b in gs['bones']] == [b['name'] for b in go['bones']], f"{len(go['bones'])} bones")
diff = [a['name'] for a, b in zip(gs['bones'], go['bones']) if strip(a) != strip(b)]
check('every bone keeps its parent, pivot and rest rotation', not diff, f'changed: {diff}')
empty_then = {b['name'] for b in gs['bones'] if not b.get('cubes')}
empty_now = {b['name'] for b in go['bones'] if not b.get('cubes')}
check('every bone that carried cubes still does', empty_now <= empty_then, f'newly empty: {sorted(empty_now - empty_then)}')

W, H = go['description']['texture_width'], go['description']['texture_height']
check('texture keeps the declared aspect ratio', tex.size[0] * H == tex.size[1] * W, f'{tex.size} for UV space {W}x{H}')
A = np.asarray(tex)
kx, ky = tex.size[0] / W, tex.size[1] / H
cubes = [c for b in go['bones'] for c in b.get('cubes', [])]
bad = offgrid = empty = boxuv = 0
alphas = set(np.unique(A[..., 3]).tolist())
for c in cubes:
    if not isinstance(c['uv'], dict):
        boxuv += 1
        continue
    for f in c['uv'].values():
        u0, v0 = f['uv']
        u1, v1 = u0 + f['uv_size'][0], v0 + f['uv_size'][1]
        if min(u0, u1) < 0 or max(u0, u1) > W or min(v0, v1) < 0 or max(v0, v1) > H or u0 == u1 or v0 == v1:
            bad += 1
            continue
        px = [u0 * kx, u1 * kx, v0 * ky, v1 * ky]
        if any(abs(p - round(p)) > 1e-6 for p in px):
            offgrid += 1
        r = A[int(round(min(px[2:]))):int(round(max(px[2:]))), int(round(min(px[:2]))):int(round(max(px[:2]))), 3]
        if r.size == 0 or r.max() == 0:
            empty += 1
check('per-face UVs only (as in the original)', boxuv == 0)
check('all face UVs inside the texture', bad == 0, f'bad={bad}')
check('all face UVs land on whole texels', offgrid == 0, f'off grid={offgrid}')
check('no face maps to a fully transparent area', empty == 0, f'empty={empty}')
check('alpha is strictly on/off (cut-out, as in the original)', alphas <= {0, 255}, str(sorted(alphas)))
check('all numbers finite', all(np.isfinite(np.array(c['origin'] + c['size'] + c.get('pivot', []) + c.get('rotation', []), float)).all() for c in cubes))
check('no negative cube sizes', all(min(c['size']) >= 0 for c in cubes))
if len(sys.argv) > 4:
    alt = Image.open(sys.argv[4]).convert('RGBA')
    k = alt.size[0] / tex.size[0]
    check('second texture is the same atlas at another resolution', alt.size[1] == tex.size[1] * k and k in (0.5, 2.0), str(alt.size))
    check('second texture alpha is on/off', set(np.unique(np.asarray(alt)[..., 3]).tolist()) <= {0, 255})
so = Image.open(os.path.join(os.path.dirname(sys.argv[1]), 'elira.png'))
check('texture is the same size as the original', tex.size == so.size, f'{tex.size} vs {so.size}')

a, b = GeoModel(data=src), GeoModel(data=out)
(alo, ahi), (blo, bhi) = a.bounds(), b.bounds()
print('     original bounds', np.round(alo, 2), np.round(ahi, 2))
print('     new      bounds', np.round(blo, 2), np.round(bhi, 2))
check('stands on the same ground', abs(blo[1]) < 0.2, f'{blo[1]:.2f}')
check('height within one unit of the original', abs(ahi[1] - bhi[1]) < 1.0, f'{ahi[1]:.2f} -> {bhi[1]:.2f}')
vb = go['description']
check('fits the declared visible bounds', max(abs(blo[0]), bhi[0], abs(blo[2]), bhi[2]) <= vb['visible_bounds_width'] * 8 and bhi[1] <= (vb['visible_bounds_offset'][1] + vb['visible_bounds_height'] / 2) * 16)
quads = lambda g: sum(len(c['uv']) for bn in g['bones'] for c in bn.get('cubes', []))
print(f"     cubes {sum(len(bn.get('cubes', [])) for bn in gs['bones'])} -> {len(cubes)}, quads {quads(gs)} -> {quads(go)}")
print('RESULT', 'OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
