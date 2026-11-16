"""Generate Elira's redesigned model from the original geometry.

    python3 tools/elira/build.py            # elira/elira.geo.json + elira.png (1024x800) + elira_2048.png
    python3 tools/elira/build.py --fast --out /tmp/elira     # quick low-resolution pass for shape work
"""
import sys, os, json, time, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import kit, paint

SRC = os.path.join(ROOT, 'elira', 'source', 'elira.geo.json')
PARTS = ['body', 'head', 'hair', 'outfit']
MATS = {
    'skin': paint.skin, 'hair': paint.hair_strip, 'hair_cap': paint.hair_cap, 'dark': paint.dark, 'gold': paint.gold,
    'gem': paint.gem, 'nail': paint.nail, 'sole': paint.flat((36, 24, 46)), 'panel': paint.panel, 'belt': paint.belt,
    'choker': paint.choker,
}
DENS = {'skin': 30, 'hair': 24, 'hair_cap': 36, 'dark': 26, 'gold': 36, 'gem': 64, 'nail': 40, 'sole': 16, 'panel': 22,
        'belt': 34, 'choker': 40}


def assemble(parts=PARTS):
    geo = json.load(open(SRC))
    B = kit.Builder(kit.Skeleton(geo))
    for p in parts:
        __import__(p).build(B)
    return geo, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--parts', default=','.join(PARTS))
    ap.add_argument('--out', default=os.path.join(ROOT, 'elira'))
    ap.add_argument('--name', default='elira')
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *s: print(f'[{time.time() - t0:6.1f}s]', *s, flush=True)
    geo, B = assemble(a.parts.split(','))
    desc = geo['minecraft:geometry'][0]['description']
    W, H = (1024, 800) if a.fast else (2048, 1600)
    assert W * desc['texture_height'] == H * desc['texture_width'], 'atlas must keep the declared aspect ratio'
    faces = kit.build_faces(B)
    log(f'cubes {len(B.cubes)} faces {len(faces)} max planarity error {B.max_dev:.4f}')
    kit.compute_ao(faces, ndirs=32 if a.fast else 128, sdens=4.0 if a.fast else 7.0, res=4.0 if a.fast else 6.0, log=log)
    own = [f for f in faces if f.twin is None]
    scale = 0.5 if a.fast else 1.0
    while True:
        kit.size_faces(own, DENS, scale)
        used = kit.pack(own, W, H)
        if used is not None:
            break
        scale *= 0.97
    log(f'{len(own)} unique faces, {sum(f.hidden for f in own)} hidden; density x{scale:.2f}; atlas rows {used}/{H}')
    atlas = kit.bake(own, W, H, MATS, log=log)
    dropped = kit.prune(B, own, atlas)
    out = kit.export_geo(B, geo, desc['texture_width'] / W)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, a.name + '.geo.json'), 'w') as fh:
        fh.write(kit.dump_geo(out))
    if a.fast:
        Image.fromarray(atlas).save(os.path.join(a.out, a.name + '.png'), optimize=True)
    else:       # the original texture is 1024x800; the same UVs also address the 2048x1600 bake
        Image.fromarray(kit.halve(atlas)).save(os.path.join(a.out, a.name + '.png'), optimize=True)
        Image.fromarray(atlas).save(os.path.join(a.out, a.name + '_2048.png'), optimize=True)
    n = sum(len(b.get('cubes', [])) for b in out['minecraft:geometry'][0]['bones'])
    log(f'wrote {a.name}.geo.json ({n} cubes, {dropped} empty faces dropped) and textures from a {W}x{H} bake')


if __name__ == '__main__':
    main()
