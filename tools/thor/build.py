"""Generate the redesigned Thor boss model.

    python3 tools/thor/build.py            # full quality -> thor/thor_boss.bbmodel + thor/thor_boss.png
    python3 tools/thor/build.py --spine-rig --name thor_boss_spine_rig
    python3 tools/thor/build.py --fast     # quick low-res pass for iterating on shapes
"""
import sys, os, io, json, time, base64, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core, rig, body
from materials import Swatches, Mats

SRC = os.path.join(ROOT, 'thor', 'source', 'tor_hair_reference_v3.bbmodel')
PARTS = ['body', 'head', 'hair', 'cloth']


def assemble(parts, spine=False):
    B = core.Builder()
    rig.build_rig(B)
    if 'body' in parts:
        body.neck(B)
        body.pelvis(B)
        for side in (1, -1):
            B.side = side
            body.torso(B)
            body.arm(B)
            body.hand(B)
            body.leg(B)
        B.side = 1
    if 'head' in parts:
        import head
        head.build(B)
    if 'hair' in parts:
        import hair
        hair.build(B)
    if 'cloth' in parts:
        import cloth
        cloth.build(B)
    if not spine:
        flatten(B)
    return B


def flatten(B):
    """Source-file hierarchy: head/body/cloth directly under root and both arms as top-level groups.
    The spine bones carry no rest rotation, so re-parenting leaves the rest pose untouched."""
    for name, b in B.bones.items():
        if b.parent in ('torso', 'chest') and name != 'chest':
            B.bones[b.parent].children.remove(name)
            b.parent = None if name in ('right arm', 'left_arm5') else 'root'
            if b.parent:
                B.bones['root'].children.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--spine-rig', action='store_true', help='parent head, arms and chest under torso > chest instead of the source hierarchy')
    ap.add_argument('--parts', default=','.join(PARTS))
    ap.add_argument('--out', default=os.path.join(ROOT, 'thor'))
    ap.add_argument('--name', default='thor_boss')
    ap.add_argument('--size', type=int, default=4096)
    ap.add_argument('--density', type=float, default=0)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *s: print(f'[{time.time() - t0:6.1f}s]', *s, flush=True)
    src = json.load(open(SRC))
    atlas_src = Image.open(io.BytesIO(base64.b64decode(src['textures'][0]['source'].split(',', 1)[1])))
    B = assemble(a.parts.split(','), a.spine_rig)
    log(f'bones {len(B.bones)} cubes {len(B.cubes)}')
    size = 2048 if a.fast else a.size
    dens = a.density or (9.0 if a.fast else 20.0)
    faces = core.build_faces(B, dens)
    core.compute_ao(faces, ndirs=24 if a.fast else 96, sdens=3.5 if a.fast else 6.0, res=3.0 if a.fast else 4.5, log=log)
    log(f'faces {len(faces)} hidden {sum(f.hidden for f in faces)}')
    scale = 1.0 if a.fast else 1.5
    while True:
        core.size_faces(faces, dens, scale)
        used = core.pack(faces, size)
        if used is not None:
            break
        scale *= 0.97
    log(f'texel density {dens * scale:.1f} px/unit, atlas rows used {used}/{size}')
    mats = Mats(Swatches(atlas_src)).table()
    atlas = core.bake(B, faces, size, mats, log=log)
    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, 'PNG', optimize=True)
    os.makedirs(a.out, exist_ok=True)
    d = core.export_bbmodel(B, src, buf.getvalue(), a.name + '.png', size, 1024)
    d['name'] = a.name
    with open(os.path.join(a.out, a.name + '.bbmodel'), 'w') as fh:
        json.dump(d, fh, separators=(',', ':'))
    with open(os.path.join(a.out, a.name + '.png'), 'wb') as fh:
        fh.write(buf.getvalue())
    log(f'wrote {a.name}.bbmodel / .png  ({len(d["elements"])} cubes, {len(d["groups"])} groups)')


if __name__ == '__main__':
    main()
