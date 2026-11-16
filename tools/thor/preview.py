"""Render preview sheets of a .bbmodel with the bundled software renderer.

    python3 tools/thor/preview.py thor/thor_boss.bbmodel thor/preview            # turnaround + close-ups
    python3 tools/thor/preview.py thor/thor_boss.bbmodel thor/preview --compare thor/source/tor_hair_reference_v3.bbmodel
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from render import Model, render, sheet


def stack(ims):
    out = Image.new('RGB', (max(i.width for i in ims), sum(i.height for i in ims)), (58, 62, 74))
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    m = Model(src)
    q, tex = m.quads(), m.texture()
    views = [(0, 0), (35, 10), (90, 0), (145, 10), (180, 0), (325, 10)]
    sheet(q, tex, views, size=(330, 700)).save(os.path.join(out, 'turnaround.png'))
    ims = [render(q, tex, yaw=y, pitch=p, size=(560, 600), center=c, scale=s)
           for y, p, c, s in ((0, 0, (0, 45.6, 0), 80), (25, 8, (0, 38.5, 0), 36), (200, 10, (0, 30, 0), 20))]
    row = Image.new('RGB', (560 * 3, 600))
    for i, im in enumerate(ims):
        row.paste(im, (i * 560, 0))
    row.save(os.path.join(out, 'closeups.png'))
    if '--compare' in sys.argv:
        o = Model(sys.argv[sys.argv.index('--compare') + 1])
        v = [(0, 0), (35, 10), (180, 0)]
        kw = dict(size=(330, 700), center=(0, 26.2, 0.5), scale=14.2)
        stack([sheet(o.quads(), o.texture(), v, **kw), sheet(q, tex, v, **kw)]).rotate(0).save(os.path.join(out, 'before_after.png'))


if __name__ == '__main__':
    main()
