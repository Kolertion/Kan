"""Render preview sheets with the bundled software renderer.

    python3 tools/elira/preview.py elira/elira.geo.json elira/elira.png elira/preview --compare elira/source
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from geo import GeoModel
from render import render, sheet

FRAME = dict(center=(0, 19.3, 0.6), scale=17.4)


def row(ims):
    out = Image.new('RGB', (sum(i.width for i in ims), max(i.height for i in ims)), (58, 62, 74))
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width
    return out


def stack(ims):
    out = Image.new('RGB', (max(i.width for i in ims), sum(i.height for i in ims)), (58, 62, 74))
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    return out


def main():
    geo, tex, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    q, T = GeoModel(geo).quads(), Image.open(tex).convert('RGBA')
    sheet(q, T, [(0, 0), (35, 10), (90, 0), (145, 10), (180, 0), (325, 10)], size=(330, 700), **FRAME).save(os.path.join(out, 'turnaround.png'))
    shots = [dict(yaw=0, pitch=0, center=(0, 35.55, 0), scale=104), dict(yaw=38, pitch=4, center=(0, 35.55, 0), scale=104),
             dict(yaw=0, pitch=0, center=(0, 29.6, 0), scale=52), dict(yaw=22, pitch=6, center=(0, 21.6, 0), scale=52)]
    row([render(q, T, size=(520, 560), **s) for s in shots]).save(os.path.join(out, 'closeups.png'))
    row([render(q, T, size=(520, 560), **s) for s in (dict(yaw=180, pitch=6, center=(0, 22, 1), scale=22),
                                                       dict(yaw=20, pitch=8, center=(0, 5.2, 0), scale=40),
                                                       dict(yaw=205, pitch=10, center=(0, 34.5, 0), scale=60))]).save(os.path.join(out, 'details.png'))
    if '--compare' in sys.argv:
        d = sys.argv[sys.argv.index('--compare') + 1]
        name = os.path.basename(geo).split('.')[0]
        qo, To = GeoModel(os.path.join(d, name + '.geo.json')).quads(), Image.open(os.path.join(d, name + '.png')).convert('RGBA')
        v = [(0, 0), (35, 10), (180, 0)]
        kw = dict(size=(380, 700), center=(0, 19.3, 0.6), scale=17.0)
        stack([sheet(qo, To, v, **kw), sheet(q, T, v, **kw)]).save(os.path.join(out, 'before_after.png'))


if __name__ == '__main__':
    main()
