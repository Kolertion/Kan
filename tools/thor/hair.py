"""Long red hair: scalp cap + bangs ride on the head, the knee-length mass hangs from the chest bone
in six three-joint columns so it can sway without swinging through the body when the head turns."""
import numpy as np
from core import catmull, norm, smoothstep
from rig import TOGA_SIDE

HD = 16.0           # hair texel density (relative to the default 20)
HZ = 0.25           # pushes the back mass clear of the robe strap
HEAD_C = np.array([0, 46.2, 0.55])

# column -> (x at the shoulders, [upper, middle, lower] bones)
COLUMNS = [
    (6.3, ['hair19', 'hair21', 'hair23']), (3.8, ['hair18', 'hair22', 'hair29']), (1.3, ['hair20', 'hair27', 'hair28']),
    (-1.3, ['hair14', 'hair24', 'hair26']), (-3.8, ['hair13', 'hair16', 'hair25']), (-6.3, ['hair15', 'hair12', 'hair17']),
]
SPLIT = (33.5, 23.5)


def taper(w0, root=0.72, tip=0.42, t0=0.70):
    return lambda t: w0 * (root + (1 - root) * smoothstep(0.0, 0.14, t)) * (1 - (1 - tip) * smoothstep(t0, 1.0, t))


def col_path(xc, dx, zo, yend, rs):
    j = lambda a: rs.uniform(-a, a)
    x = lambda f: xc * f + dx
    return catmull([
        (xc * 0.42 + dx * 0.5, 44.7, 2.75 + HZ * 0.4 + zo * 0.3), (x(0.80), 41.9, 3.75 + HZ + zo),
        (x(1.0) + j(0.15), 37.5, 4.3 + HZ + zo), (x(1.04) + j(0.2), 31.5, 4.25 + HZ + zo),
        (x(1.08) + j(0.25), 25.0, 4.7 + HZ + zo), (x(1.11) + j(0.3), 18.8, 5.3 + HZ + zo),
        (x(1.13) + j(0.35), yend, 5.9 + HZ + zo + (18.8 - yend) * 0.06)], 6)


def build(B):
    B.side = 1
    rs = np.random.RandomState(11)
    b = B.bone
    H = dict(mat='hair', dens=HD)
    # ---- bones
    b('hair', 'head', (0, 47.0, 1.0))
    b('hair5', 'hair', (2.6, 46.6, 0.5))
    b('hair2', 'hair', (-2.6, 46.6, 0.5))
    b('hair_long', 'chest', (0, 43.4, 3.0))            # NEW: root of the hanging mass
    b('hair6', 'hair_long', (0, 43.4, 3.0))
    b('back_hair7', 'hair_long', (0, 43.4, 3.2))
    for xc, (top, mid, low) in COLUMNS:
        par = 'hair6' if xc > 0 else 'hair_long'
        b(top, par, (xc * 0.6, 43.6, 3.2 + HZ))
        b(mid, top, (xc * 1.03, SPLIT[0], 4.3 + HZ))
        b(low, mid, (xc * 1.09, SPLIT[1], 4.9 + HZ))

    # ---- scalp underlay
    B.rbox('hair', (0, 46.55, 0.98), (5.1, 4.4, 5.7), k=0.78, k2=0.78, tone=0.62, **H)
    B.box('hair', (0, 47.95, -1.72), (4.5, 1.1, 1.7), tone=0.66, **H)
    B.box('hair', (0, 44.2, 2.9), (4.3, 2.4, 1.6), tone=0.6, **H)                       # nape

    def radial(t, tan, mid):
        d = mid - HEAD_C
        return (d[0], max(d[1], 0.0) * 1.1 + 0.02, d[2] * 0.35)

    # ---- cap: strands combed from the parting down both sides
    seed = 0
    for s in (1, -1):
        bone = 'hair5' if s > 0 else 'hair2'
        for k, z0 in enumerate((-1.55, -0.55, 0.5, 1.55, 2.5)):
            seed += 1
            yend = (41.4, 41.9, 42.4, 42.2, 42.6)[k] + rs.uniform(-0.3, 0.3)
            zc = z0 * (0.72 if z0 < 0 else 0.86)       # strands converge a little towards the crown
            pts = catmull([(0.2 + 0.06 * s, 49.02, zc), (1.3 * s, 48.84, zc * 0.3 + z0 * 0.7), (2.18 * s, 48.2, z0 + 0.1), (2.72 * s, 47.1, z0 + 0.25),
                           (2.94 * s, 45.4, z0 + 0.45), (3.0 * s, 43.5, z0 + 0.7), (3.18 * s, yend, z0 + 0.95)], 5)
            B.ribbon(bone, pts, taper(1.32, 0.9, 0.5), 0.3, radial, seg=1.5, tip=True, seed=seed,
                     shine=0.26, shinew=0.10, shinek=0.55, **H)
    for k, x0 in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
        seed += 1
        pts = catmull([(x0 * 0.2 + 0.2, 48.98, 1.5), (x0 * 0.6, 48.6, 2.75), (x0 * 0.95, 47.6, 3.65), (x0 * 1.15, 46.0, 4.05),
                       (x0 * 1.25, 44.4, 4.0), (x0 * 1.3, 42.7 + rs.uniform(-0.3, 0.3), 3.9 + HZ)], 5)
        B.ribbon('hair', pts, taper(1.42, 0.8, 0.6), 0.3, lambda t, tan, mid, x0=x0: (x0 * 0.3, max(mid[1] - 46.2, 0) * 1.0, 1.0),
                 seg=1.5, tip=True, seed=seed, shine=0.30, shinew=0.10, shinek=0.5, **H)

    # ---- crown: strands laid along the parting so no scalp underlay shows from above
    for k, (xo, w) in enumerate(((-0.62, 1.3), (0.72, 1.3), (0.05, 0.8))):
        seed += 1
        pts = catmull([(xo, 48.86 + (k == 2) * 0.1, -2.0), (xo, 49.08 + (k == 2) * 0.1, -0.6), (xo, 49.08 + (k == 2) * 0.1, 1.0), (xo * 1.1, 48.8, 2.4), (xo * 1.2, 48.1, 3.3)], 4)
        B.ribbon('hair', pts, w, 0.26, lambda t, tan, mid, xo=xo: (xo * 0.5, 1.0, (mid[2] - 0.5) * 0.25), seg=1.6, seed=seed,
                 shine=0.4, shinew=0.25, shinek=0.4, **H)

    # ---- short root strands fanning from the parting to the hair line
    for k, (xe, ye) in enumerate(((-2.35, 47.2), (-1.5, 47.55), (-0.65, 47.75), (0.35, 47.8), (1.25, 47.6), (2.1, 47.3), (2.55, 46.9), (-2.6, 46.8))):
        seed += 1
        pts = catmull([(0.2 + xe * 0.08, 49.0, -0.9), (xe * 0.55, 48.82, -1.85), (xe * 0.9, 48.3, -2.5), (xe, ye, -2.68)], 4)
        B.ribbon('hair5' if xe > 0 else 'hair2', pts, taper(1.05, 0.9, 0.45, 0.55), 0.24,
                 lambda t, tan, mid: (mid[0] * 0.3, 0.9 if mid[1] > 48.4 else 0.25, -1.0), seg=1.2, tip=False, seed=seed,
                 shine=0.45, shinew=0.2, shinek=0.4, **H)

    # ---- bangs (legacy bones hair7..hair11 hang from hair2)
    front = lambda t, tan, mid: (mid[0] * 0.22, max(mid[1] - 47.6, 0) * 1.6, -1.0)
    bangs = {
        'hair7': [([(0.3, 48.8, -2.2), (0.25, 47.6, -2.9), (0.05, 46.3, -3.12), (-0.2, 45.2, -3.3), (-0.42, 44.5, -3.32), (-0.55, 43.6, -3.15)], 0.32),
                  ([(0.6, 48.82, -2.15), (0.62, 47.5, -2.84), (0.44, 46.5, -3.02), (0.3, 45.75, -3.12)], 0.27)],
        'hair8': [([(-0.2, 48.82, -2.15), (-1.1, 47.75, -2.84), (-1.92, 46.5, -2.9), (-2.32, 45.0, -2.75), (-2.47, 43.4, -2.55), (-2.52, 42.1, -2.35)], 0.78),
                  ([(-0.6, 48.86, -1.9), (-1.6, 48.0, -2.62), (-2.38, 46.9, -2.62), (-2.72, 45.4, -2.4), (-2.82, 43.8, -2.15)], 0.72)],
        'hair10': [([(0.72, 48.82, -2.1), (1.5, 47.8, -2.8), (2.12, 46.6, -2.86), (2.42, 45.1, -2.7), (2.52, 43.6, -2.5), (2.58, 42.5, -2.32)], 0.76),
                   ([(1.0, 48.86, -1.85), (1.9, 48.0, -2.55), (2.5, 46.9, -2.55), (2.76, 45.5, -2.34), (2.86, 44.1, -2.12)], 0.7)],
        'hair9': [([(-1.3, 48.84, -1.3), (-2.5, 47.9, -1.95), (-2.98, 46.3, -1.95), (-3.08, 44.2, -1.85), (-3.18, 42.2, -1.95), (-3.3, 40.7, -2.75), (-3.4, 39.0, -3.72), (-3.3, 37.1, -4.0)], 0.9),
                  ([(-1.7, 48.8, -0.6), (-2.75, 47.7, -1.25), (-3.1, 46.0, -1.2), (-3.2, 44.0, -1.1), (-3.35, 42.0, -1.2), (-3.75, 40.9, -2.3), (-4.0, 39.6, -3.3)], 0.7)],
        'hair11': [([(1.4, 48.84, -1.3), (2.55, 47.9, -1.95), (3.0, 46.3, -1.95), (3.1, 44.2, -1.85), (3.2, 42.3, -2.05), (3.3, 40.9, -3.1), (3.38, 39.2, -4.25), (3.25, 37.4, -4.5)], 0.88),
                   ([(1.8, 48.8, -0.6), (2.8, 47.7, -1.25), (3.12, 46.0, -1.2), (3.22, 44.0, -1.1), (3.4, 42.2, -1.3), (3.8, 41.2, -2.6)], 0.7)],
    }
    for bone, strands in bangs.items():
        b(bone, 'hair2', strands[0][0][0])
        for pts, w in strands:
            seed += 1
            B.ribbon(bone, catmull(pts, 5), taper(w, 0.8, 0.28, 0.6), 0.2, front, seg=1.15, tip=True, seed=seed,
                     shine=0.2, shinew=0.08, shinek=0.4, **H)

    # ---- long back mass
    wid = lambda t: 5.0 + 7.8 * smoothstep(0.0, 0.26, t)
    upts = catmull([(0, 44.6, 2.55 + HZ * 0.3), (0, 41.9, 3.3 + HZ), (0, 37.5, 3.85 + HZ), (0, 31.5, 3.8 + HZ),
                    (0, 25.0, 4.25 + HZ), (0, 20.5, 4.75 + HZ), (0, 19.0, 4.9 + HZ)], 6)
    B.ribbon('back_hair7', upts, wid, 0.42, (0, 0, 1), seg=3.2, seed=97, tone=0.60, **H)
    for xc, bones in COLUMNS:
        def pick(i, mid, t, bones=bones):
            return bones[0] if mid[1] > SPLIT[0] else (bones[1] if mid[1] > SPLIT[1] else bones[2])
        lay = [(-0.72, 1.85, 0.0, 0.92), (0.72, 1.7, 0.30, 1.0), (0.05, 0.85, 0.58, 1.08)]
        for i, (dx, w, zo, tone) in enumerate(lay):
            seed += 1
            yend = rs.uniform(11.8, 17.4) if i < 2 else rs.uniform(18.0, 27.0)
            pts = col_path(xc, dx + rs.uniform(-0.15, 0.15), zo, yend, rs)
            outv = norm((xc * 0.045, 0, 1.0))
            B.ribbon(pick, pts, taper(w, 0.7, 0.3, 0.74), 0.3, outv, seg=lambda f: 2.3 + 1.7 * f, tip=True,
                     roll=rs.uniform(-7, 7), seed=seed, tone=tone * rs.uniform(0.9, 1.08), shine=0.17, shinew=0.05, shinek=0.32, **H)

    # ---- locks falling over the front of each shoulder
    for s, up, low in ((1, 'hair3', 'hair4'), (-1, 'hair30', 'hair31')):
        zx = 0.5 if s == TOGA_SIDE else 0.0           # ride on top of the robe strap
        b(up, 'hair_long', (3.6 * s, 41.6, -0.5))
        b(low, up, (3.85 * s, 36.0, -3.9 - zx))
        for i, (dx, w, yend) in enumerate(((0.0, 1.15, 28.4), (0.85, 0.82, 32.0))):
            seed += 1
            x = lambda v: (v + dx) * s
            ctrl = [(x(2.9), 43.3, 1.7), (x(3.5), 42.15, 0.3), (x(3.9), 41.45, -1.4), (x(4.1), 40.4, -3.0 - zx * 0.6),
                    (x(4.0), 38.6, -3.85 - zx), (x(3.8), 36.0, -3.9 - zx), (x(3.62), 33.2, -3.68 - zx * 0.9),
                    (x(3.48), 30.8, -3.62 - zx * 0.7)]
            ctrl = [q for q in ctrl if q[1] > yend + 0.8] + [(x(3.34), yend, -3.72 - zx * 0.6)]
            pts = catmull(ctrl, 5)

            def outf(t, tan, mid):
                return (0.12 * s, np.clip((mid[1] - 40.6) / 1.4, 0, 1) * 1.6, -1.0 if mid[2] < 0.6 else 0.4)

            B.ribbon(lambda k, mid, t, up=up, low=low: up if mid[1] > 36.0 else low, pts, taper(w, 0.7, 0.3, 0.72), 0.3, outf,
                     seg=1.9, tip=True, seed=seed, tone=1.02, shine=0.3, shinew=0.06, shinek=0.35, **H)
