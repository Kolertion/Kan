"""Belt, pleated skirt with the front opening, and the robe draped over one shoulder."""
import math
import numpy as np
from core import catmull, norm, frame, rot_matrix
from rig import TOGA_SIDE

CD = 16.0
LV = [(27.8, 4.3, 2.9, 0.15), (17.8, 6.1, 4.75, 0.25), (8.4, 6.95, 5.65, 0.35)]     # y, rx, rz, centre z
SLIT = 32.0          # half angle of the front opening covered by the two flaps
NPAN = 21


def ell(phi, y, rx, rz, cz, k=1.0):
    a = math.radians(phi)
    p = np.array([rx * k * math.sin(a), y, cz - rz * k * math.cos(a)])
    n = norm((math.sin(a) / rx, 0, -math.cos(a) / rz))
    return p, n


def ring(B, bone, n, y, h, rx, rz, cz, thick, mat, start=0.0, overlap=1.08, **mp):
    for i in range(n):
        phi = start + i * 360.0 / n
        p, nrm = ell(phi, y, rx, rz, cz)
        p2, _ = ell(phi + 360.0 / n, y, rx, rz, cz)
        w = np.linalg.norm(p2 - p) * overlap
        B.obox(bone, p, frame((0, 1, 0), nrm), (w, h, thick), mat, u0=i * w, seed=i, **mp)


def skirt(B):
    C = dict(mat='cloth', dens=CD)
    step = (360 - 2 * SLIT) / NPAN
    for k in range(NPAN):
        phi = SLIT + (k + 0.5) * step
        g = k % 2 == 1
        name = f'tor_skirt{2 + k // 2:02d}' + ('g' if g else '')
        kk = 0.955 if g else 1.0
        P = [ell(phi, *LV[i], k=kk) for i in range(3)]
        W = [np.linalg.norm(ell(phi + step / 2, *LV[i])[0] - ell(phi - step / 2, *LV[i])[0]) for i in range(3)]
        B.bone(name, 'tor_skirt', P[0][0])
        B.bone(name + 'b', name, P[1][0])
        tone = 0.90 if g else 1.0
        wmul = 1.42 if g else 1.16
        for i, bone in ((0, name), (1, name + 'b')):
            a, c = P[i][0], P[i + 1][0]
            up = a - c
            R = frame(up, P[i][1] + P[i + 1][1])
            w = W[i + 1] * wmul
            mp = dict(hem=1.15) if i == 1 else {}
            B.obox(bone, (a + c) / 2, R, (w, np.linalg.norm(up) + 0.18, 0.26), u0=k * 1.9, seed=k, sa=i * 9.8, tone=tone, strip=0, **mp, **C)
            if i == 1:        # pointed hem tooth
                d = w / math.sqrt(2)
                B.obox(bone, c, R @ rot_matrix((0, 0, 45)), (d, d, 0.24), u0=k * 1.9, seed=k, sa=19.0, tone=tone * 0.97, edge=0.5, **C)
    # the two flaps that close the front and part into an inverted V
    for s, flap, gold in ((1, 'tor_front_flap_1', 'robe_gold_openwork_R'), (-1, 'tor_front_flap_-1', 'robe_gold_openwork_L')):
        top, mid, bot = np.array([1.24 * s, 27.8, -2.6]), np.array([2.2 * s, 17.8, -4.08]), np.array([2.92 * s, 8.4, -4.86])
        B.bone(flap, 'tor_skirt', top)
        B.bone(gold, flap, top)
        outv = (0.34 * s, 0, -1.0)
        for i, (a, c) in enumerate(((top, mid), (mid, bot))):
            up = a - c
            R = frame(up, outv)
            L = np.linalg.norm(up)
            mp = dict(hem=1.15) if i == 1 else {}
            B.obox(flap, (a + c) / 2, R, (2.5, L + 0.18, 0.26), u0=40 + s, seed=30 + s, sa=i * 9.8, trimx=-s, edge=0.12, **mp, **C)
            # gold piping on the opening edge with raised lozenges
            ex = R[:, 0] * (-s) * 1.24
            B.obox(gold, (a + c) / 2 + ex - R[:, 2] * 0.03, R, (0.16, L + 0.18, 0.36), 'gold', round=0.4, dens=CD)
            for j in range(4):
                q = a + (c - a) * (j + 0.5) / 4 + ex * 0.72 - R[:, 2] * 0.12
                B.obox(gold, q, R @ rot_matrix((0, 0, 45)), (0.42, 0.42, 0.16), 'gold', round=0.5, rim=1, rimw=0.08, dens=CD)
        R = frame(mid - bot, outv)
        for j in (-1, 0, 1):   # hem teeth
            B.obox(flap, bot + R[:, 0] * j * 0.83, R @ rot_matrix((0, 0, 45)), (0.6, 0.6, 0.24), seed=3 + j, edge=0.5, **C)


def belt(B):
    y, rx, rz, cz = 28.8, 4.62, 3.2, 0.15
    ring(B, 'thor_belt', 22, y, 3.2, rx, rz, cz, 0.42, 'leather', bands=3, round=0.0)
    ring(B, 'thor_belt', 22, y - 0.05, 1.12, rx + 0.14, rz + 0.14, cz, 0.3, 'leather', start=8.0, bands=1, tone=1.18)
    ring(B, 'belt_inner_lining', 14, y, 3.3, rx - 0.3, rz - 0.3, cz, 0.3, 'leather', tone=0.6)
    # clasp
    zc = cz - rz - 0.36
    B.box('thor_belt2', (0, y, zc - 0.08), (2.9, 1.2, 0.42), mat='gold', round=0.35, rim=1, rimw=0.16)
    for s in (1, -1):
        B.box('thor_belt2', (1.52 * s, y, zc - 0.1), (0.46, 1.7, 0.5), mat='gold', round=0.5, rim=1, rimw=0.1)
    B.box('thor_belt2', (0, y, zc - 0.26), (0.78, 0.78, 0.3), (0, 0, 45), mat='gold', round=0.45, rim=1, rimw=0.12)
    B.box('belt_inner_lining2', (0, y, zc + 0.14), (3.5, 1.5, 0.2), mat='gold', tone=0.62)
    for s in (1, -1):
        B.box('tunic_soft_waist_2', (0.62 * s, 27.0, cz - 3.2), (1.0, 0.7, 0.34), (0, 0, 8 * s), mat='cloth', round=0.5, dens=CD)
    # white waistband peeking out under the belt
    ring(B, 'tunic_soft_waist_71', 18, 27.12, 0.62, rx - 0.02, rz - 0.02, cz, 0.36, 'cloth', round=0.5, edge=0.4, dens=CD)


def toga(B):
    t = TOGA_SIDE
    C = dict(mat='cloth', dens=CD, strip=1)
    ytop = lambda xs: 41.78 - (xs - 3.3) * 0.30

    def outf(tt, tan, mid):          # normal swings from 'up' on the shoulder to straight out on chest/back
        top = float(np.clip((mid[1] - 39.6) / 1.5, 0, 1))
        return ((0.12 + 0.5 * top) * t, 2.0 * top, (mid[2] - 0.3) / 2.5)

    lp = lambda a, c, f: a + (c - a) * f
    for k in range(5):                 # front drape, fanning out towards the belt
        xs, xb = 3.3 + 0.72 * k, -0.9 + 1.25 * k
        zo = -0.16 if k % 2 else 0.0
        pts = catmull([(xs * t, ytop(xs) - 0.12, 1.0), (xs * t, ytop(xs), 0.2), ((xs - 0.1) * t, ytop(xs) - 0.5, -1.55), ((xs * 0.94 - 0.1) * t, 39.95, -2.98 + zo),
                       (lp(xs, xb, 0.35) * t, 38.1, -3.8 + zo), (lp(xs, xb, 0.62) * t, 35.4, -3.6 + zo),
                       (lp(xs, xb, 0.85) * t, 32.6, -3.46 + zo), (xb * t, 30.45, -3.34 + zo)], 5)
        B.ribbon('sash_mid_cloth_04', pts, lambda f: 0.98 + 0.62 * f, 0.3, outf, mat='cloth', seg=1.7, seed=k, dens=CD, strip=1, edge=0.42)
    for k in range(4):                 # back drape
        xs, xb = 3.4 + 0.9 * k, 0.2 + 1.2 * k
        zo = 0.14 if k % 2 else 0.0
        pts = catmull([(xs * t, ytop(xs), -0.2), (xs * t, ytop(xs) - 0.12, 1.0), (xs * t, ytop(xs) - 0.75, 2.2), (xs * 0.95 * t, 39.2, 3.42 + zo),
                       (lp(xs, xb, 0.4) * t, 36.5, 3.5 + zo), (lp(xs, xb, 0.75) * t, 33.0, 3.26 + zo), (xb * t, 30.45, 3.2 + zo)], 5)
        B.ribbon(lambda i, mid, f: 'sash_mid' if mid[1] > 40.6 else 'sash_mid_cloth_36', pts, lambda f: 1.15 + 0.4 * f, 0.26, outf, mat='cloth', seg=2.0, seed=10 + k, dens=CD, strip=1, edge=0.42)
    # cloth bunching where the robe is tucked into the belt
    for i, phi in enumerate((-8, 10, 28, 46, 64, 132, 152, 172)):
        p, n = ell(phi * t, 30.5, 4.45, 3.05, 0.15)
        bone = 'tor_tunic_gathered_waist' if i % 2 == 0 else 'tunic_soft_waist_71'
        B.obox(bone, p + n * 0.1, frame((0, 1, 0), n) @ rot_matrix((0, 0, (i % 3 - 1) * 7)), (1.3, 0.5, 0.6), 'cloth', round=0.55, seed=i, dens=CD)


def build(B):
    B.side = 1
    skirt(B)
    belt(B)
    toga(B)
