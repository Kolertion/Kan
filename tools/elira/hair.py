"""Hair: a cap on the skull, long ribbons that fall as a bell around the back and shoulders, and
the locks that frame the face. The ribbons are cut into tapering locks by the texture."""
import math
import numpy as np
from kit import support, catmull, resample, dir_ex, dir_n, norm
import paint

K12 = [30 * k for k in range(12)]
# bell the hair hangs on: height, half width at the sides, centre z, depth behind, depth in front
BELL = np.array([
    (38.45, 0.7, 0.15, 0.85, 0.7), (38.2, 1.5, 0.14, 1.7, 1.5), (37.85, 2.06, 0.12, 2.26, 2.06), (37.25, 2.46, 0.1, 2.61, 2.5),
    (36.3, 2.54, 0.1, 2.68, 2.5), (35.0, 2.6, 0.12, 2.62, 2.3), (33.5, 2.85, 0.2, 2.5, 2.1), (32.3, 3.35, 0.32, 2.3, 1.9),
    (31.0, 3.9, 0.45, 2.15, 1.9), (28.0, 4.5, 0.6, 2.1, 1.9), (24.0, 5.1, 0.7, 2.25, 1.9), (20.0, 5.8, 0.85, 2.45, 1.9),
    (16.0, 6.5, 1.0, 2.65, 1.9), (12.0, 7.1, 1.1, 2.9, 1.9), (9.0, 7.5, 1.15, 3.15, 1.9), (7.0, 7.7, 1.2, 3.3, 1.9)])
UPPER = [38.0, 37.7, 37.25, 36.5, 35.5, 34.4, 33.4, 32.5, 31.6, 30.5, 29.0, 27.0, 24.8]


def bell(psi, y, grow=0.0):
    """Point of the bell at azimuth psi; squarer around the skull, where it has to clear the cap."""
    X, zc, zb, zf = [float(np.interp(-y, -BELL[:, 0], BELL[:, k])) for k in (1, 2, 3, 4)]
    a = math.radians(psi)
    e = 2.0 / float(np.interp(y, [32.0, 35.5], [2.0, 3.2]))
    sx = math.copysign(abs(math.sin(a)) ** e, math.sin(a))
    cz = math.copysign(abs(math.cos(a)) ** e, math.cos(a))
    return np.array([(X + grow) * sx, y, zc - ((zf if cz > 0 else zb) + grow) * cz])


def strand(B, psi, y_end, spread, k, seed, grow=0.0, nl=3, tone=1.0, y_top=38.0):
    """One ribbon of the bell at azimuth psi, `spread` degrees wide, falling to y_end."""
    right = psi <= 180
    wing = 40 < psi < 138 or 222 < psi < 320
    joint = 22.0 if wing else 21.0                     # where the lower hair bone hinges
    ys = [y for y in UPPER + [joint - 2.4 * i for i in range(7)] if y_end + 0.9 < y <= y_top] + [y_end]
    C = np.array([bell(psi, y, grow - (0.2 if i == 0 else 0.0)) for i, y in enumerate(ys)])      # roots start under the shell
    tang = bell(psi + 4, 24.0) - bell(psi - 4, 24.0)
    ex = -norm(np.array([tang[0], 0.0, tang[2]]))
    half = np.array([0.5 * k * np.linalg.norm(bell(psi + spread / 2, y, grow) - bell(psi - spread / 2, y, grow)) for y in ys])
    side = '_r' if right else '_l'
    if wing:
        bone = lambda i, mid: f'hair_wing{side}' if mid[1] > joint else f'hair_wing{side}_lo'
    else:
        bone = lambda i, mid: 'hair_back' if mid[1] > joint else 'hair_back_lo'
    out = norm(dir_n(psi) + np.array([0.0, 0.7, 0.0]))
    return B.ribbon(bone, C, half, half, ex, out, 'hair', afn=paint.hair_alpha, seed=seed, nl=nl, tone=tone, ao=0.6, aob=0.5, thick=0.06)


def cap(B):
    """Shell of hair over the skull. Along the forehead and temples it is cut to the hair line by the
    texture, so it ends in strands rather than a rim."""
    from head import head_ring, K8
    prof = [(38.5, 0.22, -0.22, 0.3, 0.1, 0.1), (38.4, 0.75, -0.75, 0.9, 0.38, 0.38), (38.16, 1.4, -1.42, 1.6, 0.75, 0.75),
            (37.8, 1.96, -2.02, 2.2, 1.18, 1.15), (37.25, 2.34, -2.38, 2.56, 1.56, 1.45), (36.55, 2.44, -2.4, 2.66, 1.7, 1.5),
            (35.5, 2.46, -2.4, 2.56, 1.66, 1.42), (34.7, 2.32, -2.4, 2.14, 1.5, 1.2), (34.2, 1.92, -2.3, 1.42, 1.2, 1.0)]
    B.loft('head', K8, [head_ring(*p) for p in prof], 'hair_cap', afn=paint.cap_alpha, cap0=True, dens=36)


def lock(B, pts, phi, width, seed, bone, nl=2, taper=0.3, tone=1.0, cuff=None, joint=None):
    """Free lock along a hand-placed path; `phi` is the direction its flat side faces."""
    path = catmull(pts, 8)
    n = max(4, int(round(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / 1.25)))
    C = resample(path, n)
    if joint is not None and C[-1][1] < joint < C[0][1]:        # put a vertex exactly on the lower bone's hinge
        j = int(np.argmax(path[:, 1] < joint))
        f = (path[j - 1][1] - joint) / (path[j - 1][1] - path[j][1])
        C[1 + int(np.argmin(np.abs(C[1:-1, 1] - joint)))] = path[j - 1] + f * (path[j] - path[j - 1])
    t = np.linspace(0, 1, n + 1)
    half = np.array([width(v) for v in t]) / 2
    ex = dir_ex(phi)
    out = norm(dir_n(phi) + np.array([0.0, 0.7, 0.0]))
    made = B.ribbon(bone, C, half, half, ex, out, 'hair', afn=paint.hair_alpha, seed=seed, nl=nl, tip=taper, tone=tone, wob=0.03, ao=0.8, dens=34, thick=0.06)
    if cuff is not None:                       # gold cuff around the lock
        i = int(np.argmin(np.abs(C[:, 1] - cuff)))
        i = min(max(i, 1), n - 1)
        up = norm(C[i - 1] - C[i + 1])
        nrm = norm(np.cross(up, ex))
        nrm = nrm if nrm @ out > 0 else -nrm
        R = np.stack([-ex, np.cross(nrm, ex), -nrm], axis=1)
        bn = bone(i, C[i]) if callable(bone) else bone
        B.box(bn, C[i], (2 * half[i] + 0.12, 0.3, 0.2), R=R, mat='gold', lines=(C[i][1],))
    return made


def side_locks(B, s):
    """Locks falling in front of the shoulders on one side (s = +1 right, -1 left)."""
    sd = '_r' if s > 0 else '_l'
    m = lambda pts: [(x * s, y, z) for x, y, z in pts]
    bone = lambda i, mid: f'lock{sd}' if mid[1] > 29 else f'lock{sd}_lo'
    main = [(0.5, 37.95, -1.5), (1.4, 37.7, -2.2), (2.05, 36.9, -2.25), (2.4, 35.9, -1.95), (2.47, 34.7, -1.76), (2.44, 33.5, -1.68),
            (2.36, 32.3, -1.72), (2.24, 31.2, -1.94), (2.1, 30.0, -2.4), (2.02, 29.0, -2.82), (1.98, 27.6, -2.98), (2.05, 26.2, -2.86),
            (2.1, 24.5, -2.7), (2.15, 22.8 + 0.5 * (s < 0), -2.58), (2.2, 21.2 + 0.7 * (s < 0), -2.5)]
    lock(B, m(main), 40 * s, lambda t: 0.5 + 0.62 * min(1.0, t * 5) - 0.25 * t, 11 + (s < 0), bone, nl=3, taper=0.22, cuff=30.7, joint=29.0)
    outer = [(0.45, 37.98, -1.0), (1.6, 37.78, -1.5), (2.34, 37.0, -1.3), (2.6, 35.8, -0.95), (2.66, 34.4, -0.8), (2.66, 33.0, -0.76),
             (2.7, 31.8, -0.8), (2.74, 30.4, -1.1), (2.74, 28.6 - 0.6 * (s < 0), -1.4), (2.7, 26.4 - 0.6 * (s < 0), -1.5)]
    lock(B, m(outer), 66 * s, lambda t: 0.5 + 0.55 * min(1.0, t * 5), 21 + (s < 0), f'lock{sd}', nl=2, taper=0.3, tone=0.98)
    wisp = [(0.3, 37.95, -1.75), (0.95, 37.68, -2.42), (1.55, 37.0, -2.57), (1.95, 36.2, -2.44), (2.14, 35.3, -2.2), (2.12, 34.5 + 0.3 * (s < 0), -2.06)]
    lock(B, m(wisp), 30 * s, lambda t: 0.16 + 0.3 * min(1.0, t * 4), 31 + (s < 0), f'lock{sd}', nl=2, taper=0.45, tone=1.03)


def build(B):
    cap(B)
    # main bell behind the shoulders, both sides; ends at the calves with uneven tips
    seed = 100
    for psi, y_end in ((100, 11.8), (121, 10.2), (142, 9.4), (161, 10.0), (180, 8.3)):
        for sgn in ((1, -1) if psi != 180 else (1,)):
            seed += 1
            p = psi if sgn > 0 else 360 - psi
            strand(B, p, y_end + (0.6 if sgn < 0 else 0.0) * ((seed % 3) - 1), 21.5, 1.18, seed, nl=2, tone=0.97)
    # shorter upper layer that breaks up the curtain
    for psi, y_end in ((110, 25.0), (131, 22.0), (151, 20.4), (170.5, 23.6)):
        for sgn in (1, -1):
            seed += 1
            p = psi if sgn > 0 else 360 - psi
            strand(B, p, y_end + (0.9 if sgn < 0 else 0.0), 21.0, 0.78, seed, grow=0.12, nl=2, tone=1.03)
    for s in (1, -1):
        side_locks(B, s)
