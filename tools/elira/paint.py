"""Procedural paint for Elira. Colours follow the palette of the original texture.

Every material receives a sampling context (see kit.make_ctx): `pa` are rest-pose model coordinates
of the texels, so features are placed by position and run seamlessly across neighbouring cubes.
"""
import math
import numpy as np
from kit import smoothstep, catmull


def C(*c):
    return np.array(c, np.float32)


SKIN, SKIN_SH, SKIN_DEEP, SKIN_HL = C(247, 221, 209), C(229, 184, 176), C(190, 126, 132), C(255, 241, 233)
BLUSH = C(250, 158, 168)
HAIR, HAIR_MID, HAIR_SH = C(226, 221, 242), C(199, 192, 226), C(162, 152, 203)
HAIR_DEEP, HAIR_HL, HAIR_TIP = C(110, 98, 160), C(253, 251, 255), C(182, 166, 228)
DARK, DARK_DEEP, DARK_HL = C(52, 38, 66), C(27, 17, 37), C(92, 70, 116)
WHITE, WHITE_SH = C(243, 240, 248), C(196, 188, 216)
GOLD, GOLD_DK, GOLD_DEEP, GOLD_HL = C(216, 172, 78), C(156, 108, 40), C(96, 60, 24), C(255, 234, 152)
GEM, GEM_DEEP, GEM_HL = C(112, 44, 172), C(30, 10, 48), C(226, 176, 255)
LIP_UP, LIP_LO, LIP_LINE = C(196, 98, 124), C(224, 128, 148), C(124, 46, 72)
NAIL = C(134, 52, 86)
LASH = C(40, 22, 48)


# ---------------------------------------------------------------- helpers
def mix(rgb, col, a):
    a = np.asarray(a, np.float32)
    if a.ndim == rgb.ndim - 1:
        a = a[..., None]
    return rgb * (1 - a) + np.asarray(col, np.float32) * a


def fill(c, col):
    return np.broadcast_to(np.asarray(col, np.float32), c.pa.shape).copy()


def vnoise(p, freq=1.0, seed=0):
    """Trilinear value noise in [0, 1] for an (..., 3) array of points."""
    q = np.asarray(p, np.float64) * freq
    i = np.floor(q).astype(np.int64)
    f = q - i
    f = f * f * (3 - 2 * f)

    def h(ix, iy, iz):
        n = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFF) / 65535.0

    x0, y0, z0 = i[..., 0], i[..., 1], i[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                out = out + w * h(x0 + dx, y0 + dy, z0 + dz)
    return out


def n2(a, b, seed=0):
    return vnoise(np.stack([a, b, np.zeros_like(a)], -1), 1.0, seed)


def seg(x, y, a, b):
    """Distance to segment a-b and the position along it."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = np.clip(((x - a[0]) * vx + (y - a[1]) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
    return np.hypot(x - (a[0] + t * vx), y - (a[1] + t * vy)), t


def stroke(x, y, pts, w, aa):
    """Coverage of a polyline whose half width is interpolated between its points."""
    pts = np.asarray(pts, float)
    w = np.broadcast_to(np.asarray(w, float), (len(pts),))
    cov = np.zeros(x.shape, np.float32)
    for i in range(len(pts) - 1):
        d, t = seg(x, y, pts[i], pts[i + 1])
        ww = w[i] + (w[i + 1] - w[i]) * t
        cov = np.maximum(cov, 1 - smoothstep(ww - aa, ww + aa, d))
    return cov


def gauss(x, y, cx, cy, rx, ry):
    return np.exp(-(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))


def disc(x, y, cx, cy, rx, ry, aa):
    r = np.hypot((x - cx) / rx, (y - cy) / ry)
    k = aa / min(rx, ry)
    return 1 - smoothstep(1 - k, 1 + k, r)


def band(v, lo, hi, aa):
    return smoothstep(lo - aa, lo + aa, v) * (1 - smoothstep(hi - aa, hi + aa, v))


def tri(v):
    """Triangle wave in [0, 1] with period 1."""
    return np.abs(((v % 1.0) + 1.0) % 1.0 - 0.5) * 2


def spline_fn(pts):
    """y(x) through control points, as a callable for monotone x."""
    d = catmull(np.asarray(pts, float), 12)
    return lambda x: np.interp(x, d[:, 0], d[:, 1])


# ---------------------------------------------------------------- fabric, metal, gems
def lattice(u, v, period=0.62):
    """Diamond weave like the original skirt: 1 on the lattice lines, 0 inside the lozenges."""
    p, q = (u + v) / period, (u - v) / period
    return np.maximum(1 - smoothstep(0.06, 0.16, tri(p) * 0.5), 1 - smoothstep(0.06, 0.16, tri(q) * 0.5))


def dark_cloth(c, u, v, sheen=0.0):
    rgb = fill(c, DARK)
    rgb = mix(rgb, DARK_HL, 0.22 * lattice(u, v))
    rgb = mix(rgb, DARK_DEEP, 0.35 * n2(u * 0.6, v * 0.6, 4))
    return mix(rgb, DARK_HL, np.clip(sheen, 0, 0.6))


def dark(c):
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    return dark_cloth(c, x + z * 0.5, y, 0.25 * max(0.0, float(c.n[1])))


def gold_base(c, v=None):
    pa = c.pa
    n = vnoise(pa, 1.6, 21)
    rgb = fill(c, GOLD) * (0.92 + 0.16 * n)[..., None]
    rgb = mix(rgb, GOLD_HL, 0.45 * max(0.0, float(c.n[1])))
    rgb = mix(rgb, GOLD_DK, 0.45 * max(0.0, -float(c.n[1])))
    return rgb


def gold(c):
    rgb = gold_base(c)
    mp = c.mp
    y = c.pa[..., 1]
    for ly in mp.get('lines', ()):                    # engraved rings at fixed heights
        rgb = mix(rgb, GOLD_DEEP, 0.75 * (1 - smoothstep(0.02, 0.05, np.abs(y - ly))))
    if mp.get('inlay'):                               # dark enamel band between two heights
        lo, hi = mp['inlay']
        rgb = mix(rgb, DARK, band(y, lo, hi, c.px))
    if mp.get('studs'):                               # row of small bright studs
        ya, per = mp['studs']
        ang = np.arctan2(c.pa[..., 0], -c.pa[..., 2]) * mp.get('studr', 3.0)
        d = np.hypot((tri(ang / per) - 0.5) * per * 0.5, y - ya)
        rgb = mix(rgb, GOLD_HL, 1 - smoothstep(0.05, 0.09, d))
    rgb = mix(rgb, GOLD_DK, np.clip(mp.get('shade', 0.0), 0, 1))
    return rgb


def gem(c):
    """Faceted amethyst painted in the cube's own plane, with a four-point sparkle."""
    s = c.size
    x, y = c.L[..., 0] / max(s[0], 1e-6) * 2, c.L[..., 1] / max(s[1], 1e-6) * 2
    if c.cube.poly is not None:
        lo, hi = c.cube.lo, c.cube.hi
        x = (c.L[..., 0] - (lo[0] + hi[0]) / 2) / max(s[0], 1e-6) * 2
        y = (c.L[..., 1] - (lo[1] + hi[1]) / 2) / max(s[1], 1e-6) * 2
    r = np.hypot(x, y)
    ang = np.arctan2(y, x)
    rgb = mix(fill(c, GEM), GEM_DEEP, smoothstep(0.1, 1.0, r) * 0.8)
    facet = tri(ang / (2 * math.pi) * 8)
    rgb = mix(rgb, GEM_HL, 0.28 * (1 - smoothstep(0.0, 0.25, facet)) * smoothstep(0.25, 0.7, r))
    rgb = mix(rgb, GEM, 0.6 * (1 - smoothstep(0.15, 0.5, r)))
    sx, sy = x + 0.28, y - 0.3
    star = np.maximum(np.exp(-(np.abs(sx) * 9 + np.abs(sy) * 1.6)), np.exp(-(np.abs(sx) * 1.6 + np.abs(sy) * 9)))
    rgb = mix(rgb, C(255, 240, 255), np.clip(star * 1.5, 0, 0.95))
    return mix(rgb, GEM_HL, 0.5 * gauss(x, y, 0.3, -0.35, 0.3, 0.2))


def nail(c):
    rgb = fill(c, NAIL)
    return mix(rgb, C(214, 130, 160), 0.5 * smoothstep(0.1, 0.5, c.L[..., 1] / max(c.size[1], 1e-6)))


def flat(col):
    return lambda c: fill(c, col)


# ---------------------------------------------------------------- face
EYE_X, EYE_Y = 1.0, 35.55
MOUTH_Y = 33.93
EYE_K = 0.93
UP_LID = spline_fn([(-0.61, -0.115), (-0.47, 0.025), (-0.27, 0.17), (-0.03, 0.24), (0.22, 0.228), (0.43, 0.168), (0.61, 0.085)])
LO_LID = spline_fn([(-0.61, -0.115), (-0.42, -0.16), (-0.17, -0.19), (0.09, -0.185), (0.33, -0.13), (0.50, -0.04), (0.61, 0.085)])


def hair_edge(x, z):
    """Height of the hair line around the head, by direction from the head's axis (0 = straight ahead)."""
    th = np.degrees(np.arctan2(np.abs(x), -z))
    return np.interp(th, [0, 20, 35, 50, 65, 78, 88, 100], [37.24, 37.12, 36.9, 36.5, 35.9, 35.05, 34.3, 33.5])


def cap_alpha(c):
    """The hair shell ends in fine strands along the hair line instead of a hard rim."""
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    th = np.degrees(np.arctan2(np.abs(x), -z))
    strands = 0.1 * tri(th / 6.5) ** 1.5 + 0.05 * tri(th / 2.6 + 0.3)
    return (y > hair_edge(x, z) - strands).astype(np.float32)


def paint_eye(rgb, ex, ey, sx, aa):
    """ex > 0 points to the temple; sx is +1 for her right eye (model +X)."""
    yu, yl = UP_LID(ex), LO_LID(ex)
    opening = smoothstep(-aa, aa, np.minimum(yu - ey, ey - yl)) * (np.abs(ex) < 0.62)
    # eye shadow above the lid, strongest over the outer half
    above = ey - yu
    sh = np.exp(-(np.clip(above, 0, None) / 0.2) ** 2) * (above > -0.02) * band(ex, -0.62, 0.86, 0.08)
    rgb = mix(rgb, C(178, 140, 204), 0.5 * sh * (0.5 + 0.5 * smoothstep(-0.3, 0.5, ex)))
    crease = np.stack([np.linspace(-0.4, 0.52, 12), UP_LID(np.linspace(-0.4, 0.52, 12)) + 0.1], 1)
    rgb = mix(rgb, C(176, 112, 140), 0.55 * stroke(ex, ey, crease, [0.004] + [0.012] * 10 + [0.004], aa))
    # white of the eye with the shadow of the lid
    eye = fill_like(rgb, C(251, 249, 252))
    eye = mix(eye, C(238, 208, 214), 0.6 * smoothstep(0.38, 0.6, np.abs(ex)))
    # iris
    dx, dy = ex, ey - 0.035
    rx, ry = 0.25, 0.28
    r = np.hypot(dx / rx, dy / ry)
    t = np.clip((dy / ry + 1) / 2, 0, 1)
    iris = mix(fill_like(rgb, C(196, 146, 240)), C(62, 28, 112), smoothstep(0.12, 0.92, t))
    iris = mix(iris, C(138, 82, 204), 0.5 * band(t, 0.3, 0.62, 0.12))
    iris = mix(iris, C(236, 212, 255), 0.6 * np.exp(-((dy / ry + 0.64) / 0.2) ** 2) * (r < 0.95))
    ang = np.arctan2(dy / ry, dx / rx)
    iris = iris * (0.93 + 0.14 * n2(ang * 5.0, r * 2.0, 5))[..., None]
    iris = mix(iris, C(46, 20, 84), 0.9 * smoothstep(0.78, 0.99, r))
    iris = mix(iris, C(24, 9, 42), disc(dx, dy, 0.0, 0.02, 0.098, 0.128, aa * 1.5))
    eye = mix(eye, iris, 1 - smoothstep(1 - aa / rx, 1 + aa / rx, r))
    eye = eye * (1 - 0.42 * smoothstep(yu - 0.24, yu - 0.01, ey))[..., None]          # lid shadow
    eye = mix(eye, C(255, 255, 255), 0.97 * disc(dx, dy, 0.105 * sx, 0.125, 0.074, 0.094, aa * 1.5))
    eye = mix(eye, C(255, 255, 255), 0.85 * disc(dx, dy, -0.115 * sx, -0.125, 0.04, 0.045, aa * 1.5))
    eye = mix(eye, C(244, 220, 255), 0.6 * disc(dx, dy, 0.03 * sx, -0.2, 0.024, 0.024, aa * 1.5))
    rgb = mix(rgb, eye, opening)
    # lashes
    xs = np.linspace(-0.61, 0.61, 14)
    lash = np.stack([xs, UP_LID(xs) + 0.02], 1)
    lw = np.interp(xs, [-0.61, -0.2, 0.3, 0.61], [0.016, 0.046, 0.058, 0.05])
    cov = stroke(ex, ey, lash, lw, aa)
    cov = np.maximum(cov, stroke(ex, ey, [(0.56, 0.11), (0.72, 0.17), (0.86, 0.26)], [0.05, 0.03, 0.004], aa))
    for a, b in (((0.50, 0.19), (0.66, 0.37)), ((0.36, 0.25), (0.45, 0.42)), ((0.62, 0.14), (0.83, 0.33))):
        mid = ((a[0] + b[0]) / 2 + 0.02, (a[1] + b[1]) / 2 - 0.02)
        cov = np.maximum(cov, stroke(ex, ey, [a, mid, b], [0.022, 0.014, 0.003], aa))
    rgb = mix(rgb, LASH, cov)
    xs = np.linspace(-0.1, 0.61, 9)
    low = stroke(ex, ey, np.stack([xs, LO_LID(xs) - 0.008], 1), np.interp(xs, [-0.1, 0.35, 0.61], [0.003, 0.012, 0.02]), aa)
    for a, b in (((0.33, -0.17), (0.37, -0.25)), ((0.46, -0.1), (0.53, -0.18))):
        low = np.maximum(low, stroke(ex, ey, [a, b], [0.011, 0.003], aa))
    rgb = mix(rgb, C(106, 62, 96), 0.9 * low)
    # brow
    brow = stroke(ex, ey, catmull([(-0.66, 0.6), (-0.34, 0.715), (0.06, 0.79), (0.44, 0.775), (0.88, 0.63)], 6),
                  np.interp(np.linspace(0, 1, 25), [0, 0.3, 0.6, 1], [0.045, 0.06, 0.048, 0.008]), aa)
    return mix(rgb, C(134, 122, 170), 0.94 * brow)


def fill_like(rgb, col):
    return np.broadcast_to(np.asarray(col, np.float32), rgb.shape).copy()


def paint_face(rgb, x, y, aa):
    # blush and soft modelling
    for s in (1, -1):
        rgb = mix(rgb, BLUSH, 0.3 * gauss(x, y, s * 1.2, 34.78, 0.5, 0.25))
    rgb = mix(rgb, SKIN_HL, 0.3 * gauss(x, y, 0, 35.0, 0.055, 0.32))
    rgb = mix(rgb, SKIN_SH, 0.2 * gauss(x, y, -0.15, 34.86, 0.05, 0.24))
    rgb = mix(rgb, SKIN_SH, 0.3 * gauss(x, y, 0, 34.52, 0.17, 0.035))
    for s in (1, -1):
        rgb = mix(rgb, SKIN_DEEP, 0.45 * disc(x, y, s * 0.085, 34.6, 0.034, 0.018, aa * 2))
        rgb = paint_eye(rgb, (x - s * EYE_X) * s / EYE_K, (y - EYE_Y) / EYE_K, s, aa / EYE_K)
    # lips
    hw = 0.47
    ax = np.abs(x)
    yt = MOUTH_Y + 0.08 * (1 - np.clip(ax / hw, 0, 1) ** 2.2) + 0.022 * np.exp(-((ax - 0.12) / 0.075) ** 2) - 0.02 * np.exp(-(x / 0.055) ** 2)
    ym = MOUTH_Y - 0.012 + 0.032 * (ax / hw) ** 2 - 0.008 * np.exp(-(x / 0.1) ** 2)
    yb = np.where(ax < hw * 0.86, MOUTH_Y - 0.135 * np.clip(1 - (ax / (hw * 0.86)) ** 2.4, 0, 1) ** 0.9, ym)
    soft = aa * 2.5
    inx = 1 - smoothstep(hw - 0.03, hw + 0.01, ax)
    upper = smoothstep(-soft, soft, y - ym) * (1 - smoothstep(-soft, soft, y - yt)) * inx
    lower = smoothstep(-soft, soft, y - yb) * (1 - smoothstep(-soft, soft, y - ym)) * inx
    rgb = mix(rgb, SKIN_SH, 0.35 * gauss(x, y, 0, MOUTH_Y - 0.215, 0.24, 0.045))
    rgb = mix(rgb, LIP_UP, 0.95 * upper)
    lo_col = mix(fill_like(rgb, LIP_LO), C(255, 216, 222), 0.75 * gauss(x, y, 0.05, MOUTH_Y - 0.07, 0.11, 0.022))
    rgb = mix(rgb, lo_col, 0.95 * lower)
    line = (1 - smoothstep(0.006, 0.016, np.abs(y - ym))) * (1 - smoothstep(hw - 0.01, hw + 0.03, ax))
    rgb = mix(rgb, LIP_LINE, 0.9 * line)
    for s in (1, -1):
        rgb = mix(rgb, LIP_LINE, 0.55 * gauss(x, y, s * (hw + 0.01), MOUTH_Y + 0.02, 0.03, 0.022))
    return rgb


# ---------------------------------------------------------------- hair
def cap_strands(c, seed=0):
    """Strand coordinate on the scalp: front-to-back on top, arcs over the forehead, vertical behind."""
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    r = np.hypot(x, y - 35.3)
    wb = smoothstep(1.0, 2.2, z)
    return (z + 0.85 * r) * (1 - wb) + (x * 1.1 + 4.0) * wb


def hair_cap(c):
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    u = cap_strands(c)
    st = n2(u * 5.2, (np.abs(x) + 38.4 - y) * 0.16, 31)
    st2 = n2(u * 12.0, (np.abs(x) + 38.4 - y) * 0.22, 37)
    rgb = mix(fill(c, HAIR), HAIR_MID, 0.55 * smoothstep(0.35, 0.75, st))
    rgb = mix(rgb, HAIR_SH, 0.5 * (1 - smoothstep(0.12, 0.3, st2)))
    rgb = mix(rgb, HAIR_SH, 0.5 * smoothstep(36.4, 34.2, y))
    yc = 37.62 + 0.1 * tri(u * 2.6)                                    # zig-zag gloss ring
    rgb = mix(rgb, HAIR_HL, 0.8 * np.exp(-((y - yc) / 0.13) ** 2) * (0.45 + 0.55 * st))
    part = (1 - smoothstep(0.02, 0.07, np.abs(x))) * (z < 1.4) * (y > 37.0)
    rgb = mix(rgb, HAIR_DEEP, 0.7 * part)
    return rgb * c.mp.get('tone', 1.0)


def _locks(c):
    """Split a hair ribbon into a few overlapping locks with staggered, tapering tips."""
    if 'locks' in c.cache:
        return c.cache['locks']
    mp, u, s, S = c.mp, c.u, c.s, c.S
    K = mp.get('nl', 3)
    rs = np.random.RandomState(mp.get('seed', 0) * 7 + 3)
    edges = np.linspace(-1, 1, K + 1)
    longest = rs.randint(K)
    tipz = mp.get('tip', 0.26) * S
    idx = np.full(u.shape, -1, np.int32)
    v = np.zeros(u.shape, np.float32)
    hw_units = np.ones(u.shape, np.float32)
    taper = np.zeros(u.shape, np.float32)
    tone = np.ones(u.shape, np.float32)
    for j in rs.permutation(K):
        cj, hw = (edges[j] + edges[j + 1]) / 2, (edges[j + 1] - edges[j]) / 2 * 1.3
        Lj = S if j == longest else S * (1 - rs.uniform(0.03, mp.get('stagger', 0.2)))
        Tj = tipz * rs.uniform(0.75, 1.15)
        f = np.clip((s - (Lj - Tj)) / Tj, 0, 1)
        wj = hw * (1 - f ** 1.6)
        cc = cj * (1 - 0.35 * f) + mp.get('wob', 0.05) * np.sin(s * rs.uniform(0.3, 0.55) + rs.uniform(0, 6.28))
        vv = (u - cc) / np.maximum(wj, 1e-4)
        m = (np.abs(vv) < 1) & (s <= Lj)
        idx[m], v[m], taper[m] = j, vv[m], f[m]
        hw_units[m] = (wj * c.a)[m]
        tone[m] = rs.uniform(0.95, 1.05)
    c.cache['locks'] = (idx, v, hw_units, taper, tone)
    return c.cache['locks']


def hair_alpha(c):
    return (_locks(c)[0] >= 0).astype(np.float32)


def hair_strip(c):
    idx, v, hw, taper, tone = _locks(c)
    mp = c.mp
    y = c.pa[..., 1]
    seed = mp.get('seed', 0)
    inner = c.face == 'south'
    st = n2((v * 1.6 + idx * 2.7 + seed) * 2.4, c.s * 0.085, 41)
    st2 = n2((v * 1.6 + idx * 2.7 + seed) * 6.0, c.s * 0.12, 43)
    rgb = mix(fill(c, HAIR), HAIR_MID, 0.42 * smoothstep(0.3, 0.75, st))
    rgb = mix(rgb, HAIR_SH, 0.2 * (1 - smoothstep(0.1, 0.28, st2)))
    rgb = mix(rgb, HAIR_SH, 0.5 * np.abs(v) ** 3)                         # each lock is rounder at its edges
    rgb = mix(rgb, HAIR_TIP, 0.5 * smoothstep(0.5, 1.0, c.s / max(c.S, 1e-6)))
    if not inner:
        yc = 37.62 + 0.1 * tri(v * 0.9 + idx * 0.37)
        rgb = mix(rgb, HAIR_HL, 0.8 * np.exp(-((y - yc) / 0.14) ** 2) * (0.4 + 0.6 * st))
        sheen = mp.get('sheen', 27.5)
        zz = sheen + 0.9 * tri(v * 0.8 + idx * 0.41)
        rgb = mix(rgb, HAIR_HL, 0.42 * np.exp(-((y - zz) / 1.3) ** 2) * (0.25 + 0.75 * st) * (1 - np.abs(v) ** 2))
    line = smoothstep(1 - 1.3 * c.px / np.maximum(hw, 1e-4), 1.0, np.abs(v))
    rgb = mix(rgb, HAIR_SH * 0.86, 0.6 * line)
    rgb = rgb * tone[..., None] * mp.get('tone', 1.0)
    if inner:
        rgb = rgb * C(0.87, 0.86, 0.92)
    return rgb


# ---------------------------------------------------------------- skin
def strap_mask(x, y, aa):
    """Halter straps running from the cups up to the choker, as seen from the front."""
    ax = np.abs(x)
    d, _ = seg(ax, y, (1.86, 29.85), (0.56, 32.3))
    inside = 1 - smoothstep(0.19 - aa, 0.19 + aa, d)
    edge = inside * smoothstep(0.115 - aa, 0.115 + aa, d)
    return inside, edge


def cup_masks(x, y, aa):
    """Triangle cups: (fabric, gold trim) coverage in the front projection."""
    ax = np.abs(x)
    f1 = -0.67 * (ax - 0.25) + 0.74 * (y - 28.2)        # neckline, bare above
    f2 = 0.94 * (ax - 1.98) + 0.34 * (y - 29.8)         # outer edge, bare beyond
    f = np.maximum(f1, f2)
    fabric = 1 - smoothstep(-aa, aa, f)
    trim = fabric * smoothstep(-0.17 - aa, -0.17 + aa, f)
    return fabric, trim


def skin(c):
    mp = c.mp
    pa = c.pa
    x, y, z = pa[..., 0], pa[..., 1], pa[..., 2]
    aa = c.px * 0.75
    reg = mp.get('reg')
    rgb = fill(c, SKIN) * (1 + 0.02 * (vnoise(pa, 0.7, 3) - 0.5))[..., None]
    rgb = mix(rgb, SKIN_SH, 0.18 * max(0.0, -float(c.n[1])))
    rgb = mix(rgb, SKIN_HL, 0.14 * max(0.0, float(c.n[1])))
    front, back = c.n[2] < -0.25, c.n[2] > 0.25
    if reg == 'torso':
        rgb = torso_paint(c, rgb, x, y, z, aa, front, back)
    elif reg == 'bust':
        fabric, trim = cup_masks(x, y, aa)
        under = smoothstep(27.9, 27.5, y)                # the underside is always covered
        fabric, trim = np.maximum(fabric, under), trim * (1 - under)
        cloth = dark_cloth(c, x * 1.2, y * 1.2, 0.3 * max(0.0, float(c.n[1])))
        cloth = mix(cloth, DARK_DEEP, 0.5 * smoothstep(28.3, 27.4, y))
        rgb = mix(rgb, SKIN_SH, 0.3 * gauss(np.abs(x), y, 0.0, 28.5, 0.45, 0.9))        # cleavage
        rgb = mix(rgb, cloth, fabric)
        rgb = mix(rgb, gold_base(c), trim)
        s_in, s_edge = strap_mask(x, y, aa)
        s_in = s_in * (y > 29.6)
        rgb = mix(rgb, DARK, s_in)
        rgb = mix(rgb, GOLD, s_edge * (y > 29.6))
    elif reg == 'neck':
        if front:
            s_in, s_edge = strap_mask(x, y, aa)
            rgb = mix(rgb, DARK, s_in)
            rgb = mix(rgb, GOLD, s_edge)
        rgb = mix(rgb, SKIN_SH, 0.35 * smoothstep(32.6, 33.5, y) * (z < 0.3))           # shadow under the jaw
    elif reg == 'head':
        rgb = head_paint(c, rgb, x, y, z, aa)
    elif reg == 'arm':
        s = mp.get('side', 1)
        rgb = mix(rgb, BLUSH, 0.16 * gauss(y, z, 25.4, 0.9, 0.7, 0.7) * (c.n[2] > 0))      # elbow
        rgb = mix(rgb, BLUSH, 0.12 * gauss(y, x * s, 31.0, 4.0, 0.6, 0.7))                 # shoulder
        rgb = mix(rgb, SKIN_SH, 0.25 * (1 - smoothstep(0.015, 0.05, np.abs(y - 25.42))) * (c.n[2] < -0.2))
    elif reg == 'hand':
        rgb = mix(rgb, BLUSH, 0.14 * gauss(y, z, 19.3, 0.25, 0.25, 0.9))
    elif reg == 'leg':
        rgb = leg_paint(c, rgb, x, y, z, aa)
    elif reg == 'foot':
        rgb = mix(rgb, BLUSH, 0.12 * smoothstep(1.2, 0.5, y))
    elif reg == 'ear':
        rgb = mix(rgb, SKIN_SH, 0.5 * gauss(y, z, 35.3, 0.12, 0.28, 0.16))
    elif reg == 'nose':
        rgb = mix(rgb, SKIN_HL, 0.25 * smoothstep(34.9, 34.7, y) * (float(c.n[1]) > -0.3))
        rgb = mix(rgb, SKIN_SH, 0.22 * (float(c.n[1]) < -0.5))
    return rgb * mp.get('tone', 1.0)


def torso_paint(c, rgb, x, y, z, aa, front, back):
    ax = np.abs(x)
    wb = smoothstep(-0.3, 0.9, z)
    # high-cut brief under the belt
    leg_line = (19.75 + 2.5 * (ax / 3.35) ** 1.6) * (1 - wb) + (19.75 + 2.3 * (ax / 3.45) ** 2.7) * wb
    brief = smoothstep(-aa, aa, y - leg_line) * (1 - smoothstep(22.5, 22.6, y))
    cloth = dark_cloth(c, x + z * 0.6, y, 0.2 * max(0.0, float(c.n[1])))
    rgb = mix(rgb, SKIN_SH, 0.3 * band(y - leg_line, -0.16, 0.0, 0.05))
    rgb = mix(rgb, cloth, brief)
    rgb = mix(rgb, gold_base(c), brief * (1 - smoothstep(0.06, 0.06 + 2 * aa, y - leg_line)))
    # gold band under the bust, and the back strap of the top
    rgb = mix(rgb, gold_base(c), band(y, 27.02, 27.24, aa))
    rgb = mix(rgb, GOLD_DEEP, 0.6 * band(y, 27.115, 27.145, aa))
    side = smoothstep(-1.3, -0.7, z)
    strap = band(y, 28.04, 28.62, aa) * side
    rgb = mix(rgb, cloth, strap)
    rgb = mix(rgb, gold_base(c), strap * (1 - band(y, 28.11, 28.55, aa)))
    if front:
        s_in, s_edge = strap_mask(x, y, aa)
        rgb = mix(rgb, DARK, s_in)
        rgb = mix(rgb, GOLD, s_edge)
        # collar bones and the hollow of the throat
        for s in (1, -1):
            cb = [(s * 0.3, 31.02), (s * 1.2, 31.14), (s * 2.1, 31.34), (s * 2.85, 31.46)]
            rgb = mix(rgb, SKIN_HL, 0.5 * stroke(x, y, cb, [0.03, 0.05, 0.05, 0.02], 0.03))
            rgb = mix(rgb, SKIN_SH, 0.42 * stroke(x, y - 0.0, [(p[0], p[1] - 0.11) for p in cb], [0.03, 0.055, 0.05, 0.02], 0.04))
        rgb = mix(rgb, SKIN_SH, 0.4 * gauss(x, y, 0, 30.98, 0.16, 0.12))
        # midline, navel and the jewel that hangs above it
        rgb = mix(rgb, SKIN_SH, 0.2 * gauss(x, y, 0, 25.8, 0.07, 1.1))
        rgb = mix(rgb, SKIN_HL, 0.18 * (gauss(x, y, 0.75, 25.6, 0.4, 1.0) + gauss(x, y, -0.75, 25.6, 0.4, 1.0)))
        rgb = mix(rgb, SKIN_DEEP, 0.75 * disc(x, y, 0, 24.22, 0.075, 0.15, 0.03))
        rgb = mix(rgb, SKIN_HL, 0.4 * gauss(x, y, 0, 24.45, 0.12, 0.06))
        chain = (1 - smoothstep(0.022, 0.022 + 2 * aa, ax)) * band(y, 24.98, 27.0, aa) * (tri(y / 0.13) > 0.25)
        rgb = mix(rgb, GOLD, chain)
        loz = np.abs(x) / 0.2 + np.abs(y - 26.72) / 0.26
        rgb = mix(rgb, gold_base(c), 1 - smoothstep(1 - 8 * aa, 1, loz))
        rgb = mix(rgb, GOLD_DEEP, 0.6 * band(loz, 0.45, 0.6, 0.05))
        drop = np.hypot(x / 0.11, (y - 24.82) / (0.17 + 0.1 * (y > 24.82)))
        rgb = mix(rgb, gold_base(c), 1 - smoothstep(1.15, 1.3, drop))
        rgb = mix(rgb, GEM, 1 - smoothstep(0.8, 0.95, drop))
        rgb = mix(rgb, GEM_HL, 0.8 * gauss(x, y, 0.03, 24.86, 0.03, 0.04))
    if back:
        rgb = mix(rgb, SKIN_SH, 0.26 * gauss(x, y, 0, 27.0, 0.09, 3.6) * (y > 22.9))
        for s in (1, -1):
            rgb = mix(rgb, SKIN_SH, 0.16 * gauss(x, y, s * 1.45, 29.3, 0.5, 0.9))
            rgb = mix(rgb, SKIN_SH, 0.3 * gauss(x, y, s * 0.8, 23.0, 0.13, 0.13))
    return rgb


def head_paint(c, rgb, x, y, z, aa):
    nz = float(c.n[2])
    if nz < -0.3:
        rgb = paint_face(rgb, x, y, aa)
    # scalp: hair-coloured wherever the hair grows, so nothing bald shows under the shell or between locks
    m = smoothstep(-0.02, 0.02, y - hair_edge(x, z) - 0.07)
    if float(c.n[1]) > 0.6:
        m = np.ones_like(y)
    scalp = hair_cap(c) * C(0.86, 0.85, 0.9)
    return mix(rgb, scalp, np.clip(m, 0, 1))


def leg_paint(c, rgb, x, y, z, aa):
    s = c.mp.get('side', 1)
    cx = np.interp(y, [3.9, 11.9, 20.6], [1.45, 1.55, 1.72])
    cz = np.interp(y, [3.9, 6.5, 9.9, 11.9, 20.6], [0.25, 0.16, -0.02, -0.15, 0.05])
    psi = np.arctan2((x * s - cx), -(z - cz))                 # 0 at the front of the leg
    rad = np.hypot(x * s - cx, z - cz)
    # knee
    rgb = mix(rgb, BLUSH, 0.16 * gauss(y, psi, 11.95, 0.0, 0.7, 0.8))
    rgb = mix(rgb, SKIN_SH, 0.2 * gauss(y, psi, 11.3, 0.0, 0.16, 0.5))
    rgb = mix(rgb, SKIN_SH, 0.2 * gauss(y, psi, 12.6, 0.0, 0.16, 0.5))
    # gilded sandal laces: rings joined by crosses up the shin
    ring_y = [4.75, 6.35, 7.95, 9.55, 10.95]
    gw = 0.062
    cov = np.zeros(y.shape, np.float32)
    for ry in ring_y:
        cov = np.maximum(cov, 1 - smoothstep(gw - aa, gw + aa, np.abs(y - ry)))
    for a, b in zip(ring_y[:-1], ring_y[1:]):
        f = (y - a) / (b - a)
        inb = (f > 0) & (f < 1)
        for sg in (1, -1):
            d = np.abs(psi - sg * 1.05 * (2 * f - 1)) * rad
            cov = np.maximum(cov, (1 - smoothstep(gw - aa, gw + aa, d * 0.8)) * inb * (np.abs(psi) < 1.25))
    shadow = np.zeros(y.shape, np.float32)
    for ry in ring_y:
        shadow = np.maximum(shadow, band(ry - y, 0.03, 0.2, 0.05))
    rgb = mix(rgb, SKIN_SH, 0.4 * shadow * (y < 11.2))
    lace = mix(gold_base(c), GOLD_DEEP, 0.3 * (tri(y / 0.09) < 0.3))
    return mix(rgb, lace, cov * (y < 11.1) * (y > 4.6))


# ---------------------------------------------------------------- jewellery painted on cut-out plates
def _earring(c):
    lo, hi = c.cube.lo, c.cube.hi
    x = c.L[..., 0] - (lo[0] + hi[0]) / 2
    y = c.L[..., 1] - (lo[1] + hi[1]) / 2
    aa = c.px
    stud = disc(x, y, 0, 0.46, 0.085, 0.085, aa)
    chain = (1 - smoothstep(0.02, 0.02 + aa, np.abs(x))) * band(y, 0.02, 0.4, aa)
    drop = np.hypot(x / 0.13, (y + 0.22) / (0.2 + 0.12 * (y > -0.22)))
    rim = 1 - smoothstep(1.0, 1.0 + aa / 0.13, drop)
    stone = 1 - smoothstep(0.72, 0.72 + aa / 0.13, drop)
    return x, y, np.maximum(np.maximum(stud, chain), rim), stone


def earring_alpha(c):
    return _earring(c)[2]


def earring(c):
    x, y, _, stone = _earring(c)
    g = mix(fill(c, GEM), GEM_DEEP, 0.6 * smoothstep(-0.1, -0.42, y))
    g = mix(g, GEM_HL, 0.85 * gauss(x, y, -0.03, -0.13, 0.035, 0.05))
    return mix(gold_base(c), g, stone)


def _tikka(c):
    x, y = c.pa[..., 0], c.pa[..., 1]
    aa = c.px
    ax = np.abs(x)
    yc = 37.05 - 0.3 * (ax / 1.42) ** 1.6                  # side chains swag just under the hair line
    cov = (1 - smoothstep(0.022, 0.022 + aa, np.abs(y - yc))) * (ax < 1.44)
    beads = np.zeros(x.shape, np.float32)
    for bx in (0.38, 0.76, 1.12):
        by = 37.05 - 0.3 * (bx / 1.42) ** 1.6 - 0.075
        beads = np.maximum(beads, disc(ax, y, bx, by, 0.05, 0.068, aa))
    cov = np.maximum(cov, (1 - smoothstep(0.022, 0.022 + aa, ax)) * band(y, 36.9, 37.06, aa))
    drop = np.hypot(x / 0.125, (y - 36.72) / (0.16 + 0.1 * (y > 36.72)))
    cov = np.maximum(np.maximum(cov, beads), 1 - smoothstep(1.0, 1.0 + aa / 0.125, drop))
    stone = np.maximum(beads, 1 - smoothstep(0.7, 0.7 + aa / 0.125, drop))
    return x, y, cov, stone


def tikka_alpha(c):
    return _tikka(c)[2]


def tikka(c):
    x, y, _, stone = _tikka(c)
    g = mix(fill(c, GEM), GEM_HL, 0.6 * gauss(x, y, 0.03, 36.76, 0.04, 0.05))
    return mix(gold_base(c), g, stone)


# ---------------------------------------------------------------- outfit
def panel_alpha(c):
    mp = c.mp
    return (c.pa[..., 1] >= mp['tip'] + mp['hv'] * np.abs(c.u) ** 1.15).astype(np.float32)


def panel(c):
    """Skirt gore: patterned cloth, gold border with a row of lozenges, pointed hem."""
    mp, u, a, aa = c.mp, c.u, c.a, c.px
    y = c.pa[..., 1]
    kind, seed = mp['kind'], mp.get('seed', 0)
    hem = mp['tip'] + mp['hv'] * np.abs(u) ** 1.15
    dl = (1 + u) * a if mp.get('bl') else np.full(u.shape, 1e3)
    dr = (1 - u) * a if mp.get('br') else np.full(u.shape, 1e3)
    dh = (y - hem) * 0.72
    d = np.minimum(np.minimum(dl, dr), dh)
    xx = u * a
    if c.face == 'south':                                   # lining
        base = C(78, 60, 104) if kind == 'dark' else C(208, 201, 228)
        rgb = fill(c, base) * (0.93 + 0.08 * np.cos(u * 3.1))[..., None]
        return mix(rgb, base * 0.78, 0.5 * (1 - smoothstep(0.0, 0.45, d)))
    if kind == 'dark':
        rgb = dark_cloth(c, xx + seed * 1.7, y)
        rgb = mix(rgb, DARK_DEEP, 0.3 * np.abs(u) ** 2)
        rgb = mix(rgb, DARK_HL, 0.16 * np.exp(-(u / 0.45) ** 2))
        rgb = mix(rgb, DARK_DEEP, 0.45 * smoothstep(21.6, 22.7, y))
    else:
        rgb = mix(fill(c, WHITE), C(226, 220, 242), 0.5 * smoothstep(14.0, 22.0, y))
        fold = 0.5 - 0.5 * np.cos(u * 2.6 * math.pi + seed)
        rgb = mix(rgb, WHITE_SH, 0.3 * fold * smoothstep(21.0, 6.0, y) + 0.16 * np.abs(u) ** 2)
        rgb = mix(rgb, WHITE_SH, 0.12 * n2(xx * 2.0, y * 0.25, 9))
        rgb = mix(rgb, WHITE_SH, 0.5 * smoothstep(21.6, 22.7, y))
    g = gold_base(c)
    bord = band(d, -1.0, 0.075, aa) + band(d, 0.15, 0.19, aa) + band(d, 0.36, 0.4, aa)
    along = np.where(dh <= np.minimum(dl, dr), xx, y)
    loz = np.abs(d - 0.275) / 0.07 + np.abs((along / 0.36) % 1.0 - 0.5) * 0.36 / 0.12
    bord = np.maximum(bord, 1 - smoothstep(0.75, 1.0, loz))
    rgb = mix(rgb, g, np.clip(bord, 0, 1))
    if mp.get('crest'):                                     # embroidered emblem above the point
        cy = mp['tip'] + mp['hv'] + 1.15
        l2 = np.abs(xx) / 0.36 + np.abs(y - cy) / 0.6
        em = band(l2, 0.72, 1.0, 0.04) + (l2 < 0.4)
        for dy in (0.95, -0.95):
            em = em + disc(xx, y, 0, cy + dy, 0.09, 0.09, aa)
        rgb = mix(rgb, g, np.clip(em, 0, 1))
        rgb = mix(rgb, GEM, 1.0 * (l2 < 0.3))
    return rgb


def belt(c):
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    aa = c.px
    g = gold_base(c)
    rgb = mix(dark_cloth(c, x + z, y), g, 1 - band(y, 22.22, 22.83, aa))
    ang = np.arctan2(x, -(z - 0.1)) * 3.2
    loz = np.abs(y - 22.525) / 0.2 + np.abs((ang / 0.62) % 1.0 - 0.5) * 0.62 / 0.17
    rgb = mix(rgb, g, 1 - smoothstep(0.8, 1.0, loz))
    return mix(rgb, GOLD_DEEP, 0.55 * (band(y, 22.2, 22.24, aa) + band(y, 22.81, 22.85, aa)))


def choker(c):
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    aa = c.px
    rgb = mix(fill(c, DARK), gold_base(c), 1 - band(y, 32.36, 32.74, aa))
    ang = np.arctan2(x, -z) * 0.75
    dots = np.hypot(((ang / 0.2) % 1.0 - 0.5) * 0.2, y - 32.55)
    return mix(rgb, GOLD, 1 - smoothstep(0.035, 0.055, dots))


def _chains(c):
    x, y, z = c.pa[..., 0], c.pa[..., 1], c.pa[..., 2]
    aa = c.px
    th = np.degrees(np.arctan2(x, -(z - 0.25)))
    per = 36.0
    f = ((th + 18.0) / per) % 1.0
    y1, y2 = 22.02 - 0.72 * np.sin(math.pi * f), 22.02 - 1.42 * np.sin(math.pi * f)
    w = 0.036
    cov = np.maximum(1 - smoothstep(w, w + aa, np.abs(y - y1)), 1 - smoothstep(w, w + aa, np.abs(y - y2)))
    cov = cov * (tri(th / 2.2) > 0.2)
    arc = (f - 0.5) * per * 0.063
    stone = disc(arc, y, 0.0, 22.02 - 1.42 - 0.16, 0.1, 0.15, aa)
    cov = np.maximum(cov, stone)
    return cov, stone


def chains_alpha(c):
    return _chains(c)[0]


def chains(c):
    cov, stone = _chains(c)
    return mix(gold_base(c), mix(fill(c, GEM), GEM_HL, 0.3), stone * 0.9)
