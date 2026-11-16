"""Material swatches lifted from the original Thor texture atlas + procedural paint."""
import math
import numpy as np
from PIL import Image
from core import smoothstep

SEEDS = {
    'hair': (420, 100), 'hair2': (910, 100), 'hair3': (60, 560), 'hair4': (1160, 100),
    'cloth': (500, 350), 'cloth_trim': (250, 370), 'cloth2': (40, 120),
    'gold': (40, 1260), 'gold2': (190, 1060), 'skin': (334, 720), 'leather': (700, 1780),
}


def tile_rect(alpha, x, y):
    x0 = x1 = x
    y0 = y1 = y
    while x0 > 0 and alpha[y, x0 - 1] > 128: x0 -= 1
    while x1 < alpha.shape[1] - 1 and alpha[y, x1 + 1] > 128: x1 += 1
    while y0 > 0 and alpha[y0 - 1, x] > 128: y0 -= 1
    while y1 < alpha.shape[0] - 1 and alpha[y1 + 1, x] > 128: y1 += 1
    return x0, y0, x1 + 1, y1 + 1


class Swatches:
    def __init__(self, atlas_img):
        A = np.asarray(atlas_img.convert('RGBA'))
        self.sw = {}
        self.rects = {}
        for name, (x, y) in SEEDS.items():
            x0, y0, x1, y1 = tile_rect(A[..., 3], x, y)
            self.rects[name] = (x0, y0, x1, y1)
            m = 4
            self.sw[name] = A[y0 + m:y1 - m, x0 + m:x1 - m, :3].astype(np.float32)

    def mean(self, name):
        return self.sw[name].reshape(-1, 3).mean(0)

    def sample(self, name, u, v, scale=16.0):
        """Mirror-tiled bilinear lookup; u,v in model units, `scale` swatch px per unit."""
        s = self.sw[name]
        h, w = s.shape[:2]

        def mirror(t, n):
            t = np.mod(t, 2 * (n - 1))
            return np.where(t > n - 1, 2 * (n - 1) - t, t)

        x = mirror(np.asarray(u, np.float32) * scale, w)
        y = mirror(np.asarray(v, np.float32) * scale, h)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        return (s[y0, x0] * (1 - fx) * (1 - fy) + s[y0, x1] * fx * (1 - fy) + s[y1, x0] * (1 - fx) * fy + s[y1, x1] * fx * fy)


# ---------------------------------------------------------------- procedural paint
def vnoise(p, freq, seed=0):
    """Trilinear value noise in [0,1] for an (...,3) array of points."""
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
    c = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                c = c + w * h(x0 + dx, y0 + dy, z0 + dz)
    return c


def lerp(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1 - t) + np.asarray(b, np.float32) * t


def rounding(c):
    """0 at the middle of a face rising to 1+ at the silhouette of the cube (or its parent blob)."""
    blob = c.mp.get('blob')
    q = c.pl * 2.0
    if blob is not None:
        q = q * (c.size / np.maximum(np.asarray(blob, float), 1e-6))
    ax = {'east': 0, 'west': 0, 'up': 1, 'down': 1, 'north': 2, 'south': 2}[c.face]
    o = [k for k in range(3) if k != ax]
    return smoothstep(0.30, 1.22, np.sqrt(q[..., o[0]] ** 2 + q[..., o[1]] ** 2))


def poly_dist(x, y, pts):
    """Distance from (x, y) to a polyline."""
    d = np.full(x.shape, 1e9, np.float32)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        vx, vy = bx - ax, by - ay
        t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.minimum(d, np.hypot(x - (ax + t * vx), y - (ay + t * vy)))
    return d


# gold markings kept from the source model: chest emblem and the lines running down from the eyes
CHEST_MARKS = [
    [(0.55, 40.5), (1.15, 39.75), (2.2, 39.3), (2.95, 38.3), (2.85, 37.1), (2.2, 36.3)],
    [(0.45, 40.2), (0.72, 38.9), (1.25, 37.7), (1.2, 36.6)],
    [(1.95, 39.25), (3.3, 39.35), (4.1, 38.8)],
]
FACE_MARKS = [(1, [(1.36, 45.1), (1.42, 44.2), (1.62, 43.45)]), (1, [(0.92, 45.08), (0.94, 44.35)]),
              (-1, [(-1.36, 45.1), (-1.42, 44.2), (-1.62, 43.45)])]
MARK_GOLD = (204, 150, 52)


def edge_dist(c):
    return np.minimum(np.minimum(c.U, c.wu - c.U), np.minimum(c.Vv, c.hu - c.Vv))


def arm_paint(pa, side):
    """Anime-style muscle separation for the arms, in cylindrical coordinates around the arm axis.

    Returns (soft shadow, line) masks. theta: 0 = outer side, -90 = front, +90 = back."""
    lx = (pa[..., 0] - side * 8.05) * side
    lz = pa[..., 2] - 0.25
    y = pa[..., 1]
    th = np.degrees(np.arctan2(lz, lx))
    arc = lambda t0: np.abs((th - t0 + 180) % 360 - 180) * (1.8 * np.pi / 180)     # distance along the surface
    ln = np.zeros(y.shape, np.float32)
    sh = np.zeros(y.shape, np.float32)

    def stroke(d, w, k, lo=None, hi=None, fade=0.5):
        m = (1 - smoothstep(w * 0.5, w * 1.6, d)) * k
        if lo is not None:
            m = m * smoothstep(lo, lo + fade, y)
        if hi is not None:
            m = m * (1 - smoothstep(hi - fade, hi, y))
        return m

    # lower border of the deltoid: dips to a point on the outer side of the arm
    yd = 36.25 + 1.9 * np.clip(np.abs(th) / 95.0, 0, 1) ** 1.3
    vis = (np.abs(th) < 118).astype(np.float32)
    ln += stroke(np.abs(y - yd), 0.07, 0.75) * vis
    sh += 0.26 * (1 - smoothstep(0.0, 0.8, yd - y)) * (y < yd) * vis * smoothstep(34.8, 35.4, y)
    # biceps / triceps groove on the outer side, biceps under-curve and inner groove
    ln += stroke(arc(-24), 0.07, 0.6, 32.3, 36.2)
    ln += stroke(arc(52), 0.07, 0.45, 32.6, 35.6)
    ln += stroke(arc(-150), 0.07, 0.5, 32.4, 36.6)
    yb = 32.55 + 0.004 * (th + 90) ** 2 / 10
    ln += stroke(np.abs(y - yb), 0.06, 0.6) * (np.abs(th + 90) < 46)
    sh += 0.22 * np.exp(-(arc(-90) / 0.9) ** 2) * (1 - smoothstep(32.5, 33.6, y)) * smoothstep(31.9, 32.5, y)
    # triceps horseshoe on the back
    ln += stroke(np.abs(arc(92) - (y - 33.0) * 0.32), 0.06, 0.5, 33.0, 36.0)
    # elbow crease and the two long forearm muscle edges
    ln += stroke(np.abs(y - 31.35), 0.06, 0.55) * (np.abs(th + 95) < 55)
    ln += stroke(arc(-40 - (31.0 - y) * 7.5), 0.07, 0.55, 26.2, 31.2, 0.9)
    ln += stroke(arc(48 + (31.0 - y) * 5.0), 0.07, 0.45, 26.6, 31.0, 0.9)
    ln += stroke(arc(-165 + (31.0 - y) * 4.0), 0.06, 0.4, 27.0, 30.8, 0.9)
    # wrist tendons
    ln += stroke(arc(-82), 0.05, 0.4, 24.4, 26.4, 0.6) + stroke(arc(-104), 0.05, 0.4, 24.4, 26.2, 0.6)
    return sh, np.clip(ln, 0, 0.8) * 0.62


SCLERA = (26, 20, 26)
SKIN = np.array([235, 201, 184], np.float32)
SKIN_SH = np.array([186, 130, 118], np.float32)
SKIN_LINE = np.array([128, 74, 70], np.float32)
SKIN_HL = np.array([250, 228, 214], np.float32)
HAIR_TINT = np.array([1.06, 0.60, 0.66], np.float32)
HAIR_HL = np.array([238, 104, 82], np.float32)
HAIR_DK = np.array([82, 12, 20], np.float32)
CLOTH_SH = np.array([168, 182, 206], np.float32)
GOLD_TRIM = np.array([214, 160, 58], np.float32)
GOLD_DK = np.array([138, 88, 30], np.float32)
ARMOR = np.array([201, 163, 100], np.float32)
ARMOR_DK = np.array([112, 78, 44], np.float32)


class Mats:
    def __init__(self, sw):
        self.S = sw

    def table(self):
        names = ['skin', 'hair', 'cloth', 'leather', 'gold', 'armor', 'eye', 'iris', 'lash', 'lip', 'brow', 'dark', 'nail']
        return {n: getattr(self, n) for n in names}

    # ---- skin
    def skin(self, c):
        mp = c.mp
        pa = c.pa
        n = vnoise(pa, 0.8, 3) - 0.5
        n2 = vnoise(pa, 3.3, 7) - 0.5
        rgb = np.broadcast_to(np.array(mp.get('base', SKIN), np.float32), pa.shape).copy()
        rgb *= (1 + 0.035 * n + 0.018 * n2)[..., None]
        sh = np.zeros(pa.shape[:2], np.float32)
        line = np.zeros(pa.shape[:2], np.float32)
        sh += 0.26 * max(0.0, -float(c.na[1]))
        sh += mp.get('round', 0.0) * rounding(c)
        sh += mp.get('shade', 0.0)
        reg = mp.get('region')
        x, y = pa[..., 0], pa[..., 1]
        if reg == 'torso' and c.na[2] < -0.5:
            d = np.sqrt((x / 0.17) ** 2 + ((y - 30.85) / 0.30) ** 2)
            line += 0.85 * (1 - smoothstep(0.55, 1.25, d))
            for sx in (-1, 1):
                d = np.sqrt((x - sx * 3.2) ** 2 + (y - 36.35) ** 2)
                m = 1 - smoothstep(0.16, 0.30, d)
                rgb = lerp(rgb, (190, 126, 118), 0.75 * m)
            if c.cube.bone in ('body2', 'body8'):
                for pts in CHEST_MARKS:
                    m = 1 - smoothstep(0.075, 0.13, poly_dist(np.abs(x), y, pts))
                    rgb = lerp(rgb, MARK_GOLD, 0.92 * m)
        if reg == 'arm':
            sh_a, ln_a = arm_paint(pa, mp.get('side', 1))
            sh += sh_a
            line += ln_a
        if reg == 'face' and c.na[2] < -0.5:
            for _, pts in FACE_MARKS:
                m = 1 - smoothstep(0.045, 0.085, poly_dist(x, y, pts))
                rgb = lerp(rgb, MARK_GOLD, 0.9 * m)
            # soft shadow under the brow, beside the nose and under the lower lip
            sh += 0.22 * smoothstep(45.75, 46.15, y) * (1 - smoothstep(46.15, 46.5, y))
            sh += 0.20 * (1 - smoothstep(0.25, 0.75, np.abs(x))) * smoothstep(44.3, 44.8, y) * (1 - smoothstep(45.6, 46.1, y))
            sh += 0.16 * (1 - smoothstep(0.0, 0.5, np.abs(y - 43.35))) * (1 - smoothstep(0.5, 1.2, np.abs(x)))
        rgb = lerp(rgb, SKIN_SH, np.clip(sh, 0, 0.9))
        rgb = lerp(rgb, SKIN_LINE, np.clip(line, 0, 1))
        rgb = lerp(rgb, SKIN_HL, 0.14 * max(0.0, float(c.na[1])) * (1 - np.clip(sh * 2, 0, 1)))
        return rgb * mp.get('tone', 1.0)

    def nail(self, c):
        rgb = np.broadcast_to(np.array([240, 214, 204], np.float32), c.pa.shape).copy()
        return lerp(rgb, SKIN_SH, 0.5 * rounding(c))

    # ---- hair
    def hair(self, c):
        mp, size, pl = c.mp, c.size, c.pl
        seed = mp.get('seed', 0)
        u = pl[..., 0] * size[0] + mp.get('u0', 0.0)
        along = 0.5 - pl[..., 1]
        s = mp.get('sa', 0.0) + along * size[1]
        va, vb = mp.get('va', 0.0), mp.get('vb', 1.0)
        v = va + along * (vb - va)
        a = self.S.sample('hair', u * 1.15 + seed * 2.37, s * 0.42 + seed * 5.1, 16)
        b = self.S.sample('hair2', u * 0.9 + seed * 1.13, s * 0.36 + seed * 3.3, 14)
        q = np.stack([u * 5.5 + seed * 13.7, s * 0.10, np.full_like(u, seed * 0.37)], -1)
        st = vnoise(q, 1.0, 5)
        st2 = vnoise(q * [2.6, 1.6, 1], 1.0, 9)
        rgb = lerp(a, b, 0.22 + 0.25 * st) * HAIR_TINT
        rgb *= (0.78 + 0.30 * st + 0.16 * st2)[..., None]
        fine = vnoise(q * [3.4, 0.7, 1], 1.0, 17)                      # thin dark partings between strands
        rgb = lerp(rgb, HAIR_DK, 0.45 * (1 - smoothstep(0.16, 0.30, fine)))
        g = (0.80 + 0.20 * smoothstep(0.0, 0.10, v)) * (1 - 0.20 * smoothstep(0.78, 1.0, v))
        rgb *= g[..., None]
        k = mp.get('shinek', 0.0)
        if k:
            hb = np.exp(-((v - mp.get('shine', 0.3)) / mp.get('shinew', 0.08)) ** 2) * k * (0.35 + 1.0 * st)
            rgb = lerp(rgb, HAIR_HL, np.clip(hb, 0, 0.75))
        if c.face in ('north', 'south'):
            ex = smoothstep(0.30, 0.5, np.abs(pl[..., 0]))
            rgb = lerp(rgb, HAIR_DK, 0.42 * ex)
            if c.face == 'south':
                rgb = lerp(rgb, HAIR_DK, 0.35)
        else:
            rgb = lerp(rgb, HAIR_DK, 0.40)
        return rgb * mp.get('tone', 1.0)

    def brow(self, c):
        rgb = np.broadcast_to(np.array([118, 26, 28], np.float32), c.pa.shape).copy()
        return lerp(rgb, (70, 12, 16), 0.6 * rounding(c))

    # ---- cloth
    def cloth(self, c):
        mp, size, pl = c.mp, c.size, c.pl
        u = pl[..., 0] * size[0] + mp.get('u0', 0.0)
        along = 0.5 - pl[..., 1]
        s = mp.get('sa', 0.0) + along * size[1]
        if mp.get('world'):
            u, s = c.pa[..., 0] + c.pa[..., 2] * 0.6, -c.pa[..., 1]
        rgb = self.S.sample('cloth', u * mp.get('fu', 1.0) + mp.get('seed', 0) * 1.7, s * 0.55, 15) * 1.07
        q = np.stack([u * 2.2 + mp.get('seed', 0) * 3.1, s * 0.16, np.zeros_like(u)], -1)
        f = vnoise(q, 1.0, 21)
        rgb = lerp(rgb, CLOTH_SH, 0.38 * smoothstep(0.52, 0.9, f) * mp.get('folds', 1.0))
        sh = mp.get('round', 0.0) * rounding(c) + mp.get('shade', 0.0)
        broad = c.face in ('north', 'south')
        if broad and mp.get('edge', 0.25):
            sh = sh + mp.get('edge', 0.25) * smoothstep(0.36, 0.5, np.abs(pl[..., 0]))
        if mp.get('strip'):
            sh = sh + (0.0 if c.face == 'north' else 0.28)
        rgb = lerp(rgb, CLOTH_SH * 0.86, np.clip(sh, 0, 0.9))
        hem = mp.get('hem')
        if hem and broad:
            # gold zig-zag band above the hem, one tooth per panel, plus straight piping lines
            d = (pl[..., 1] + 0.5) * size[1] + mp.get('hem0', 0.0)
            tri = np.abs(pl[..., 0]) * 2.0
            m = 1 - smoothstep(0.07, 0.15, np.abs(d - (hem + 0.95 * (1 - tri))))
            m = np.maximum(m, 1 - smoothstep(0.05, 0.11, np.abs(d - hem * 0.42)))
            m = np.maximum(m, 1 - smoothstep(0.05, 0.11, np.abs(d - (hem + 1.35))))
            rgb = lerp(rgb, GOLD_TRIM * (0.86 + 0.25 * f)[..., None], m)
        tr = mp.get('trimx')
        if tr and broad:
            # gold piping + triangles along one vertical edge (the robe's front opening)
            dx = (0.5 - pl[..., 0] * tr) * size[0]
            ph = np.abs(np.mod(s, 1.15) / 1.15 - 0.5) * 2.0
            m = 1 - smoothstep(0.06, 0.13, np.abs(dx - (0.42 + 0.55 * ph)))
            m = np.maximum(m, 1 - smoothstep(0.05, 0.11, np.abs(dx - 0.16)))
            m = np.maximum(m, 1 - smoothstep(0.04, 0.10, np.abs(dx - 1.18)))
            rgb = lerp(rgb, GOLD_TRIM * (0.86 + 0.25 * f)[..., None], m)
        return rgb * mp.get('tone', 1.0)

    # ---- leather / gold / armor
    def leather(self, c):
        mp = c.mp
        u = c.pl[..., 0] * c.size[0] + mp.get('u0', 0.0)
        y = c.pl[..., 1] * c.size[1]
        rgb = self.S.sample('leather', u, y, 14) * np.array([1.42, 1.38, 1.36], np.float32)
        n = vnoise(np.stack([u * 0.8, y * 6.0, np.zeros_like(u)], -1), 1.0, 31)
        rgb *= (0.86 + 0.28 * n)[..., None]
        bands = mp.get('bands', 0)
        if bands and c.face in ('north', 'south'):
            t = (c.pl[..., 1] + 0.5) * bands + 0.06 * np.sin(u * 0.9 + mp.get('seed', 0))
            fr = t - np.floor(t)
            rgb = lerp(rgb, (26, 10, 8), 0.75 * (1 - smoothstep(0.0, 0.10, np.minimum(fr, 1 - fr))))
            rgb = lerp(rgb, (128, 78, 52), 0.35 * smoothstep(0.55, 0.85, fr) * (1 - smoothstep(0.85, 0.97, fr)))
        rgb = lerp(rgb, (24, 10, 8), np.clip(mp.get('round', 0.0) * rounding(c) + mp.get('shade', 0.0), 0, 0.9))
        return rgb * mp.get('tone', 1.0)

    def dark(self, c):
        rgb = np.broadcast_to(np.array(c.mp.get('base', (58, 30, 26)), np.float32), c.pa.shape).copy()
        return lerp(rgb, (20, 8, 8), np.clip(c.mp.get('round', 0.3) * rounding(c), 0, 1)) * c.mp.get('tone', 1.0)

    def _metal(self, c, sw, tint, dk, hl):
        mp = c.mp
        pa = c.pa
        rgb = self.S.sample(sw, pa[..., 0] * 0.9 + pa[..., 2] * 0.7, pa[..., 1] * 0.9 + pa[..., 2] * 0.3, 10)
        rgb = lerp(rgb, tint, mp.get('mix', 0.45))
        n = vnoise(pa, 1.7, 41)
        rgb *= (0.90 + 0.20 * n)[..., None]
        sh = mp.get('round', 0.0) * rounding(c) + mp.get('shade', 0.0) + 0.22 * max(0.0, -float(c.na[1]))
        rgb = lerp(rgb, dk, np.clip(sh, 0, 0.92))
        if mp.get('rim'):
            d = edge_dist(c)
            w = mp.get('rimw', 0.16)
            rgb = lerp(rgb, hl, 0.75 * (1 - smoothstep(w * 0.55, w, d)))
            rgb = lerp(rgb, dk, 0.70 * (1 - smoothstep(0.03, 0.09, np.abs(d - w * 1.25))))
        if mp.get('groove'):
            g = mp['groove']            # engraved horizontal lines every g units
            t = pa[..., 1] / g
            fr = np.abs(t - np.round(t))
            rgb = lerp(rgb, dk, 0.55 * (1 - smoothstep(0.03, 0.09, fr * g)))
        rgb = lerp(rgb, hl, 0.28 * max(0.0, float(c.na[1])))
        return rgb * mp.get('tone', 1.0)

    def gold(self, c):
        return self._metal(c, 'gold2', np.array([226, 176, 70], np.float32), GOLD_DK, np.array([255, 226, 140], np.float32))

    def armor(self, c):
        return self._metal(c, 'gold', ARMOR, ARMOR_DK, np.array([236, 206, 150], np.float32))

    # ---- face details
    def eye(self, c):
        sd = c.mp.get('side', 1)
        x = c.pl[..., 0] * sd            # -0.5 inner corner .. +0.5 outer corner
        y = c.pl[..., 1]
        rgb = np.broadcast_to(np.array(SCLERA, np.float32), c.pa.shape).copy()      # black sclera, as in the source
        rgb = lerp(rgb, (52, 36, 44), 0.6 * smoothstep(0.0, -0.5, y))               # faint light on the lower half
        rgb = lerp(rgb, (10, 8, 10), 0.7 * smoothstep(0.34, 0.5, np.abs(x)))
        return rgb

    def iris(self, c):
        x, y = c.pl[..., 0] * c.size[0], c.pl[..., 1] * c.size[1]
        r = np.sqrt(x * x + y * y)
        R = c.mp.get('r', 0.19)
        rgb = np.broadcast_to(np.array(SCLERA, np.float32), c.pa.shape).copy()
        rgb = lerp(rgb, (52, 36, 44), 0.6 * smoothstep(0.0, -0.5, c.pl[..., 1]))
        gold = lerp(np.array([255, 205, 70], np.float32), np.array([214, 132, 24], np.float32), smoothstep(-R, R, y)[..., None])
        rgb = lerp(rgb, gold, 1 - smoothstep(R * 0.86, R * 1.02, r))
        rgb = lerp(rgb, (84, 44, 12), (1 - smoothstep(0.02, 0.05, np.abs(r - R * 0.92))) * 0.9)
        # slit pupil with a small round core
        pup = (1 - smoothstep(0.022, 0.05, np.abs(x))) * (1 - smoothstep(R * 0.55, R * 0.8, np.abs(y)))
        pup = np.maximum(pup, 1 - smoothstep(0.045, 0.075, r))
        rgb = lerp(rgb, (22, 10, 8), pup)
        hlm = 1 - smoothstep(0.02, 0.05, np.sqrt((x - 0.07 * c.mp.get('side', 1)) ** 2 + (y - 0.07) ** 2))
        rgb = lerp(rgb, (255, 255, 250), 0.9 * hlm)
        return rgb

    def lash(self, c):
        return np.broadcast_to(np.array(c.mp.get('base', (46, 18, 22)), np.float32), c.pa.shape).copy()

    def lip(self, c):
        rgb = np.broadcast_to(np.array(c.mp.get('base', (206, 142, 134)), np.float32), c.pa.shape).copy()
        return lerp(rgb, (150, 86, 86), np.clip(0.7 * rounding(c) + c.mp.get('shade', 0.0), 0, 1))


if __name__ == '__main__':
    import sys, json, base64, io
    d = json.load(open(sys.argv[1]))
    im = Image.open(io.BytesIO(base64.b64decode(d['textures'][0]['source'].split(',', 1)[1])))
    S = Swatches(im)
    tiles = []
    for k, r in S.rects.items():
        print(k, r, S.sw[k].shape[:2], np.round(S.mean(k)), 'p5', np.round(np.percentile(S.sw[k].reshape(-1, 3), 5, axis=0)), 'p95', np.round(np.percentile(S.sw[k].reshape(-1, 3), 95, axis=0)))
        t = Image.fromarray(S.sw[k].astype(np.uint8))
        t.thumbnail((200, 260))
        tiles.append(t)
    out = Image.new('RGB', (sum(t.width + 6 for t in tiles), 270), (30, 30, 40))
    x = 0
    for t in tiles:
        out.paste(t, (x, 4)); x += t.width + 6
    out.save(sys.argv[2] if len(sys.argv) > 2 else 'swatches.png')
