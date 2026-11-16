"""Low-poly modelling kit for cube-based (Bedrock / GeckoLib) entity models.

Geometry is authored in Blockbench space in the rest pose: the character faces -Z and her right
hand is at +X. Besides ordinary boxes, any flat convex polygon becomes a thin cube whose texture is
cut to the polygon with binary alpha, so limbs can taper smoothly instead of stair-stepping.
"""
import math, json, copy, re
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
FACES = ('north', 'east', 'south', 'west', 'up', 'down')
FACE_CORNERS = {      # TL, TR, BR, BL of each face, as indexes into (lo, hi)
    'north': [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
    'south': [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    'east': [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    'west': [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    'up': [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    'down': [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
}
FACE_NORMAL = {'north': (0, 0, -1), 'south': (0, 0, 1), 'east': (1, 0, 0), 'west': (-1, 0, 0), 'up': (0, 1, 0), 'down': (0, -1, 0)}
MIRROR_FACE = {'east': 'west', 'west': 'east'}
UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------- math
def norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot_matrix(rot):
    """Blockbench euler angles in degrees -> matrix (R = Rz Ry Rx)."""
    if rot is None or not any(rot):
        return np.eye(3)
    x, y, z = [math.radians(a) for a in rot]
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_euler(R):
    sy = max(-1.0, min(1.0, -R[2, 0]))
    y = math.asin(sy)
    if abs(sy) < 0.999999:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: fold everything into x
        z = 0.0
        x = math.atan2(-R[1, 2], R[1, 1])
    return [0.0 if abs(a) < 1e-9 else math.degrees(a) for a in (x, y, z)]


def pivot_xf(origin, rot):
    M = np.eye(4)
    R = rot_matrix(rot)
    o = np.asarray(origin, float)
    M[:3, :3] = R
    M[:3, 3] = o - R @ o
    return M


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def catmull(pts, n=8):
    """Catmull-Rom spline through pts with n samples per span."""
    P = [np.asarray(p, float) for p in pts]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return np.array(out)


def resample(poly, n):
    """n+1 points evenly spaced by arc length along a polyline."""
    poly = np.asarray(poly, float)
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    ts = np.linspace(0, s[-1], n + 1)
    return np.stack([np.interp(ts, s, poly[:, k]) for k in range(poly.shape[1])], axis=1)


def interp(x, xs, ys):
    """Smooth (Catmull-Rom) interpolation of a 1-D table at scalar x."""
    xs = list(xs)
    ys = np.asarray(ys, float)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    p1, p2 = ys[i], ys[i + 1]
    m1 = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1]) if i > 0 else (p2 - p1) / (xs[i + 1] - xs[i])
    m2 = (ys[i + 2] - ys[i]) / (xs[i + 2] - xs[i]) if i + 2 < len(xs) else (p2 - p1) / (xs[i + 1] - xs[i])
    h = xs[i + 1] - xs[i]
    return (2 * t ** 3 - 3 * t ** 2 + 1) * p1 + (t ** 3 - 2 * t ** 2 + t) * h * m1 + (-2 * t ** 3 + 3 * t ** 2) * p2 + (t ** 3 - t ** 2) * h * m2


def mirror_name(n):
    return re.sub(r'(_|s)r(?=_|$)', lambda m: m.group(1) + 'l', n)


def dir_n(phi):
    """Horizontal unit normal for azimuth phi (degrees): 0 = front (-Z), 90 = her right (+X)."""
    a = math.radians(phi)
    return np.array([math.sin(a), 0.0, -math.cos(a)])


def dir_ex(phi):
    """Direction that reads as 'right' when the facet with azimuth phi is seen from outside."""
    a = math.radians(phi)
    return np.array([-math.cos(a), 0.0, -math.sin(a)])


def support(dirs, rx, rzf, rzb=None, p=2.0):
    """Support distances of a (super)ellipse with separate front/back depth for each direction."""
    rzb = rzf if rzb is None else rzb
    q = p / (p - 1.0)
    out = []
    for d in dirs:
        a = math.radians(d)
        nx, nz = math.sin(a), -math.cos(a)
        rz = rzf if nz < 0 else rzb
        out.append(((rx * abs(nx)) ** q + (rz * abs(nz)) ** q) ** (1 / q))
    return np.array(out)


def ring_vertices(dirs, h, y, c=(0.0, 0.0)):
    """Polygon circumscribed around the support distances; vertex j sits between direction j and j+1."""
    K = len(dirs)
    out = []
    for j in range(K):
        a, b = math.radians(dirs[j]), math.radians(dirs[(j + 1) % K])
        A = np.array([[math.sin(a), -math.cos(a)], [math.sin(b), -math.cos(b)]])
        x, z = np.linalg.solve(A, [h[j], h[(j + 1) % K]])
        out.append((x + c[0], y, z + c[1]))
    return np.array(out)


def frame_xf(origin, axis, front=(0, 0, -1)):
    """Rigid transform taking the loft's +Y to `axis` and its front (-Z) as close to `front` as possible."""
    y = norm(axis)
    f = np.asarray(front, float)
    f = norm(f - y * (f @ y))
    z = -f
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1), np.asarray(origin, float)


# ---------------------------------------------------------------- skeleton
class Skeleton:
    """Bones of the source geometry, converted to Blockbench space. Never modified."""

    def __init__(self, geo):
        g = geo['minecraft:geometry'][0]
        self.desc = g['description']
        self.bones = {}
        for b in g['bones']:
            p, r = b.get('pivot', [0, 0, 0]), b.get('rotation', [0, 0, 0])
            self.bones[b['name']] = dict(parent=b.get('parent'), pivot=np.array([-p[0], p[1], p[2]], float), rot=[-r[0], -r[1], r[2]])
        self._w = {}

    def world(self, name):
        if name not in self._w:
            b = self.bones[name]
            M = self.world(b['parent']) if b['parent'] else np.eye(4)
            self._w[name] = M @ pivot_xf(b['pivot'], b['rot'])
        return self._w[name]

    def pivot(self, name):
        """Rest-pose world position of a bone's pivot."""
        return (self.world(name) @ np.append(self.bones[name]['pivot'], 1.0))[:3]


# ---------------------------------------------------------------- scene
class Cube:
    __slots__ = ('bone', 'O', 'R', 'lo', 'hi', 'mat', 'mp', 'dens', 'faces', 'poly', 'twin', 'F', 'pad')

    def __init__(self, bone, O, R, lo, hi, mat, mp, faces, poly, dens):
        self.bone, self.O, self.R = bone, np.asarray(O, float), np.asarray(R, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.mat, self.mp, self.faces, self.poly, self.dens = mat, mp, tuple(faces), poly, dens
        self.twin = None
        self.F = {}
        self.pad = 0.0


class Builder:
    def __init__(self, skel, pad=0.1):
        self.skel = skel
        self.cubes = []
        self.pad = pad
        self.max_dev = 0.0

    def _add(self, bone, O, R, lo, hi, mat, faces, poly=None, dens=None, **mp):
        assert bone in self.skel.bones, bone
        mp.setdefault('side', 1)
        cu = Cube(bone, O, R, lo, hi, mat, mp, faces, poly, dens)
        self.cubes.append(cu)
        return cu

    # -- solid boxes
    def box(self, bone, c, s, rot=(0, 0, 0), mat=None, faces=FACES, R=None, **mp):
        s = np.abs(np.asarray(s, float))
        return self._add(bone, c, rot_matrix(rot) if R is None else R, -s / 2, s / 2, mat, faces, **mp)

    def bar(self, bone, a, b, w, d, mat, side=(1, 0, 0), ext=0.0, faces=FACES, **mp):
        """Box running from point a to point b; `w` is its size along `side`, `d` across."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        y = norm(a - b)
        x = np.asarray(side, float)
        x = norm(x - y * (x @ y))
        R = np.stack([x, y, np.cross(x, y)], axis=1)
        L = np.linalg.norm(a - b) + 2 * ext
        return self._add(bone, (a + b) / 2, R, (-w / 2, -L / 2, -d / 2), (w / 2, L / 2, d / 2), mat, faces, **mp)

    # -- flat cut-out polygons
    def poly(self, bone, pts, mat, out, ex=None, two=False, thick=0.03, strip=None, **mp):
        """Convex planar polygon. For quads pass TL, TR, BR, BL as seen from outside; `out` is any
        vector on the outside. `strip=(s0, s1, S)` adds across/along coordinates for trapezoids."""
        Q = np.array(pts, float)
        P = [Q[0]]
        for p in Q[1:]:
            if np.linalg.norm(p - P[-1]) > 1e-4:
                P.append(p)
        if len(P) > 1 and np.linalg.norm(P[0] - P[-1]) < 1e-4:
            P.pop()
        if len(P) < 3:
            return None
        P = np.array(P)
        c = P.mean(0)
        n = np.zeros(3)
        for i in range(len(P)):
            n += np.cross(P[i] - c, P[(i + 1) % len(P)] - c)
        if np.linalg.norm(n) < 1e-9:
            return None
        n = norm(n)
        if n @ np.asarray(out, float) < 0:
            n = -n
        ex = Q[1] - Q[0] if ex is None else np.asarray(ex, float)
        ex = norm(ex - n * (ex @ n))
        R = np.stack([-ex, np.cross(n, ex), -n], axis=1)
        L = (P - c) @ R
        self.max_dev = max(self.max_dev, float(np.abs(L[:, 2]).max()))
        m = self.pad
        lo = (L[:, 0].min() - m, L[:, 1].min() - m, 0.0)
        hi = (L[:, 0].max() + m, L[:, 1].max() + m, thick)
        if strip is not None and len(Q) == 4:
            e = (Q - c) @ ex
            yv = (Q - c) @ R[:, 1]
            mp['trap'] = (0.5 * (yv[0] + yv[1]), 0.5 * (yv[2] + yv[3]), e[0], e[1], e[3], e[2]) + tuple(strip)
        cu = self._add(bone, c, R, lo, hi, mat, ('north', 'south') if two else ('north',), poly=L[:, :2].copy(), **mp)
        cu.pad = m
        return cu

    def loft(self, bone, dirs, rings, mat, skip=None, xf=None, two=False, strip=False, mpf=None,
             cap0=False, cap1=False, **mp):
        """Skin between successive rings; ring = (height, support distances, (cx, cz)). Every facet is
        a planar trapezoid because all rings share the same facet directions."""
        K = len(dirs)
        Rm, t = (np.eye(3), np.zeros(3)) if xf is None else xf
        T = lambda p: np.asarray(p, float) @ Rm.T + t
        Vs = [T(ring_vertices(dirs, h, y, c)) for (y, h, c) in rings]
        mids = np.array([[(V[k] + V[k - 1]) / 2 for k in range(K)] for V in Vs])
        cum = np.concatenate([np.zeros((1, K)), np.cumsum(np.linalg.norm(np.diff(mids, axis=0), axis=2), axis=0)])
        made = []
        for i in range(len(rings) - 1):
            a, b = (i, i + 1) if rings[i][0] >= rings[i + 1][0] else (i + 1, i)
            for k in range(K):
                if skip and skip(i, k):
                    continue
                pts = [Vs[a][k], Vs[a][k - 1], Vs[b][k - 1], Vs[b][k]]
                bn = bone(i, k) if callable(bone) else bone
                extra = dict(mpf(i, k)) if mpf else {}
                st = (cum[a][k], cum[b][k], cum[-1][k]) if strip else None
                cu = self.poly(bn, pts, mat, Rm @ dir_n(dirs[k]), Rm @ dir_ex(dirs[k]), two, strip=st, **dict(mp, **extra))
                if cu is not None:
                    made.append(cu)
        for flag, i in ((cap0, 0), (cap1, len(rings) - 1)):      # close the first / last ring
            if flag:
                up = rings[i][0] >= rings[len(rings) - 1 - i][0]
                bn = bone(min(i, len(rings) - 2), 0) if callable(bone) else bone
                cu = self.poly(bn, Vs[i], mat, Rm @ (UP if up else -UP), Rm @ np.array([-1.0, 0, 0]), False, **mp)
                if cu is not None:
                    made.append(cu)
        return made

    def ribbon(self, bone, C, wl, wr, ex, out, mat, two=True, s0=0.0, S=None, mpf=None, **mp):
        """Strip of trapezoids along centre line C (root first). wl/wr are the half widths to the left
        and right as seen from outside, `ex` the fixed direction of 'right'."""
        C = np.asarray(C, float)
        ex = norm(ex)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]) + s0
        S = s[-1] if S is None else S
        made = []
        for i in range(len(C) - 1):
            mid = (C[i] + C[i + 1]) / 2
            pts = [C[i] - ex * wl[i], C[i] + ex * wr[i], C[i + 1] + ex * wr[i + 1], C[i + 1] - ex * wl[i + 1]]
            bn = bone(i, mid) if callable(bone) else bone
            o = out(i, mid) if callable(out) else out
            extra = dict(mpf(i, mid)) if mpf else {}
            cu = self.poly(bn, pts, mat, o, ex, two, strip=(s[i], s[i + 1], S), **dict(mp, **extra))
            if cu is not None:
                made.append(cu)
        return made

    def mirror(self, cubes, share=True):
        """Mirror cubes to the other side; with share=True the copies reuse the originals' texels."""
        M = np.diag([-1.0, 1.0, 1.0])
        out = []
        for cu in cubes:
            lo, hi = cu.lo.copy(), cu.hi.copy()
            lo[0], hi[0] = -cu.hi[0], -cu.lo[0]
            mp = dict(cu.mp, side=-cu.mp.get('side', 1))
            if 'trap' in mp:
                yT, yB, lT, rT, lB, rB, s0, s1, S = mp['trap']
                mp['trap'] = (yT, yB, -rT, -lT, -rB, -lB, s0, s1, S)
            m = self._add(mirror_name(cu.bone), M @ cu.O, M @ cu.R @ M, lo, hi, cu.mat, cu.faces,
                          poly=None if cu.poly is None else cu.poly * [-1.0, 1.0], dens=cu.dens, **mp)
            m.twin = cu if share else None
            m.pad = cu.pad
            out.append(m)
        return out


# ---------------------------------------------------------------- faces
class Face:
    __slots__ = ('cube', 'name', 'cw', 'wu', 'hu', 'n', 'ao', 'aol', 'hidden', 'x', 'y', 'w', 'h', 'twin')


class Ctx:
    pass


def build_faces(B):
    faces = []
    for cu in B.cubes:
        b = (cu.lo, cu.hi)
        for fname in cu.faces:
            f = Face()
            f.cube, f.name = cu, fname
            loc = np.array([[b[ix][0], b[iy][1], b[iz][2]] for ix, iy, iz in FACE_CORNERS[fname]])
            f.cw = loc @ cu.R.T + cu.O
            f.wu = float(np.linalg.norm(loc[1] - loc[0]))
            f.hu = float(np.linalg.norm(loc[3] - loc[0]))
            f.n = cu.R @ np.array(FACE_NORMAL[fname], float)
            f.ao, f.hidden, f.twin = None, False, None
            f.x = f.y = f.w = f.h = 0
            cu.F[fname] = f
            faces.append(f)
    for cu in B.cubes:
        if cu.twin is not None:
            for fname, f in cu.F.items():
                f.twin = cu.twin.F[MIRROR_FACE.get(fname, fname)]
    return faces


def poly_sd(xy, poly):
    """Signed distance-like value to a convex polygon: negative inside, positive outside."""
    area = 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        area += a[0] * b[1] - b[0] * a[1]
    sgn = 1.0 if area > 0 else -1.0
    sd = np.full(xy.shape[:-1], -1e9)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        L = math.hypot(e[0], e[1])
        if L < 1e-9:
            continue
        sd = np.maximum(sd, sgn * ((xy[..., 0] - a[0]) * e[1] - (xy[..., 1] - a[1]) * e[0]) / L)
    return sd


def make_ctx(f, w, h, S=None, T=None):
    """Sampling context for a face at w x h texels (or at explicit face coordinates S, T)."""
    cu = f.cube
    c = Ctx()
    c.face, c.cube, c.mp, c.n = f.name, cu, cu.mp, f.n
    c.w, c.h = w, h
    if S is None:
        S, T = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    c.S, c.T = S, T
    c.pa = f.cw[0] + S[..., None] * (f.cw[1] - f.cw[0]) + T[..., None] * (f.cw[3] - f.cw[0])
    c.L = (c.pa - cu.O) @ cu.R
    c.size = cu.hi - cu.lo
    c.px = max(f.wu / w, f.hu / h)          # texel size in model units
    c.cache = {}
    tr = cu.mp.get('trap')
    if tr is not None:
        yT, yB, lT, rT, lB, rB, s0, s1, Stot = tr
        tt = np.clip((yT - c.L[..., 1]) / (yT - yB + 1e-9), 0, 1)
        l, r = lT + (lB - lT) * tt, rT + (rB - rT) * tt
        c.a = np.maximum((r - l) / 2, 1e-4)
        c.u = (-c.L[..., 0] - (l + r) / 2) / c.a
        c.t, c.s, c.S = tt, s0 + (s1 - s0) * tt, Stot
    return c


def alpha_of(c, margin):
    """Coverage of the cube's cut-out shape (polygon plus optional painted alpha)."""
    cu = c.cube
    a = np.ones(c.pa.shape[:2], np.float32)
    if cu.poly is not None:
        a = (poly_sd(c.L[..., :2], cu.poly) <= margin).astype(np.float32)
    afn = cu.mp.get('afn')
    if afn is not None:
        a = a * (afn(c) > 0.5)
    return a


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], axis=1)


def compute_ao(faces, ndirs=96, sdens=6.0, res=5.0, bias=0.22, log=print):
    """Shadow-map ambient occlusion on a coarse per-face grid; cut-out texels neither cast nor receive."""
    pts, nrm, ok, shape, ofs = [], [], [], [], [0]
    for f in faces:
        ps, pt = f.cube.pad / f.wu, f.cube.pad / f.hu          # the grid spans the polygon, not its padding
        n = max(1, int(math.ceil(f.wu * (1 - 2 * ps) * sdens)))
        m = max(1, int(math.ceil(f.hu * (1 - 2 * pt) * sdens)))
        f.aol = (ps, pt, n, m)
        S, T = np.meshgrid(ps + (1 - 2 * ps) * np.arange(n + 1) / n, pt + (1 - 2 * pt) * np.arange(m + 1) / m)
        c = make_ctx(f, n + 1, m + 1, S, T)
        pts.append(c.pa.reshape(-1, 3))
        ok.append(alpha_of(c, 0.02).reshape(-1) > 0.5)
        nrm.append(np.repeat(f.n[None], (n + 1) * (m + 1), 0))
        shape.append((m + 1, n + 1))
        ofs.append(ofs[-1] + (n + 1) * (m + 1))
    ok = np.concatenate(ok)
    P = np.concatenate(pts).astype(np.float32)[ok]
    N = np.concatenate(nrm).astype(np.float32)[ok]
    log(f'  ao samples {len(P)}')
    acc = np.zeros(len(P), np.float32)
    wsum = np.zeros(len(P), np.float32)
    seen = np.zeros(len(P), bool)
    for d in fib_sphere(ndirs):
        a = np.array([0, 1, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
        r = norm(np.cross(a, d))
        u = np.cross(d, r)
        pr, pu, dep = P @ r.astype(np.float32), P @ u.astype(np.float32), P @ d.astype(np.float32)
        ix = ((pr - pr.min()) * res).astype(np.int64)
        iy = ((pu - pu.min()) * res).astype(np.int64)
        W = ix.max() + 1
        idx = iy * W + ix
        dm = np.full((iy.max() + 1) * W, -1e9, np.float32)
        np.maximum.at(dm, idx, dep)
        cw = N @ d.astype(np.float32)
        cc = np.clip(cw, 0.12, 1.0)   # slope-scaled bias
        vis = dep >= dm[idx] - (bias + (1.3 / res) * np.sqrt(1 - cc * cc) / cc)
        wgt = np.clip(cw, 0, None) * (0.62 + 0.38 * d[1])
        acc += vis * wgt
        wsum += wgt
        seen |= vis & (cw > 0.08)
    ao = np.ones(len(ok), np.float32)
    ao[ok] = acc / np.maximum(wsum, 1e-6)
    sn = np.zeros(len(ok), bool)
    sn[ok] = seen
    for i, f in enumerate(faces):
        a, b = ofs[i], ofs[i + 1]
        f.ao = fill_masked(ao[a:b].reshape(shape[i]), ok[a:b].reshape(shape[i]))
        f.hidden = not sn[a:b].any()
    for f in faces:               # mirrored pairs share texels: blend their occlusion
        if f.twin is not None:
            t = f.twin
            m = 0.5 * (t.ao + f.ao[:, ::-1])
            t.ao, f.ao = m, m[:, ::-1]
            t.hidden = f.hidden = t.hidden and f.hidden


def fill_masked(a, m):
    """Grow valid cells into masked ones so later interpolation never mixes in placeholder values."""
    if m.all() or not m.any():
        return a
    a, m = a.copy(), m.copy()
    for _ in range(max(a.shape)):
        if m.all():
            break
        p, pm = np.pad(a * m, 1), np.pad(m.astype(np.float32), 1)
        s = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        n = pm[:-2, 1:-1] + pm[2:, 1:-1] + pm[1:-1, :-2] + pm[1:-1, 2:]
        new = (~m) & (n > 0)
        a[new] = s[new] / n[new]
        m |= new
    return a


def ao_lookup(f, c):
    """Bilinear read of the face's occlusion grid at the texels of context c."""
    ps, pt, n, m = f.aol
    x = np.clip((c.S - ps) / (1 - 2 * ps) * n, 0, n)
    y = np.clip((c.T - pt) / (1 - 2 * pt) * m, 0, m)
    x0, y0 = np.minimum(np.floor(x).astype(int), n - 1), np.minimum(np.floor(y).astype(int), m - 1)
    fx, fy = x - x0, y - y0
    a = f.ao
    return (a[y0, x0] * (1 - fy) * (1 - fx) + a[y0, x0 + 1] * (1 - fy) * fx
            + a[y0 + 1, x0] * fy * (1 - fx) + a[y0 + 1, x0 + 1] * fy * fx)


# ---------------------------------------------------------------- atlas
def size_faces(faces, density, scale=1.0, hidden=6.0):
    """Texel size of each face, kept even so a half-resolution atlas stays pixel aligned."""
    for f in faces:
        base = density.get(f.cube.mat, 24) if isinstance(density, dict) and not callable(f.cube.mat) else 24
        d = hidden if f.hidden else (f.cube.dens or base) * scale
        if f.name == 'south' and f.cube.poly is not None:
            d *= f.cube.mp.get('back', 0.55)
        f.w = max(2, 2 * int(round(f.wu * d / 2)))
        f.h = max(2, 2 * int(round(f.hu * d / 2)))


def pack(faces, W, H, pad=2):
    """Shelf packing; returns the used height or None when the atlas is full."""
    x = y = pad
    rowh = 0
    for f in sorted(faces, key=lambda f: (-f.h, -f.w)):
        if x + f.w + pad > W:
            x = pad
            y += rowh + pad
            rowh = 0
        if y + f.h + pad > H:
            return None
        f.x, f.y = x, y
        x += f.w + pad
        rowh = max(rowh, f.h)
    return y + rowh + pad


def bake(faces, W, H, materials, ao_strength=0.5, log=print):
    atlas = np.zeros((H, W, 4), np.uint8)
    for f in faces:
        if f.twin is not None:
            continue
        cu = f.cube
        c = make_ctx(f, f.w, f.h)
        alpha = alpha_of(c, 0.55 * c.px)      # just enough overlap that neighbouring facets leave no pinholes
        mat = cu.mat if callable(cu.mat) else materials[cu.mat]
        rgb = np.asarray(mat(c), np.float32)
        if f.ao is not None:
            k = cu.mp.get('ao', 1.0) * (cu.mp.get('aob', 1.0) if f.name == 'south' else 1.0) * ao_strength
            sh = 1 - k * (1 - np.clip(ao_lookup(f, c), 0, 1) ** 0.85)
            tint = np.array(cu.mp.get('aotint', (0.95, 0.90, 0.97)), np.float32)   # shadows lean cool
            rgb = rgb * (sh[..., None] * (1 - (1 - sh[..., None]) * (1 - tint)))
        px = np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8)
        px = np.pad(px, ((1, 1), (1, 1), (0, 0)), mode='edge')
        atlas[f.y - 1:f.y + f.h + 1, f.x - 1:f.x + f.w + 1] = px
    return atlas


def prune(B, faces, atlas):
    """Remove faces whose texels are entirely transparent, and cubes left with no face at all."""
    gone = {id(f) for f in faces if atlas[f.y:f.y + f.h, f.x:f.x + f.w, 3].max() == 0}
    n = 0
    for cu in B.cubes:
        for name in list(cu.F):
            f = cu.F[name]
            if id(f.twin if f.twin is not None else f) in gone:
                del cu.F[name]
                n += 1
    B.cubes = [cu for cu in B.cubes if cu.F]
    return n


def halve(atlas):
    """2:1 reduction that keeps alpha binary and avoids dark fringes."""
    a = atlas.astype(np.float32)
    H, W = a.shape[:2]
    q = a.reshape(H // 2, 2, W // 2, 2, 4)
    al = q[..., 3:4] / 255.0
    wsum = al.sum(axis=(1, 3))
    rgb = np.where(wsum > 0, (q[..., :3] * al).sum(axis=(1, 3)) / np.maximum(wsum, 1e-6), q[..., :3].mean(axis=(1, 3)))
    out = np.dstack([rgb, (wsum[..., 0] >= 2.0) * 255.0])
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- export
def _r(v, n=4):
    v = round(float(v), n)
    return 0.0 if v == 0 else v


def export_geo(B, src_geo, uv_scale):
    """Swap the cubes of the source geometry for the builder's, leaving every bone untouched."""
    out = copy.deepcopy(src_geo)
    by_bone = {}
    for cu in B.cubes:
        by_bone.setdefault(cu.bone, []).append(cu)
    for b in out['minecraft:geometry'][0]['bones']:
        b.pop('cubes', None)
        cubes = []
        Wb = B.skel.world(b['name'])
        Rb, tb = Wb[:3, :3], Wb[:3, 3]
        for cu in by_bone.get(b['name'], []):
            O = Rb.T @ (cu.O - tb)             # rest-pose world -> the bone's own space
            rot = mat_to_euler(Rb.T @ cu.R)
            frm, to = O + cu.lo, O + cu.hi
            e = {'origin': [_r(-to[0]), _r(frm[1]), _r(frm[2])], 'size': [_r(v) for v in (to - frm)]}
            if any(abs(a) > 1e-6 for a in rot):
                e['pivot'] = [_r(-O[0]), _r(O[1]), _r(O[2])]
                e['rotation'] = [_r(-rot[0]), _r(-rot[1]), _r(rot[2])]
            uv = {}
            for fname in FACES:
                f = cu.F.get(fname)
                if f is None:
                    continue
                flip = f.twin is not None
                g = f.twin if flip else f
                u1, v1, u2, v2 = g.x * uv_scale, g.y * uv_scale, (g.x + g.w) * uv_scale, (g.y + g.h) * uv_scale
                if flip:
                    u1, u2 = u2, u1
                if fname in ('up', 'down'):    # Bedrock stores these two faces turned half round
                    u1, v1, u2, v2 = u2, v2, u1, v1
                uv[fname] = {'uv': [_r(u1), _r(v1)], 'uv_size': [_r(u2 - u1), _r(v2 - v1)]}
            e['uv'] = uv
            cubes.append(e)
        if cubes:
            b['cubes'] = cubes
    return out


def dump_geo(geo):
    """JSON with one cube per line, so diffs stay readable."""
    g = geo['minecraft:geometry'][0]
    js = lambda v: json.dumps(v, separators=(', ', ': '))
    L = ['{', f'\t"format_version": {js(geo["format_version"])},', '\t"minecraft:geometry": [', '\t\t{',
         f'\t\t\t"description": {js(g["description"])},', '\t\t\t"bones": [']
    for bi, b in enumerate(g['bones']):
        L.append('\t\t\t\t{')
        items = [(k, v) for k, v in b.items() if k != 'cubes']
        for i, (k, v) in enumerate(items):
            comma = ',' if (i < len(items) - 1 or 'cubes' in b) else ''
            L.append(f'\t\t\t\t\t{js(k)}: {js(v)}{comma}')
        if 'cubes' in b:
            L.append('\t\t\t\t\t"cubes": [')
            L += [f'\t\t\t\t\t\t{js(c)}{"," if i < len(b["cubes"]) - 1 else ""}' for i, c in enumerate(b['cubes'])]
            L.append('\t\t\t\t\t]')
        L.append('\t\t\t\t}' + (',' if bi < len(g['bones']) - 1 else ''))
    L += ['\t\t\t]', '\t\t}', '\t]', '}']
    return '\n'.join(L) + '\n'
