"""Core helpers for the Thor boss model generator.

Bones and cubes are authored in Blockbench "rest" coordinates (the same space the
.bbmodel stores), baked into a texture atlas and written back into the original
project file so every legacy bone name survives.
"""
import math, json, io, base64, uuid, copy
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
V = lambda *a: np.array(a, dtype=float)


# ---------------------------------------------------------------- math
def rot_matrix(rot):
    """Blockbench euler (degrees, ZYX order) -> 3x3 matrix."""
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
    out = [math.degrees(a) for a in (x, y, z)]
    return [0.0 if abs(a) < 1e-9 else a for a in out]


def pivot_xf(origin, rot):
    M = np.eye(4)
    R = rot_matrix(rot)
    o = np.array(origin if origin is not None else [0, 0, 0], dtype=float)
    M[:3, :3] = R
    M[:3, 3] = o - R @ o
    return M


def norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame(up, out):
    """Rotation matrix whose local +Y is `up` and local -Z (north face) points along `out`."""
    y = norm(up)
    z = -np.asarray(out, float)
    z = norm(z - y * (z @ y))
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def catmull(pts, n=8):
    """Catmull-Rom spline through pts, n samples per span."""
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


def resample(poly, seg):
    """Resample a polyline to roughly `seg` long pieces (seg may be a function of arc fraction)."""
    poly = np.asarray(poly, float)
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    ts = [0.0]
    while ts[-1] < L - 1e-6:
        step = seg(ts[-1] / L) if callable(seg) else seg
        ts.append(min(L, ts[-1] + step))
    if len(ts) > 2 and ts[-1] - ts[-2] < 0.35 * (seg(1.0) if callable(seg) else seg):
        ts.pop(-2)
    ts = np.array(ts)
    return np.stack([np.interp(ts, s, poly[:, k]) for k in range(3)], axis=1), ts / L


FACE_CORNERS = {
    'north': [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
    'south': [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    'east': [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    'west': [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    'up': [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    'down': [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
}
FACE_NORMAL = {'north': (0, 0, -1), 'south': (0, 0, 1), 'east': (1, 0, 0), 'west': (-1, 0, 0), 'up': (0, 1, 0), 'down': (0, -1, 0)}


# ---------------------------------------------------------------- scene
class Bone:
    def __init__(self, name, parent, pivot, rot=(0, 0, 0)):
        self.name, self.parent = name, parent
        self.pivot = [float(v) for v in pivot]
        self.rot = [float(v) for v in rot]
        self.cubes, self.children = [], []


class Cube:
    __slots__ = ('name', 'bone', 'frm', 'to', 'origin', 'rot', 'mat', 'mp', 'dens', 'faces', 'R')

    def __init__(self, name, bone, frm, to, origin, rot, mat, mp, dens):
        self.name, self.bone = name, bone
        self.frm, self.to = np.minimum(frm, to).astype(float), np.maximum(frm, to).astype(float)
        self.origin = np.asarray(origin, float)
        self.rot = [float(r) for r in rot]
        self.R = rot_matrix(self.rot)
        self.mat, self.mp, self.dens = mat, mp, dens
        self.faces = {}


class Builder:
    def __init__(self):
        self.bones = {}
        self.cubes = []
        self.side = 1          # +1 authoring the right (+X) half, -1 mirrors to the left
        self.names = {}        # per-side bone-name remapping

    # -- bones
    def bone(self, name, parent, pivot, rot=(0, 0, 0)):
        name, parent = self.n(name), self.n(parent)
        s = self.side
        b = Bone(name, parent, (pivot[0] * s, pivot[1], pivot[2]), (rot[0], rot[1] * s, rot[2] * s))
        assert name not in self.bones, name
        self.bones[name] = b
        if parent:
            self.bones[parent].children.append(name)
        return b

    def n(self, name):
        if name is None:
            return None
        return self.names.get(name, name) if self.side < 0 else name

    def world(self, bone):
        chain = []
        b = self.bones[bone]
        while b:
            chain.append(b)
            b = self.bones[b.parent] if b.parent else None
        M = np.eye(4)
        for b in reversed(chain):
            M = M @ pivot_xf(b.pivot, b.rot)
        return M

    # -- cubes
    def box(self, bone, c, s, rot=(0, 0, 0), mat='skin', name=None, dens=None, origin=None, **mp):
        """Axis box from centre `c` and size `s`, rotated about its centre (or `origin`)."""
        sd = self.side
        c = np.array([c[0] * sd, c[1], c[2]], float)
        s = np.abs(np.asarray(s, float))
        rot = (rot[0], rot[1] * sd, rot[2] * sd)
        o = c if origin is None else np.array([origin[0] * sd, origin[1], origin[2]], float)
        cu = Cube(name or mat, self.n(bone), c - s / 2, c + s / 2, o, rot, mat, dict(mp, side=sd), dens)
        self.bones[cu.bone].cubes.append(cu)
        self.cubes.append(cu)
        return cu

    def obox(self, bone, c, R, s, mat='skin', name=None, dens=None, **mp):
        """Oriented box: columns of R are the local axes (already in final, un-mirrored space)."""
        c = np.asarray(c, float)
        s = np.abs(np.asarray(s, float))
        cu = Cube(name or mat, bone, c - s / 2, c + s / 2, c, mat_to_euler(R), mat, dict(mp, side=self.side), dens)
        self.bones[bone].cubes.append(cu)
        self.cubes.append(cu)
        return cu

    def rbox(self, bone, c, s, rot=(0, 0, 0), mat='skin', k=0.74, k2=None, **kw):
        """Chamfered 'rounded' box made of three intersecting boxes."""
        k2 = k if k2 is None else k2
        s = np.asarray(s, float)
        out = []
        for i in range(3):
            f = np.array([k2, k2, k2])
            f[i] = 1.0
            if i == 1:
                f = np.array([k, 1.0, k])
            out.append(self.box(bone, c, s * f, rot, mat, **kw))
        return out

    def ring(self, bone, c, rx, rz, h, mat='skin', rot=(0, 0, 0), **kw):
        """Octagonal prism slice built from four slabs. rx/rz are half extents."""
        t = 0.4142
        out = [self.box(bone, c, (2 * rx, h, 2 * rz * t), rot, mat, **kw),
               self.box(bone, c, (2 * rx * t, h, 2 * rz), rot, mat, **kw)]
        n = math.hypot(rx, rz)
        hl, hw = (1 + t) / 2 * n, (1 - t) / 2 * n * 0.98
        ang = math.degrees(math.atan2(rz, rx))
        sd = self.side
        Rb = rot_matrix((rot[0], rot[1] * sd, rot[2] * sd))
        cc = (c[0] * sd, c[1], c[2])
        for a in (-ang, ang):
            out.append(self.obox(self.n(bone), cc, Rb @ rot_matrix((0, a, 0)), (2 * hl, h, 2 * hw), mat, **kw))
        return out

    def dome(self, bone, c, sx, sy, depth, rot=(0, 0, 0), axis=(0, 0, -1), layers=3, shrink=0.18, mat='skin', **kw):
        """Terraced bulge (muscle belly, armour boss). `c` is the centre of its base plane; the two
        lateral sizes follow the remaining axes in x,y,z order and it grows `depth` along `axis`."""
        ax = int(np.argmax(np.abs(axis)))
        sgn = 1.0 if axis[ax] > 0 else -1.0
        lat = [k for k in range(3) if k != ax]
        out = []
        for i in range(layers):
            f = 1 - shrink * i
            th = depth * (i + 1) / layers
            size = np.zeros(3)
            size[lat[0]], size[lat[1]], size[ax] = sx * f, sy * f, th
            cc = np.array(c, float)
            cc[ax] += sgn * th / 2
            blob = np.zeros(3)
            blob[lat[0]], blob[lat[1]], blob[ax] = sx, sy, depth
            out.append(self.box(bone, cc, size, rot, mat, origin=c, blob=tuple(blob), **kw))
        return out

    def limb(self, bone, x, z, ys, rxs, rzs, mat='skin', zoff=None, xoff=None, **kw):
        """Stack of ring slices between successive heights `ys` using radii profiles."""
        out = []
        for i in range(len(ys) - 1):
            y0, y1 = ys[i], ys[i + 1]
            rx = (rxs[i] + rxs[i + 1]) / 2
            rz = (rzs[i] + rzs[i + 1]) / 2
            zo = 0 if zoff is None else (zoff[i] + zoff[i + 1]) / 2
            xo = 0 if xoff is None else (xoff[i] + xoff[i + 1]) / 2
            ov = 0.12
            out += self.ring(bone, (x + xo, (y0 + y1) / 2, z + zo), rx, rz, abs(y1 - y0) + ov, mat, **kw)
        return out

    def ribbon(self, bone, pts, width, thick, out, mat='hair', seg=2.2, overlap=0.18, tip=False,
               v0=0.0, v1=1.0, roll=0.0, **mp):
        """Chain of oriented thin boxes following a polyline (root first).

        bone  : name or f(i, mid_point, t) -> name
        width : number or f(t) ; out: outward vector or f(t, tangent, mid) ; thick: number or f(t)
        """
        P, T = resample(np.asarray(pts, float), seg)
        cubes = []
        nseg = len(P) - 1
        Ltot = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        for i in range(nseg):
            a, b = P[i], P[i + 1]
            t = (T[i] + T[i + 1]) / 2
            up = a - b
            L = np.linalg.norm(up)
            if L < 1e-6:
                continue
            mid = (a + b) / 2
            o = out(t, up / L, mid) if callable(out) else out
            R = frame(up, o)
            if roll:
                rr = roll(t) if callable(roll) else roll
                R = R @ rot_matrix((0, rr, 0))
            w = width(t) if callable(width) else width
            th = thick(t) if callable(thick) else thick
            bn = bone(i, mid, t) if callable(bone) else bone
            vv0 = v0 + (v1 - v0) * T[i]
            vv1 = v0 + (v1 - v0) * T[i + 1]
            cubes.append(self.obox(bn, mid, R, (w, L + overlap, th), mat, va=vv0, vb=vv1, sa=T[i] * Ltot,
                                   lead=(i == 0), last=(i == nseg - 1), **mp))
            if tip and i == nseg - 1:
                Rt = R @ rot_matrix((0, 0, 45))
                d = w / math.sqrt(2)
                cubes.append(self.obox(bn, b, Rt, (d, d, th * 0.9), mat, va=vv1, vb=vv1, sa=Ltot, tipd=1, **mp))
        return cubes


# ---------------------------------------------------------------- faces / baking
class Face:
    __slots__ = ('cube', 'name', 'c0', 'ca', 'cw', 'wu', 'hu', 'w', 'h', 'x', 'y', 'ao', 'hidden', 'na', 'nw')


def build_faces(B, density):
    faces = []
    Mw = {name: B.world(name) for name in B.bones}
    for cu in B.cubes:
        M = Mw[cu.bone]
        b = (cu.frm, cu.to)
        size = cu.to - cu.frm
        for fname, idx in FACE_CORNERS.items():
            f = Face()
            f.cube, f.name = cu, fname
            c0 = np.array([[b[ix][0], b[iy][1], b[iz][2]] for ix, iy, iz in idx])
            f.c0 = c0
            f.ca = (cu.R @ (c0 - cu.origin).T).T + cu.origin
            f.cw = (M[:3, :3] @ f.ca.T).T + M[:3, 3]
            f.wu = float(np.linalg.norm(c0[1] - c0[0]))
            f.hu = float(np.linalg.norm(c0[3] - c0[0]))
            f.na = cu.R @ np.array(FACE_NORMAL[fname], float)
            f.nw = M[:3, :3] @ f.na
            f.w = f.h = 1
            f.hidden = False
            f.ao = None
            cu.faces[fname] = f
            faces.append(f)
    return faces


def size_faces(faces, density, scale=1.0):
    """Texel size of every face; per-cube densities are authored relative to a 20 px/unit default."""
    for f in faces:
        d = (f.cube.dens * density / 20.0 if f.cube.dens else density) * scale
        if f.hidden:
            f.w = f.h = 2
        else:
            f.w = max(1, int(round(f.wu * d)))
            f.h = max(1, int(round(f.hu * d)))


def face_grid(corners, w, h):
    s = (np.arange(w) + 0.5) / w
    t = (np.arange(h) + 0.5) / h
    S, T = np.meshgrid(s, t)
    return corners[0] + S[..., None] * (corners[1] - corners[0]) + T[..., None] * (corners[3] - corners[0])


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], axis=1)


def compute_ao(faces, ndirs=64, sdens=5.0, res=4.0, bias=0.30, log=print):
    """Shadow-map ambient occlusion sampled on a coarse per-face grid."""
    pts, nrm, shape, ofs = [], [], [], [0]
    for f in faces:
        wc = max(1, int(math.ceil(f.wu * sdens)))
        hc = max(1, int(math.ceil(f.hu * sdens)))
        g = face_grid(f.cw, wc, hc).reshape(-1, 3)
        pts.append(g)
        nrm.append(np.repeat(f.nw[None], len(g), 0))
        shape.append((hc, wc))
        ofs.append(ofs[-1] + len(g))
    P = np.concatenate(pts).astype(np.float32)
    N = np.concatenate(nrm).astype(np.float32)
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
        # slope-scaled bias: a depth-map pixel spans more depth on surfaces tilted away from the light
        cc = np.clip(cw, 0.12, 1.0)
        vis = dep >= dm[idx] - (bias + (1.3 / res) * np.sqrt(1 - cc * cc) / cc)
        wgt = np.clip(cw, 0, None) * (0.62 + 0.38 * d[1])
        acc += vis * wgt
        wsum += wgt
        seen |= vis & (cw > 0.08)
    ao = acc / np.maximum(wsum, 1e-6)
    for i, f in enumerate(faces):
        a, b = ofs[i], ofs[i + 1]
        f.ao = ao[a:b].reshape(shape[i])
        f.hidden = not seen[a:b].any()
    return ao


def pack(faces, size, pad=1):
    """Shelf-pack face rects into a size x size atlas. Returns used height or None."""
    order = sorted(faces, key=lambda f: (-f.h, -f.w))
    x = y = pad
    rowh = 0
    for f in order:
        if x + f.w + pad > size:
            x = pad
            y += rowh + 2 * pad
            rowh = 0
        if y + f.h + pad > size:
            return None
        f.x, f.y = x, y
        x += f.w + 2 * pad
        rowh = max(rowh, f.h)
    return y + rowh + pad


class Ctx:
    pass


def upsample(a, h, w):
    """Bilinear resize of a small 2D array with edge clamping."""
    hc, wc = a.shape
    if hc == 1 and wc == 1:
        return np.full((h, w), float(a[0, 0]), np.float32)
    y = np.clip((np.arange(h) + 0.5) / h * hc - 0.5, 0, hc - 1)
    x = np.clip((np.arange(w) + 0.5) / w * wc - 0.5, 0, wc - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, hc - 1), np.minimum(x0 + 1, wc - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    return (a[y0][:, x0] * (1 - fy) * (1 - fx) + a[y0][:, x1] * (1 - fy) * fx
            + a[y1][:, x0] * fy * (1 - fx) + a[y1][:, x1] * fy * fx)


def bake(B, faces, size, materials, ao_strength=0.55, log=print):
    atlas = np.zeros((size, size, 4), np.uint8)
    Mw = {name: B.world(name) for name in B.bones}
    for f in faces:
        cu = f.cube
        c = Ctx()
        c.face, c.mp, c.cube = f.name, cu.mp, cu
        c.size = cu.to - cu.frm
        c.na, c.nw = f.na, f.nw
        c.w, c.h, c.wu, c.hu = f.w, f.h, f.wu, f.hu
        c.pa = face_grid(f.ca, f.w, f.h)
        M = Mw[cu.bone]
        c.pw = c.pa @ M[:3, :3].T + M[:3, 3]
        p0 = face_grid(f.c0, f.w, f.h)
        c.pl = (p0 - (cu.frm + cu.to) / 2) / np.maximum(c.size, 1e-6)
        s = (np.arange(f.w) + 0.5) / f.w
        t = (np.arange(f.h) + 0.5) / f.h
        c.S, c.T = np.meshgrid(s, t)
        c.U, c.Vv = c.S * f.wu, c.T * f.hu
        rgb = materials[cu.mat](c)
        if f.ao is not None:
            ao = upsample(np.clip(f.ao, 0, 1), f.h, f.w)
            k = cu.mp.get('ao', 1.0) * ao_strength
            sh = 1 - k * (1 - np.clip(ao, 0, 1) ** 0.85)
            # shadows drift slightly towards a warm/cool tint instead of plain black
            tint = np.array(cu.mp.get('aotint', (0.94, 0.90, 0.96)))
            rgb = rgb * (sh[..., None] * (1 - (1 - sh[..., None]) * (1 - tint)))
        if f.hidden:
            rgb = np.broadcast_to(rgb.reshape(-1, 3).mean(0) * 0.8, (f.h, f.w, 3))
        px = np.clip(rgb, 0, 255).astype(np.uint8)
        px = np.pad(px, ((1, 1), (1, 1), (0, 0)), mode='edge')
        atlas[f.y - 1:f.y + f.h + 1, f.x - 1:f.x + f.w + 1, :3] = px
        atlas[f.y - 1:f.y + f.h + 1, f.x - 1:f.x + f.w + 1, 3] = 255
    return atlas


# ---------------------------------------------------------------- export
def _uid(tag):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'thor-boss/' + tag))


def export_bbmodel(B, src, atlas_png_bytes, tex_name, size, uv_size):
    """Write bones/cubes into a copy of the source project, reusing legacy group entries by name."""
    d = copy.deepcopy(src)
    G = {g['uuid']: g for g in d['groups']}
    # legacy groups reachable from populated top-level trees keep their uuid
    E = {e['uuid'] for e in d['elements']}

    def has_cubes(n):
        return isinstance(n, str) and n in E or (not isinstance(n, str) and any(has_cubes(c) for c in n.get('children', [])))

    legacy = {}

    lparent = {}

    def collect(n, parent=None):
        if isinstance(n, str):
            return
        name = G[n['uuid']]['name']
        if name not in legacy:
            legacy[name] = G[n['uuid']]
            lparent[name] = parent
        for c in n.get('children', []):
            collect(c, name)

    used_top, empty_top = [], []
    for n in d['outliner']:
        if isinstance(n, str):
            continue
        (used_top if has_cubes(n) else empty_top).append(n)
    for n in used_top:
        collect(n)
    template = next(iter(legacy.values()))
    groups, by_name = [], {}
    scale = uv_size / size
    elements = []
    for name, b in B.bones.items():
        if name in legacy:
            g = legacy[name]
        else:
            g = copy.deepcopy(template)
            g['name'], g['uuid'] = name, _uid('bone/' + name)
        g['origin'] = [round(v, 4) for v in b.pivot]
        g['rotation'] = [round(v, 4) for v in b.rot]
        g['children'] = []
        g['isOpen'] = b.parent is None
        by_name[name] = g
        groups.append(g)
    # legacy bones the generator does not populate stay in the rig as empty groups under their old parent
    extra = {}
    for name in legacy:
        if name in B.bones:
            continue
        par = lparent.get(name)
        while par is not None and par not in B.bones and par not in extra:
            par = lparent.get(par)
        g = legacy[name]
        g['children'] = []
        by_name[name] = g
        groups.append(g)
        extra.setdefault(par, [])
        extra.setdefault(name, [])
        extra[par].append(name)
    colors = {'skin': 1, 'hair': 5, 'cloth': 0, 'gold': 2, 'leather': 6, 'armor': 3}
    for i, cu in enumerate(B.cubes):
        e = {
            'name': cu.name, 'box_uv': False, 'render_order': 'default', 'locked': False, 'export': True,
            'scope': 0, 'allow_mirror_modeling': True,
            'from': [round(float(v), 4) for v in cu.frm], 'to': [round(float(v), 4) for v in cu.to],
            'autouv': 0, 'color': colors.get(cu.mat, 4),
            'rotation': [round(float(v), 4) for v in cu.rot], 'origin': [round(float(v), 4) for v in cu.origin],
            'faces': {}, 'type': 'cube', 'uuid': _uid(f'cube/{i}'),
        }
        for fname in ('north', 'east', 'south', 'west', 'up', 'down'):
            f = cu.faces[fname]
            e['faces'][fname] = {'uv': [round(f.x * scale, 4), round(f.y * scale, 4), round((f.x + f.w) * scale, 4), round((f.y + f.h) * scale, 4)], 'texture': 0}
        cu.faces['_uuid'] = e['uuid']
        elements.append(e)

    def node(name):
        if name not in B.bones:
            return {'uuid': by_name[name]['uuid'], 'isOpen': False, 'children': [node(c) for c in extra.get(name, [])]}
        b = B.bones[name]
        return {'uuid': by_name[name]['uuid'], 'isOpen': b.parent is None,
                'children': [node(c) for c in b.children] + [node(c) for c in extra.get(name, [])]
                + [cu.faces['_uuid'] for cu in b.cubes]}

    roots = [n for n, b in B.bones.items() if b.parent is None]
    outliner = [node(r) for r in roots]
    outliner += [node(c) for c in extra.get(None, [])]
    moved = {g['uuid'] for g in by_name.values()}
    outliner += [n for n in empty_top if n['uuid'] not in moved]
    d['elements'] = elements
    d['groups'] = groups + [G[n['uuid']] for n in empty_top if n['uuid'] not in moved]
    d['outliner'] = outliner
    t = d['textures'][0]
    t['name'] = tex_name
    t['width'] = t['height'] = size
    t['uv_width'] = t['uv_height'] = uv_size
    t['source'] = 'data:image/png;base64,' + base64.b64encode(atlas_png_bytes).decode()
    d['resolution'] = {'width': uv_size, 'height': uv_size}
    for cu in B.cubes:
        cu.faces.pop('_uuid', None)
    return d
