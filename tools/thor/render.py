"""Minimal Blockbench (free format) loader + orthographic software renderer."""
import json, math, base64, io
import numpy as np
from PIL import Image


def rot_matrix(rot):
    if not rot or not any(rot):
        return np.eye(3)
    x, y, z = [math.radians(a) for a in rot]
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pivot_xf(origin, rot):
    """4x4 transform rotating around origin."""
    M = np.eye(4)
    R = rot_matrix(rot)
    o = np.array(origin or [0, 0, 0], dtype=float)
    M[:3, :3] = R
    M[:3, 3] = o - R @ o
    return M


FACE_CORNERS = {
    # indices into (x, y, z) picking from (lo, hi): TL, TR, BR, BL
    'north': [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
    'south': [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    'east': [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    'west': [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    'up': [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    'down': [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
}


class Model:
    def __init__(self, path=None, data=None):
        self.d = data if data is not None else json.load(open(path))
        self.G = {g['uuid']: g for g in self.d.get('groups', [])}
        self.E = {e['uuid']: e for e in self.d['elements']}
        self.el_xf = {}      # element uuid -> parent world matrix
        self.el_path = {}    # element uuid -> list of group names
        self.group_node = {}
        for n in self.d['outliner']:
            self._walk(n, np.eye(4), [])

    def group(self, node):
        return self.G.get(node.get('uuid'), node)

    def _walk(self, n, M, path):
        if isinstance(n, str):
            self.el_xf[n] = M
            self.el_path[n] = path
            return
        g = self.group(n)
        M2 = M @ pivot_xf(g.get('origin'), g.get('rotation'))
        self.group_node[g['uuid']] = n
        p2 = path + [g.get('name')]
        for c in n.get('children', []):
            self._walk(c, M2, p2)

    def texture(self, i=0):
        src = self.d['textures'][i]['source']
        return Image.open(io.BytesIO(base64.b64decode(src.split(',', 1)[1]))).convert('RGBA')

    def quads(self, filt=None):
        """Yield (world 4x3 corners TL,TR,BR,BL, uv 4x2 in 0..1, element)."""
        W, H = self.d['resolution']['width'], self.d['resolution']['height']
        out = []
        for uid, M in self.el_xf.items():
            e = self.E[uid]
            if e.get('type', 'cube') != 'cube' or e.get('visibility') is False:
                continue
            if filt and not filt(e, self.el_path[uid]):
                continue
            inf = e.get('inflate', 0) or 0
            lo = np.array(e['from'], float) - inf
            hi = np.array(e['to'], float) + inf
            Me = M @ pivot_xf(e.get('origin'), e.get('rotation'))
            b = (lo, hi)
            for fname, f in e['faces'].items():
                if f.get('texture') is None:
                    continue
                uv = f['uv']
                if uv[0] == uv[2] or uv[1] == uv[3]:
                    continue
                cs = np.array([[b[ix][0], b[iy][1], b[iz][2], 1.0] for ix, iy, iz in FACE_CORNERS[fname]])
                wc = (Me @ cs.T).T[:, :3]
                u1, v1, u2, v2 = uv
                uvs = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
                r = int(f.get('rotation', 0) or 0) // 90
                uvs = uvs[-r:] + uvs[:-r] if r else uvs
                out.append((wc, np.array(uvs, float) / [W, H], e))
        return out

    def bounds(self):
        pts = np.concatenate([q[0] for q in self.quads()])
        return pts.min(0), pts.max(0)


def render(quads, tex, yaw=0, pitch=0, size=(700, 1100), center=None, scale=None, bg=(58, 62, 74), light=True, ss=2):
    """Orthographic render. yaw=0 looks at the front (north face, camera at -Z)."""
    w, h = size[0] * ss, size[1] * ss
    ya, pa = math.radians(yaw), math.radians(pitch)
    # camera basis: start looking toward +Z from -Z, right = -X
    right = np.array([-math.cos(ya), 0, -math.sin(ya)])
    fwd0 = np.array([-math.sin(ya), 0, math.cos(ya)])
    up = np.array([0, 1, 0]) * math.cos(pa) - fwd0 * math.sin(pa) * -1
    fwd = fwd0 * math.cos(pa) - np.array([0, 1, 0]) * math.sin(pa)
    up = np.cross(fwd, right) * -1
    up = up / np.linalg.norm(up)
    if up[1] < 0:
        up = -up
    allp = np.concatenate([q[0] for q in quads])
    if center is None:
        center = (allp.min(0) + allp.max(0)) / 2
    center = np.array(center, float)
    if scale is None:
        px = (allp - center) @ right
        py = (allp - center) @ up
        scale = 0.94 * min(w / (2 * np.abs(px).max() + 1e-6), h / (2 * np.abs(py).max() + 1e-6))
    else:
        scale = scale * ss
    T = np.asarray(tex)
    th, tw = T.shape[:2]
    col = np.zeros((h, w, 3), np.float32)
    col[:] = bg
    zb = np.full((h, w), np.inf, np.float32)
    L = np.array([0.35, 0.75, -0.55])
    L = L / np.linalg.norm(L)
    Lw = right * -0.35 + up * 0.6 - fwd * 0.72
    Lw = Lw / np.linalg.norm(Lw)
    for wc, uv, e in quads:
        rel = wc - center
        sx = rel @ right * scale + w / 2
        sy = h / 2 - rel @ up * scale
        sz = rel @ fwd
        if sx.max() < 0 or sx.min() >= w or sy.max() < 0 or sy.min() >= h:
            continue
        n = np.cross(wc[3] - wc[0], wc[1] - wc[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if n @ fwd > 0.0:   # back-facing
            continue
        shade = 1.0
        if light:
            shade = 0.62 + 0.38 * max(0.0, float(n @ Lw)) + 0.08 * float(n @ up)
        for tri in ((0, 1, 2), (0, 2, 3)):
            ax, ay, bx, by, cx, cy = sx[tri[0]], sy[tri[0]], sx[tri[1]], sy[tri[1]], sx[tri[2]], sy[tri[2]]
            x0, x1 = max(int(math.floor(min(ax, bx, cx))), 0), min(int(math.ceil(max(ax, bx, cx))) + 1, w)
            y0, y1 = max(int(math.floor(min(ay, by, cy))), 0), min(int(math.ceil(max(ay, by, cy))) + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-9:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
            l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
            l2 = 1 - l0 - l1
            m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
            if not m.any():
                continue
            z = l0 * sz[tri[0]] + l1 * sz[tri[1]] + l2 * sz[tri[2]]
            sub = zb[y0:y1, x0:x1]
            m &= z < sub
            if not m.any():
                continue
            u = l0 * uv[tri[0], 0] + l1 * uv[tri[1], 0] + l2 * uv[tri[2], 0]
            v = l0 * uv[tri[0], 1] + l1 * uv[tri[1], 1] + l2 * uv[tri[2], 1]
            # keep samples strictly inside the face rect to avoid bleeding
            umin, umax = uv[:, 0].min(), uv[:, 0].max()
            vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
            ui = np.clip((np.clip(u, umin, umax) * tw).astype(int), int(math.floor(umin * tw)), max(int(math.ceil(umax * tw)) - 1, int(math.floor(umin * tw))))
            vi = np.clip((np.clip(v, vmin, vmax) * th).astype(int), int(math.floor(vmin * th)), max(int(math.ceil(vmax * th)) - 1, int(math.floor(vmin * th))))
            ui = np.clip(ui, 0, tw - 1)
            vi = np.clip(vi, 0, th - 1)
            s = T[vi, ui]
            m &= s[..., 3] > 100
            if not m.any():
                continue
            sub[m] = z[m]
            csub = col[y0:y1, x0:x1]
            csub[m] = s[..., :3][m] * shade
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    if ss > 1:
        img = img.resize(size, Image.LANCZOS)
    return img


def sheet(quads, tex, views, size=(420, 760), **kw):
    ims = [render(quads, tex, yaw=y, pitch=p, size=size, **kw) for y, p in views]
    out = Image.new('RGB', (size[0] * len(ims), size[1]))
    for i, im in enumerate(ims):
        out.paste(im, (i * size[0], 0))
    return out
