"""Bedrock geometry (.geo.json) loader that yields render quads in Blockbench space.

Conversions follow Blockbench's Bedrock importer: pivots have X negated, rotations have X and Y
negated, cube origin is the min corner with X flipped, and up/down face UVs swap their corners.
"""
import json, math
import numpy as np

FACE_CORNERS = {
    'north': [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
    'south': [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    'east': [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    'west': [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    'up': [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    'down': [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
}


def rot_matrix(rot):
    if rot is None or not any(rot):
        return np.eye(3)
    x, y, z = [math.radians(a) for a in rot]
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pivot_xf(origin, rot):
    M = np.eye(4)
    R = rot_matrix(rot)
    o = np.array(origin, float)
    M[:3, :3] = R
    M[:3, 3] = o - R @ o
    return M


def bb_pivot(p):
    p = p or [0, 0, 0]
    return [-p[0], p[1], p[2]]


def bb_rot(r):
    r = r or [0, 0, 0]
    return [-r[0], -r[1], r[2]]


class GeoModel:
    def __init__(self, path=None, data=None, pose=None):
        self.d = data if data is not None else json.load(open(path))
        g = self.d['minecraft:geometry'][0]
        self.desc = g['description']
        self.tw, self.th = self.desc['texture_width'], self.desc['texture_height']
        self.bones = {b['name']: b for b in g['bones']}
        self.order = [b['name'] for b in g['bones']]
        self.pose = pose or {}
        self._w = {}

    def path(self, name):
        out = []
        while name:
            out.append(name)
            name = self.bones[name].get('parent')
        return out[::-1]

    def world(self, name):
        if name not in self._w:
            b = self.bones[name]
            par = b.get('parent')
            M = self.world(par) if par else np.eye(4)
            rot = bb_rot(b.get('rotation'))
            if name in self.pose:        # pose given in Blockbench degrees, added to the rest rotation
                rot = [a + c for a, c in zip(rot, self.pose[name])]
            self._w[name] = M @ pivot_xf(bb_pivot(b.get('pivot')), rot)
        return self._w[name]

    def quads(self, filt=None):
        out = []
        for name in self.order:
            b = self.bones[name]
            path = self.path(name)
            if filt and not filt(name, path):
                continue
            M = self.world(name)
            for ci, c in enumerate(b.get('cubes', [])):
                o, s = c['origin'], c['size']
                inf = c.get('inflate', 0) or 0
                lo = np.array([-(o[0] + s[0]), o[1], o[2]], float) - inf
                hi = np.array([-o[0], o[1] + s[1], o[2] + s[2]], float) + inf
                Me = M @ pivot_xf(bb_pivot(c.get('pivot')), bb_rot(c.get('rotation')))
                bx = (lo, hi)
                uvd = c['uv']
                assert isinstance(uvd, dict), 'box uv not supported'
                for fname, f in uvd.items():
                    u, v = f['uv']
                    w, h = f['uv_size']
                    if w == 0 or h == 0:
                        continue
                    u1, v1, u2, v2 = u, v, u + w, v + h
                    if fname in ('up', 'down'):
                        u1, v1, u2, v2 = u2, v2, u1, v1
                    cs = np.array([[bx[ix][0], bx[iy][1], bx[iz][2], 1.0] for ix, iy, iz in FACE_CORNERS[fname]])
                    wc = (Me @ cs.T).T[:, :3]
                    uvs = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
                    rr = int(f.get('uv_rotation', 0) or 0) // 90
                    uvs = uvs[-rr:] + uvs[:-rr] if rr else uvs
                    out.append((wc, np.array(uvs, float) / [self.tw, self.th], {'bone': name, 'cube': ci, 'face': fname}))
        return out

    def bounds(self, filt=None):
        pts = np.concatenate([q[0] for q in self.quads(filt)])
        return pts.min(0), pts.max(0)
