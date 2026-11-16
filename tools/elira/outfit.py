"""Belt, skirt panels, sash and jewellery."""
import math
import numpy as np
from kit import support, ring_vertices, dir_n, dir_ex, norm, interp
from body import K8, K12, tor_ring, tor, CZ
import paint

# facet directions of the skirt: centre tabard, right front (2), right side (2), back (5), left side (2), left front (2)
SK = [0, 24, 49, 76, 104, 130.4, 155.2, 180, 204.8, 229.6, 256, 284, 311, 336]
# height, half width, front z, back z of the skirt surface
SKP = np.array([(0.3, 7.0, -3.7, 5.0), (6.0, 5.7, -2.95, 4.2), (12.0, 4.4, -2.3, 3.4), (17.0, 3.72, -2.02, 2.78), (21.0, 3.66, -1.96, 2.56), (22.7, 3.22, -1.76, 2.06)])
SKZ = 0.25
_rings = {}


def sk_h(y, grow=0.0):
    rx, zf, zb = [interp(y, SKP[:, 0], SKP[:, k]) for k in (1, 2, 3)]
    return support(SK, rx + grow, SKZ - zf + grow, zb - SKZ + grow)


def sk_ring(y):
    if y not in _rings:
        _rings[y] = ring_vertices(SK, sk_h(y), y, (0.0, SKZ))
    return _rings[y]


def facet(B, k, ys, bone, kind, off=0.0, bl=True, br=True, tip=0.4, hv=1.4, trim_l=None, over=0.1, seed=0, **mp):
    """One gore of the skirt: the facet with direction SK[k], running down the heights ys."""
    n, ex = dir_n(SK[k]), dir_ex(SK[k])
    C, half = [], []
    for y in ys:
        V = sk_ring(y)
        l, r = V[k] + n * off, V[k - 1] + n * off
        if trim_l is not None:
            l = l + (r - l) * trim_l(y)
        l, r = l - ex * over, r + ex * over
        C.append((l + r) / 2)
        half.append(np.linalg.norm(r - l) / 2)
    mp.setdefault('crest', 1)
    return B.ribbon(bone, np.array(C), half, half, ex, n, 'panel', afn=paint.panel_alpha, kind=kind, bl=bl, br=br,
                    tip=tip, hv=hv, seed=seed, thick=0.08, **mp)


def skirt(B):
    up, low = [22.7, 21.0, 19.0, 17.0, 14.5, 12.0], [9.6, 7.2, 4.8, 2.4]
    split = lambda name, y0=12.0: (lambda i, mid: name if mid[1] > y0 else name + '_lo')
    # centre tabard
    facet(B, 0, [22.7, 21.0, 19.0, 17.0, 15.0, 13.0, 11.0, 9.6, 8.3], split('skirt_m', 15.0), 'dark', off=0.12, tip=8.3, hv=1.25, seed=1, crest=1)
    # right front: white silk
    facet(B, 1, up + low + [0.6], split('skirt_fr'), 'white', bl=False, tip=0.6, seed=2)
    facet(B, 2, up + low + [0.6], split('skirt_fr'), 'white', br=False, tip=0.6, seed=3)
    # right side (mirrored onto the left side)
    side = facet(B, 3, up + low + [0.5], split('skirt_sr'), 'dark', off=0.09, bl=False, tip=0.5, seed=4)
    side += facet(B, 4, up + low + [0.5], split('skirt_sr'), 'dark', off=0.09, br=False, tip=0.5, seed=5)
    B.mirror(side)
    # back: five gores, alternately dark and white, with a little train
    for j, k in enumerate((5, 6, 7, 8, 9)):
        facet(B, k, up + low + [0.25], split('skirt_back'), 'dark' if j % 2 == 0 else 'white', off=0.0 if j % 2 else 0.07, tip=0.25, seed=6 + j)
    # left front: dark, cut away into a slit that opens towards the hem
    facet(B, 12, up + low + [0.6], split('skirt_fl'), 'dark', br=False, tip=0.6, seed=12)
    slit = lambda y: 0.2 + 0.62 * min(1.0, (22.7 - y) / 16.0) ** 0.8
    facet(B, 13, up + low + [0.6], split('skirt_fl'), 'dark', bl=True, br=False, tip=0.6, trim_l=slit, seed=13, crest=0)


def sk_point(phi, y):
    """Point of the skirt ellipse whose outward normal has azimuth phi."""
    rx, zf, zb = [interp(y, SKP[:, 0], SKP[:, k]) for k in (1, 2, 3)]
    a = math.radians(phi)
    rz = SKZ - zf if math.cos(a) > 0 else zb - SKZ
    h = math.hypot(rx * math.sin(a), rz * math.cos(a))
    return np.array([rx * rx * math.sin(a) / h, y, SKZ - rz * rz * math.cos(a) / h])


def sash(B):
    """Dark sash with a pointed end, hanging from a jewelled clasp on the right hip."""
    phi = 58.0
    n, ex = dir_n(phi), dir_ex(phi)
    ys = [22.5, 20.8, 19.0, 17.0, 15.0, 13.0, 11.0, 9.0, 7.0, 5.2]
    C = np.array([sk_point(phi, y) + n * (0.36 + 0.015 * (22.5 - y)) for y in ys])
    half = [0.62 + 0.012 * (22.5 - y) for y in ys]
    bone = lambda i, mid: 'sash' if mid[1] > 13.0 else 'sash_lo'
    B.ribbon(bone, C, half, half, ex, n, 'panel', afn=paint.panel_alpha, kind='dark', bl=True, br=True,
             tip=5.2, hv=1.5, seed=20, crest=1, thick=0.08)
    c0 = C[0] + n * 0.12 + np.array([0.0, 0.2, 0.0])
    R = np.stack([-ex, np.array([0.0, 1.0, 0.0]), -n], axis=1) @ rotz(45)
    B.box('hips', c0, (0.9, 0.9, 0.14), R=R, mat='gold')
    B.box('hips', c0 + n * 0.08, (0.52, 0.52, 0.1), R=R, mat='gem', dens=64)


def rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def band_loft(B, bone, dirs, ring_fn, y0, y1, grow, mat, bevel=0.07, **mp):
    """Raised band hugging a limb: ring_fn(y, grow) gives the ring; the rims are bevelled back to the skin."""
    rings = [ring_fn(y0, 0.015), ring_fn(y0 + bevel, grow), ring_fn(y1 - bevel, grow), ring_fn(y1, 0.015)]
    return B.loft(bone, dirs, rings, mat, **mp)


def belt(B):
    band_loft(B, 'hips', K12, tor_ring, 21.93, 23.12, 0.2, 'belt', bevel=0.12, dens=34)
    zb = tor(22.52)[1] - 0.2
    B.box('hips', (0, 22.52, zb - 0.04), (1.62, 1.62, 0.14), (0, 0, 45), 'gold')
    B.box('hips', (0, 22.52, zb - 0.11), (1.24, 1.24, 0.1), (0, 0, 45), 'dark')
    B.box('hips', (0, 22.52, zb - 0.17), (0.9, 0.9, 0.1), (0, 0, 45), 'gem', dens=64)
    for s in (1, -1):
        B.box('hips', (s * 1.36, 22.52, zb + 0.03), (0.5, 0.32, 0.12), (0, -s * 14, 0), 'gold', lines=(22.52,))
    # gold chains swagged over the hips
    rings = [(y, sk_h(y, 0.2), (0.0, SKZ)) for y in (22.1, 21.3, 20.5, 19.9)]
    B.loft('hips', SK, rings, paint.chains, afn=paint.chains_alpha, dens=40)


def jewellery(B):
    # choker with a pendant stone
    nk = np.array([(31.3, 1.0, 0.92, 0.25), (31.9, 0.76, 0.76, 0.17), (32.9, 0.7, 0.72, 0.06), (33.75, 0.72, 0.76, 0.0)])
    nring = lambda y, g: (y, support(K8, np.interp(y, nk[:, 0], nk[:, 1]) + g, np.interp(y, nk[:, 0], nk[:, 2]) + g), (0.0, float(np.interp(y, nk[:, 0], nk[:, 3]))))
    band_loft(B, 'neck', K8, nring, 32.18, 32.92, 0.1, 'choker', dens=40)
    zf = float(np.interp(32.2, nk[:, 0], nk[:, 3] - nk[:, 2])) - 0.1
    B.box('neck', (0, 32.2, zf - 0.03), (0.3, 0.3, 0.1), (0, 0, 45), 'gold')
    B.box('neck', (0, 31.86, zf - 0.02), (0.26, 0.42, 0.1), (-12, 0, 0), 'gem', dens=72)
    # brooch between the cups
    B.box('chest', (0, 28.14, -2.2), (0.56, 0.56, 0.16), (0, 0, 45), 'gold')
    B.box('chest', (0, 28.14, -2.3), (0.3, 0.3, 0.1), (0, 0, 45), 'gem', dens=72)
    # arm rings and bangles (right, mirrored)
    up = np.array([(29.3, 0.66, 0.72, 3.51, 0.19), (28.3, 0.6, 0.66, 3.6, 0.22), (26.8, 0.54, 0.59, 3.72, 0.29)])
    fo = np.array([(22.4, 0.44, 0.48, 4.04, 0.18), (21.3, 0.37, 0.42, 4.1, 0.1), (20.45, 0.31, 0.42, 4.15, 0.05)])

    def limb(T):
        T = T[np.argsort(T[:, 0])]
        f = lambda y, k: float(np.interp(y, T[:, 0], T[:, k]))
        return lambda y, g: (y, support(K8, f(y, 1) + g, f(y, 2) + g), (f(y, 3), f(y, 4)))

    made = band_loft(B, 'arm_r', K8, limb(up), 28.12, 28.78, 0.09, 'gold', inlay=(28.38, 28.52), dens=40)
    made.append(B.box('arm_r', (4.31, 28.45, 0.2), (0.12, 0.34, 0.34), (45, 0, 0), 'gem', dens=64))
    made += band_loft(B, 'forearm_r', K8, limb(fo), 21.5, 21.74, 0.06, 'gold', bevel=0.035, dens=40)
    made += band_loft(B, 'forearm_r', K8, limb(fo), 21.06, 21.38, 0.07, 'gold', bevel=0.035, lines=(21.22,), dens=40)
    B.mirror(made)
    # garter on the left thigh, the one the slit shows
    th = np.array([(15.4, 1.21, 1.22, 1.34, 1.63, -0.06), (17.4, 1.4, 1.32, 1.56, 1.69, 0.0), (19.2, 1.57, 1.45, 1.8, 1.75, 0.06)])
    f = lambda y, k: float(np.interp(y, th[:, 0], th[:, k]))
    gring = lambda y, g: (y, support(K8, f(y, 1) + g, f(y, 2) + g, f(y, 3) + g, 2.15), (-f(y, 4), f(y, 5)))
    band_loft(B, 'leg_l', K8, gring, 16.9, 17.34, 0.08, 'gold', inlay=(17.07, 17.17), dens=40)
    B.box('leg_l', (-2.15, 16.74, -1.36), (0.2, 0.34, 0.08), (0, 20, 0), 'gem', dens=64)


def build(B):
    skirt(B)
    sash(B)
    belt(B)
    jewellery(B)
