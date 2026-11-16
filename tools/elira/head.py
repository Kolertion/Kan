"""Head: skull and face, nose, ears, earrings and the jewel on the forehead."""
import numpy as np
from kit import dir_ex, dir_n
import paint

K8 = [45 * k for k in range(8)]
# height, half width, front z, back z, flat half width in front, flat half width behind
HEAD = [
    (33.18, 0.34, -2.14, -1.45, 0.24, 0.2), (33.4, 0.8, -2.26, -1.05, 0.5, 0.45), (33.72, 1.3, -2.34, -0.5, 0.86, 0.75),
    (34.3, 1.8, -2.34, 0.55, 1.24, 1.0), (34.95, 2.1, -2.34, 1.75, 1.52, 1.2), (36.55, 2.2, -2.34, 2.34, 1.62, 1.4),
    (37.2, 2.1, -2.2, 2.3, 1.45, 1.3), (37.75, 1.72, -1.82, 1.95, 1.05, 1.0), (38.05, 1.05, -1.1, 1.25, 0.55, 0.55)]


def head_ring(y, rx, zf, zb, af, ab):
    """Octagon with a wide flat front, so both eyes sit on one plane."""
    hd, d = (zb - zf) / 2, 0.70710678
    return (y, np.array([hd, d * (af + hd), rx, d * (ab + hd), hd, d * (ab + hd), rx, d * (af + hd)]), (0.0, (zf + zb) / 2))


def face_z(y):
    """Depth of the flat front of the face at height y."""
    H = np.array(HEAD)
    return float(np.interp(y, H[:, 0], H[:, 2]))


def build(B):
    def mpf(i, k):
        if HEAD[i][0] < 37.3 and k in (0, 1, 7):
            return dict(dens=96 if k == 0 else 60)
        return {}

    B.loft('head', K8, [head_ring(*h) for h in HEAD], 'skin', reg='head', mpf=mpf, cap0=True, cap1=True)
    # nose: a small ridge that gives the profile a tip
    top, tip = (0.0, 35.1, -2.345), (0.0, 34.68, -2.47)
    br, bl, bc = (0.19, 34.62, -2.33), (-0.19, 34.62, -2.33), (0.0, 34.585, -2.335)
    for pts, out in (([top, tip, br], (1, 0.2, -1)), ([top, bl, tip], (-1, 0.2, -1)), ([tip, br, bc], (0.3, -1, -0.4)), ([tip, bc, bl], (-0.3, -1, -0.4))):
        B.poly('head', pts, 'skin', out, reg='nose', dens=96)
    for s in (1, -1):
        B.box('head', (2.26 * s, 35.25, 0.12), (0.2, 0.95, 0.62), (0, -12 * s, -6 * s), 'skin', reg='ear')
        # earring: a little plate turned towards the front so it reads from both sides
        c = np.array([2.36 * s, 34.32, -0.02])
        ex, up = dir_ex(45 * s), np.array([0.0, 1.0, 0.0])
        pts = [c - ex * 0.19 + up * 0.56, c + ex * 0.19 + up * 0.56, c + ex * 0.19 - up * 0.56, c - ex * 0.19 - up * 0.56]
        B.poly('head', pts, paint.earring, dir_n(45 * s), ex, two=True, afn=paint.earring_alpha, dens=72, back=1.0)
    # forehead jewel: fine chains from the parting with an amethyst drop
    y0, y1 = 36.4, 37.16
    fz = lambda y: -2.34 + 0.2154 * (y - 36.55) - 0.045       # the forehead plane, a hair in front
    z0, z1 = fz(y0), fz(y1)
    pts = [(1.46, y1, z1), (-1.46, y1, z1), (-1.46, y0, z0), (1.46, y0, z0)]
    B.poly('head', pts, paint.tikka, (0, 0.2, -1), (-1, 0, 0), afn=paint.tikka_alpha, dens=96)
    B.box('head', (0.0, 36.72, fz(36.72) - 0.05), (0.16, 0.24, 0.08), (11, 0, 0), 'gem', dens=96)
