"""Torso, bust, neck, arms, hands, legs and feet. Everything hangs on the original joints."""
import numpy as np
from kit import support, frame_xf, norm

K8 = [45 * k for k in range(8)]
K12 = [30 * k for k in range(12)]
CZ = 0.1
# height, half width, front z, back z, squareness
TORSO = [
    (19.5, 0.9, -0.9, 1.3, 2.0), (20.1, 2.9, -1.45, 1.95, 2.2), (20.8, 3.42, -1.62, 2.3, 2.3), (21.5, 3.45, -1.7, 2.32, 2.3),
    (22.2, 3.32, -1.72, 2.15, 2.3), (22.8, 3.1, -1.69, 1.95, 2.2), (23.4, 2.84, -1.63, 1.7, 2.2), (24.0, 2.58, -1.55, 1.5, 2.1),
    (24.6, 2.36, -1.46, 1.38, 2.0), (25.2, 2.24, -1.4, 1.31, 2.0), (25.7, 2.2, -1.38, 1.3, 2.0), (26.4, 2.32, -1.41, 1.38, 2.1),
    (27.0, 2.5, -1.46, 1.48, 2.2), (27.8, 2.72, -1.52, 1.6, 2.3), (28.6, 2.9, -1.55, 1.7, 2.3), (29.4, 3.02, -1.5, 1.76, 2.3),
    (30.1, 3.12, -1.38, 1.74, 2.3), (30.7, 3.2, -1.2, 1.66, 2.3), (31.12, 3.08, -0.98, 1.52, 2.2), (31.48, 2.3, -0.8, 1.3, 2.0),
    (31.85, 1.15, -0.6, 1.05, 2.0)]


def tor(y):
    """Torso cross-section at height y: half width, front z, back z, squareness."""
    T = np.array(TORSO)
    return [float(np.interp(y, T[:, 0], T[:, k])) for k in (1, 2, 3, 4)]


def tor_ring(y, grow=0.0, dirs=K12):
    rx, zf, zb, p = tor(y)
    return (y, support(dirs, rx + grow, CZ - zf + grow, zb - CZ + grow, p), (0.0, CZ))


def ball(B, bone, c, r, mat='skin', dirs=K8, **mp):
    """Small faceted sphere that fills a joint when it bends."""
    rings = [(c[1] + r * f, support(dirs, max(r * g, 0.03), max(r * g, 0.03)), (c[0], c[2]))
             for f, g in ((0.98, 0.1), (0.5, 0.87), (-0.5, 0.87), (-0.98, 0.1))]
    return B.loft(bone, dirs, rings, mat, **mp)


def torso(B):
    ys = [t[0] for t in TORSO]

    def bone(i, k):
        ym = (ys[i] + ys[i + 1]) / 2
        return 'hips' if ym < 23.4 else ('waist' if ym < 26.4 else 'chest')

    B.loft(bone, K12, [tor_ring(y) for y in ys], 'skin', reg='torso', cap0=True, cap1=True)
    for yb, bn in ((23.4, 'hips'), (26.4, 'waist')):       # inner sleeves keep the waist closed when it bends
        B.loft(bn, K12, [tor_ring(yb - 0.5, -0.05), tor_ring(yb + 0.5, -0.05)], 'skin', reg='torso')


def bust(B):
    xf = frame_xf((1.3, 28.5, -1.12), norm((0.17, -0.1, -1.0)), front=(0, 1, 0))
    prof = [(0.0, 1.42, 0.0), (0.5, 1.4, 0.0), (0.95, 1.22, 0.05), (1.3, 0.93, 0.1), (1.55, 0.56, 0.14), (1.7, 0.18, 0.16)]
    rings = [(t, support(K12, r * 0.97, r * 1.02, r), (0.0, sag)) for t, r, sag in prof]
    return B.loft('chest', K12, rings, 'skin', xf=xf, reg='bust', cap1=True)


def neck(B):
    prof = [(31.3, 1.0, 0.92, 0.25), (31.9, 0.76, 0.76, 0.17), (32.9, 0.7, 0.72, 0.06), (33.75, 0.72, 0.76, 0.0)]
    B.loft('neck', K8, [(y, support(K8, rx, rz), (0.0, cz)) for y, rx, rz, cz in prof], 'skin', reg='neck')


def arm(B):
    up = [(31.32, 0.1, 0.1, 3.1, 0.15), (31.26, 0.3, 0.32, 3.16, 0.15), (31.1, 0.48, 0.52, 3.26, 0.15), (30.84, 0.62, 0.68, 3.36, 0.15),
          (30.46, 0.71, 0.77, 3.43, 0.16), (29.95, 0.72, 0.78, 3.47, 0.17), (29.3, 0.66, 0.72, 3.51, 0.19), (28.3, 0.6, 0.66, 3.6, 0.22), (26.8, 0.54, 0.59, 3.72, 0.29),
          (25.75, 0.51, 0.55, 3.82, 0.34), (25.2, 0.51, 0.54, 3.86, 0.36)]
    fore = [(25.55, 0.5, 0.54, 3.84, 0.35), (24.7, 0.55, 0.6, 3.89, 0.33), (23.6, 0.52, 0.56, 3.96, 0.27), (22.4, 0.44, 0.48, 4.04, 0.18),
            (21.3, 0.37, 0.42, 4.1, 0.1), (20.45, 0.31, 0.42, 4.15, 0.05)]
    ring = lambda t: (t[0], support(K8, t[1], t[2]), (t[3], t[4]))
    made = B.loft('arm_r', K8, [ring(t) for t in up], 'skin', reg='arm', cap0=True, cap1=True)
    made += B.loft('forearm_r', K8, [ring(t) for t in fore], 'skin', reg='arm', cap0=True, cap1=True)
    made += ball(B, 'forearm_r', (3.85, 25.35, 0.35), 0.52, reg='arm')
    return made


def hand(B):
    P = B.skel.pivot
    palm = [(20.55, 0.3, 0.42, 4.15, 0.05), (20.1, 0.27, 0.62, 4.22, 0.14), (19.6, 0.25, 0.74, 4.32, 0.24), (19.22, 0.22, 0.72, 4.37, 0.26)]
    made = B.loft('hand_r', K8, [(y, support(K8, rx, rz, p=2.6), (cx, cz)) for y, rx, rz, cx, cz in palm], 'skin', reg='hand', cap0=True, cap1=True)
    made += ball(B, 'hand_r', (4.15, 20.42, 0.06), 0.36, reg='hand')
    X = np.array([1.0, 0.0, 0.0])
    for f, (w, d, tip) in {1: (0.32, 0.36, 0.7), 2: (0.33, 0.37, 0.76), 3: (0.32, 0.35, 0.7), 4: (0.29, 0.31, 0.58)}.items():
        p = [P(f'finger{f}_{j}_r') for j in (1, 2, 3)]
        p.append(p[2] + tip * norm(p[2] - p[1]))
        for j in range(3):
            made.append(B.bar(f'finger{f}_{j + 1}_r', p[j], p[j + 1], w - 0.02 * j, d - 0.025 * j, 'skin', side=X, ext=0.05, reg='hand'))
        dr = norm(p[3] - p[2])
        ox = norm(X - dr * (X @ dr)) * (w / 2 - 0.02)
        made.append(B.bar(f'finger{f}_3_r', p[2] + dr * 0.26 + ox, p[3] + dr * 0.02 + ox, 0.06, d - 0.13, 'nail', side=X))
    t1, t2 = P('thumb_1_r'), P('thumb_2_r')
    t3 = t2 + 0.84 * norm(norm(t2 - t1) + np.array([0.0, -0.25, 0.05]))
    side = norm(np.array([0.75, 0.0, -0.65]))
    made.append(B.bar('thumb_1_r', t1 + np.array([0.05, 0.1, 0.1]), t2, 0.4, 0.44, 'skin', side=side, ext=0.06, reg='hand'))
    made.append(B.bar('thumb_2_r', t2, t3, 0.34, 0.36, 'skin', side=side, ext=0.05, reg='hand'))
    dr = norm(t3 - t2)
    ox = norm(side - dr * (side @ dr)) * 0.15
    made.append(B.bar('thumb_2_r', t2 + dr * 0.34 + ox, t3 + dr * 0.02 + ox, 0.06, 0.25, 'nail', side=side))
    return made


def leg(B):
    th = [(21.35, 1.45, 1.4, 1.7, 1.74, 0.12), (20.5, 1.62, 1.5, 1.86, 1.78, 0.1), (19.2, 1.57, 1.45, 1.8, 1.75, 0.06),
          (17.4, 1.4, 1.32, 1.56, 1.69, 0.0), (15.4, 1.21, 1.22, 1.34, 1.63, -0.06), (13.5, 1.04, 1.1, 1.15, 1.58, -0.12),
          (12.3, 0.96, 1.04, 1.04, 1.55, -0.15), (11.6, 0.94, 1.0, 1.0, 1.55, -0.15)]
    sh = [(12.2, 0.94, 0.98, 1.0, 1.55, -0.15), (11.3, 0.9, 0.94, 0.98, 1.545, -0.12), (9.9, 0.9, 0.86, 1.12, 1.53, -0.02),
          (8.3, 0.8, 0.76, 0.98, 1.51, 0.08), (6.5, 0.62, 0.64, 0.74, 1.485, 0.16), (5.0, 0.5, 0.54, 0.6, 1.465, 0.22),
          (3.75, 0.46, 0.52, 0.58, 1.45, 0.25)]
    ring = lambda t: (t[0], support(K8, t[1], t[2], t[3], 2.15), (t[4], t[5]))
    made = B.loft('leg_r', K8, [ring(t) for t in th], 'skin', reg='leg', cap0=True, cap1=True, ao=0.6)
    made += B.loft('shin_r', K8, [ring(t) for t in sh], 'skin', reg='leg', cap0=True, cap1=True, ao=0.6)
    made += ball(B, 'shin_r', (1.55, 11.9, -0.15), 0.93, reg='leg', ao=0.6)
    return made


def foot(B):
    """Foot in a high-heeled sandal: the instep runs down to the ball, the heel rides on a stiletto."""
    x = 1.45
    A, Bp = np.array([x, 3.9, 0.12]), np.array([x, 0.74, -2.28])
    ax = norm(A - Bp)
    xf = frame_xf(Bp, ax, front=(0.0, ax[2], -ax[1]))
    prof = [(-0.1, 0.6, 0.2, 0.22), (0.5, 0.6, 0.3, 0.28), (1.4, 0.55, 0.38, 0.36), (2.6, 0.5, 0.47, 0.5), (3.5, 0.47, 0.5, 0.62), (4.15, 0.46, 0.5, 0.56)]
    made = B.loft('foot_r', K8, [(t, support(K8, rx, rf, rb), (0.0, 0.0)) for t, rx, rf, rb in prof], 'skin', xf=xf, reg='foot', cap0=True, cap1=True, ao=0.6)
    made += ball(B, 'foot_r', (x, 3.05, 0.78), 0.5, reg='foot', ao=0.6)                                  # heel
    made.append(B.box('foot_r', (x, 0.5, -2.78), (1.2, 0.4, 1.15), (9, 0, 0), 'skin', reg='foot', ao=0.6))   # toes
    made.append(B.box('foot_r', (x, 0.46, -3.36), (0.96, 0.34, 0.42), (14, 0, 0), 'skin', reg='foot', ao=0.6))
    for i, dx in enumerate((-0.36, -0.12, 0.12, 0.36)):
        made.append(B.box('foot_r', (x + dx, 0.66 - 0.01 * i, -3.3), (0.17, 0.04, 0.2), (14, 0, 0), 'nail'))
    # sandal: sole, arch, stiletto and gilded straps
    made.append(B.box('foot_r', (x, 0.2, -2.8), (1.3, 0.16, 1.9), (0, 0, 0), 'sole'))
    made.append(B.bar('foot_r', (x, 2.48, 1.0), (x, 0.3, -1.9), 1.04, 0.14, 'sole', side=(1, 0, 0), ext=0.05))
    made.append(B.bar('foot_r', (x, 2.5, 1.02), (x, 0.06, 1.2), 0.3, 0.3, 'sole', side=(1, 0, 0)))
    made.append(B.box('foot_r', (x, 0.07, 1.2), (0.36, 0.14, 0.36), (0, 0, 0), 'gold'))
    made.append(B.box('foot_r', (x, 0.76, -2.5), (1.32, 0.16, 0.3), (-20, 0, 0), 'gold', lines=()))
    made.append(B.bar('foot_r', (x, 3.6, -0.42), (x, 0.84, -2.5), 0.26, 0.07, 'gold', side=(1, 0, 0)))
    ank = [(4.62, 0.035), (4.56, 0.1), (4.2, 0.1), (4.14, 0.035)]
    rings = [(y, support(K8, 0.5 + g, 0.55 + g, 0.6 + g), (1.46, 0.23)) for y, g in ank]
    made += B.loft('shin_r', K8, rings, 'gold', lines=(4.38,))
    return made


def build(B):
    torso(B)
    neck(B)
    side = bust(B) + arm(B) + hand(B) + leg(B) + foot(B)
    B.mirror([cu for cu in side if cu is not None])
