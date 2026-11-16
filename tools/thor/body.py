"""Torso, arms, hands, pelvis and armoured legs."""
from rig import ARM_X, FINGERS, FINGER_Z, HAND_Y, LEG_X, SOLE

SK = dict(mat='skin')


def half_slab(B, bone, y0, y1, hw, zf, zb, **kw):
    """Half of a chamfered torso slice from the centre line out to +hw."""
    h, d, zc, yc = y1 - y0 + 0.12, zb - zf, (zf + zb) / 2, (y0 + y1) / 2
    B.box(bone, (hw / 2, yc, zc), (hw, h, d * 0.70), **SK, **kw)
    B.box(bone, (hw * 0.90 / 2, yc, zc), (hw * 0.90, h, d * 0.88), **SK, **kw)
    B.box(bone, (hw * 0.76 / 2, yc, zc), (hw * 0.76, h, d), **SK, **kw)


def torso(B):
    """Right half of the torso (mirrored by the caller)."""
    T = dict(region='torso')
    M = dict(round=0.5, **SK, **T)
    # abdomen (waist bone)
    for y0, y1, hw, zf, zb in ((26.8, 27.6, 4.15, -2.35, 2.45), (27.6, 28.4, 4.0, -2.35, 2.40), (28.4, 29.2, 3.88, -2.35, 2.32),
                               (29.2, 30.0, 3.84, -2.36, 2.30), (30.0, 30.8, 3.88, -2.38, 2.32), (30.8, 31.6, 3.98, -2.42, 2.38),
                               (31.6, 32.45, 4.14, -2.46, 2.48), (32.45, 33.3, 4.36, -2.50, 2.60)):
        half_slab(B, 'body', y0, y1, hw, zf, zb, **T)
    # rib cage / chest (chest bone)
    for y0, y1, hw, zf, zb in ((33.3, 34.1, 4.62, -2.56, 2.70), (34.1, 34.9, 4.88, -2.60, 2.80), (34.9, 35.7, 5.14, -2.66, 2.88),
                               (35.7, 36.5, 5.40, -2.70, 2.94), (36.5, 37.3, 5.66, -2.72, 3.00), (37.3, 38.1, 5.90, -2.72, 3.02),
                               (38.1, 38.9, 6.10, -2.66, 3.00), (38.9, 39.6, 6.16, -2.56, 2.96), (39.6, 40.5, 5.5, -2.25, 2.75)):
        half_slab(B, 'body2', y0, y1, hw, zf, zb, **T)
    # abdominal wall: four rows of terraced muscle bellies
    rows = ((34.12, 1.30, 1.62, 'body2'), (32.72, 1.28, 1.58, 'body'), (31.34, 1.26, 1.52, 'body'), (29.72, 1.72, 1.44, 'body'))
    for y, h, w, bone in rows:
        B.dome(bone, (0.10 + w / 2, y, -2.38), w, h, 0.42, (0, -5, 0), layers=1, **M)
    # obliques
    B.dome('body5', (3.1, 31.0, -1.9), 1.7, 3.6, 0.3, (0, -32, -5), layers=1, **M)
    B.dome('body5', (3.92, 30.9, 0.2), 3.4, 3.0, 0.22, (0, 0, -4), axis=(1, 0, 0), layers=1, **M)
    # serratus fingers under the pec
    for x, y in ((4.5, 35.0), (4.34, 34.2), (4.16, 33.42)):
        B.box('body2', (x, y, -2.02), (1.3, 0.5, 0.7), (0, -34, 24), round=0.6, **SK, **T)
    # pectoral: broad terraced slab that overhangs the ribs
    B.rbox('body2', (2.9, 37.42, -2.78), (5.4, 3.6, 0.95), (7, -7, 4), k=0.93, k2=0.9, round=0.55, blob=(5.4, 3.6, 0.95), **SK, **T)
    # collar bone and trapezius slope
    B.box('body2', (3.25, 40.28, -2.18), (5.3, 0.46, 0.6), (0, -10, 6), round=0.55, **SK, **T)
    B.rbox('body2', (3.2, 40.78, 0.75), (4.6, 1.5, 3.3), (0, 0, -16), k=0.80, round=0.3, blob=(4.6, 1.5, 3.3), **SK, **T)
    B.box('body2', (1.5, 41.35, 1.3), (2.0, 1.5, 2.3), (0, 0, -30), round=0.3, **SK, **T)
    # lats: flare from waist to arm pit
    B.dome('body2', (5.5, 36.3, 0.95), 6.0, 3.6, 0.3, (0, 0, -9), axis=(1, 0, 0), layers=1, **M)
    B.dome('body2', (3.9, 34.7, 2.66), 3.4, 4.6, 0.2, (0, 12, -14), axis=(0, 0, 1), layers=1, round=0.3, **SK, **T)
    # upper back: shoulder blade, rear spine columns
    B.dome('body2', (3.0, 38.0, 2.9), 4.7, 3.6, 0.2, (0, 8, -5), axis=(0, 0, 1), layers=1, round=0.3, **SK, **T)
    B.dome('body2', (0.95, 36.5, 2.86), 1.5, 6.4, 0.26, (-2, 0, 0), axis=(0, 0, 1), layers=1, **M)
    B.dome('body', (0.92, 30.4, 2.3), 1.45, 6.4, 0.24, (2, 0, 0), axis=(0, 0, 1), layers=1, **M)


def neck(B):
    T = dict(region='torso')
    B.ring('chest', (0, 41.7, 0.55), 1.42, 1.5, 2.9, rot=(-9, 0, 0), **SK)
    for s in (1, -1):
        B.box('chest', (0.92 * s, 41.55, -0.55), (0.52, 2.5, 0.62), (-24, 0, 15 * s), round=0.5, **SK, **T)
    B.box('chest', (0, 41.75, -1.0), (0.5, 0.55, 0.4), (-10, 0, 0), round=0.6, **SK, **T)       # adam's apple
    B.box('chest', (0, 40.35, -2.0), (1.1, 0.7, 0.5), round=0.5, shade=0.12, **SK, **T)         # sternal notch


def arm(B):
    """Smooth octagonal limb stacks; muscle separation is painted by the skin material."""
    A = dict(region='arm')
    x, z = ARM_X, 0.2
    # deltoid cap
    B.limb('bone14', x, z, [41.0, 40.6, 40.0, 39.2, 38.3, 37.4, 36.6], [0.95, 1.55, 1.98, 2.12, 2.06, 1.92, 1.78],
           [1.15, 1.8, 2.22, 2.36, 2.28, 2.14, 2.0], xoff=[-0.55, -0.32, -0.08, 0.08, 0.14, 0.1, 0.04], **SK, **A)
    B.box('bone4', (x + 0.15, 38.9, z + 2.16), (1.9, 2.4, 0.3), (8, 8, -6), round=0.22, **SK, **A)            # rear deltoid head
    B.box('bone5', (x + 2.06, 38.5, z), (0.3, 2.6, 2.2), (0, 0, 6), round=0.22, **SK, **A)                    # lateral head
    # upper arm: biceps pushes the profile forward, triceps back
    B.limb('bone15', x, z, [37.0, 36.2, 35.4, 34.6, 33.8, 33.0, 32.2, 31.5, 31.0],
           [1.72, 1.80, 1.84, 1.80, 1.70, 1.58, 1.48, 1.43, 1.42], [1.98, 2.16, 2.24, 2.18, 2.02, 1.84, 1.70, 1.60, 1.56],
           zoff=[0.0, -0.04, -0.08, -0.08, -0.04, 0.0, 0.04, 0.06, 0.08], **SK, **A)
    B.box('bone2', (x - 0.1, 35.3, z - 2.22), (1.7, 3.0, 0.3), (2, 0, 0), round=0.22, **SK, **A)             # biceps peak
    # elbow + forearm
    B.limb('bone23', x, 0.3, [32.1, 31.3, 30.5, 29.7, 28.9, 28.1, 27.3, 26.5, 25.7, 25.0, 24.4],
           [1.44, 1.60, 1.74, 1.76, 1.66, 1.50, 1.33, 1.17, 1.04, 0.95, 0.88],
           [1.56, 1.72, 1.86, 1.88, 1.78, 1.63, 1.48, 1.36, 1.27, 1.21, 1.16], **SK, **A)
    B.box('bone30', (x + 0.05, 31.25, 1.72), (1.1, 1.3, 0.7), (14, 0, 0), round=0.6, **SK, **A)             # olecranon
    B.box('bone31', (x + 1.48, 29.9, -0.5), (0.5, 3.0, 1.4), (4, 0, 5), round=0.22, **SK, **A)               # brachioradialis
    B.box('bone32', (x - 1.46, 29.8, 0.2), (0.5, 2.8, 1.6), (0, 0, -5), round=0.22, **SK, **A)                # flexors
    B.box('bone33', (x + 0.5, 29.6, 1.9), (1.5, 3.0, 0.5), (-4, 0, 0), round=0.22, **SK, **A)               # extensors
    # tendon ridges of the lower forearm, one per legacy quadrant bone
    B.box('bone26', (x, 26.9, 1.52), (0.8, 2.4, 0.3), (-5, 0, 0), round=0.25, **SK, **A)
    B.box('bone27', (x + 1.2, 26.9, 0.3), (0.3, 2.4, 1.1), (0, 0, 5), round=0.25, **SK, **A)
    B.box('bone28', (x, 26.9, -0.94), (0.8, 2.4, 0.3), (5, 0, 0), round=0.25, **SK, **A)
    B.box('bone29', (x - 1.2, 26.9, 0.3), (0.3, 2.4, 1.1), (0, 0, -5), round=0.25, **SK, **A)
    # wrist: carpal bumps
    B.box('bone36', (x + 0.55, 24.95, 1.25), (0.6, 0.6, 0.6), round=0.7, **SK, **A)                         # ulna head
    B.box('bone38', (x, 24.75, -0.86), (1.4, 0.7, 0.4), round=0.6, **SK, **A)
    B.box('bone40', (x - 0.76, 24.8, 0.3), (0.34, 0.8, 1.9), round=0.6, **SK, **A)
    B.box('bone42', (x + 0.8, 24.8, 0.1), (0.34, 0.8, 1.7), round=0.6, **SK, **A)


def hand(B):
    H = dict(region='hand')
    x = ARM_X
    hb = 'RightThumb'
    B.rbox(hb, (x + 0.03, 23.2, 0.03), (1.08, 2.6, 2.5), k=0.86, round=0.4, blob=(1.08, 2.6, 2.5), **SK, **H)       # palm
    B.box(hb, (x - 0.42, 23.55, -0.72), (0.75, 1.5, 1.15), round=0.6, **SK, **H)                                       # thenar pad
    B.box(hb, (x - 0.40, 23.1, 0.75), (0.6, 1.9, 0.9), round=0.6, **SK, **H)                                           # hypothenar
    B.box(hb, (x + 0.16, 22.1, 0.03), (1.0, 0.55, 2.42), round=0.5, **SK, **H)                                         # knuckle ridge
    lens = {'Index': (1.22, 0.86, 0.66), 'Middle': (1.34, 0.95, 0.70), 'Ring': (1.24, 0.88, 0.66), 'Pinky': (1.0, 0.70, 0.58)}
    wid = {'Index': 0.56, 'Middle': 0.58, 'Ring': 0.55, 'Pinky': 0.49}
    for f in FINGERS:
        z = FINGER_Z[f]
        l1, l2, l3 = lens[f]
        w = wid[f]
        fx = x + 0.05
        b1, b2 = f'Right{f}1', f'Right{f}2'
        B.box(b1, (fx, HAND_Y - l1 / 2, z), (w, l1 + 0.12, w - 0.02), round=0.55, **SK, **H)
        B.box(b1, (fx + 0.16, HAND_Y - 0.05, z), (0.42, 0.42, w + 0.02), (0, 0, 45), round=0.6, **SK, **H)            # knuckle
        B.bone(b2, b1, (fx, HAND_Y - l1, z), (0, 0, -20))
        B.box(b2, (fx, HAND_Y - l1 - l2 / 2, z), (w - 0.05, l2 + 0.1, w - 0.06), round=0.55, **SK, **H)
        y3 = HAND_Y - l1 - l2
        B.box(b2, (fx, y3 - l3 / 2, z), (w - 0.09, l3 + 0.06, w - 0.10), (0, 0, -16), origin=(fx, y3, z), round=0.6, **SK, **H)
        B.box(b2, (fx + 0.23, y3 - l3 * 0.60, z), (0.08, l3 * 0.55, w - 0.22), (0, 0, -16), origin=(fx, y3, z), mat='nail')
    # thumb
    tb, tt = 'RightThumbBase', 'RightThumbTip'
    px, py, pz = x - 0.2, 23.75, -0.95
    B.box(tb, (px, py - 0.62, pz), (0.74, 1.45, 0.8), round=0.55, **SK, **H)
    B.bone(tt, tb, (px, py - 1.3, pz), (12, 0, -6))
    B.box(tt, (px, py - 1.3 - 0.52, pz), (0.64, 1.15, 0.68), round=0.6, **SK, **H)
    B.box(tt, (px + 0.02, py - 1.3 - 0.72, pz - 0.33), (0.42, 0.6, 0.08), mat='nail')


def pelvis(B):
    D = dict(mat='dark', base=(64, 34, 28), ao=0.5)
    B.rbox('pelvis_connection', (0, 25.85, 0.15), (7.9, 3.6, 4.7), k=0.82, round=0.3, blob=(7.9, 3.6, 4.7), **D)
    B.box('pelvis_connection', (0, 24.3, 0.1), (3.2, 1.6, 3.6), **D)
    for s in (1, -1):
        B.rbox('pelvis_connection', (1.7 * s, 25.3, 1.5), (3.0, 3.3, 1.9), k=0.8, round=0.35, blob=(3.0, 3.3, 1.9), **D)


def leg(B):
    AR = dict(mat='armor', ao=0.5)          # legs sit inside the skirt: keep their baked shadow light
    LE = dict(mat='leather', ao=0.5)
    PL = dict(round=0.4, rim=1, rimw=0.18, **AR)
    x = LEG_X
    hip, shin, foot = 'hip_right', 'rightLeg', 'RightFoot'
    # thigh
    B.limb(hip, x - 0.15, 0.1, [27.2, 26.2, 25.2, 24.2, 23.2, 22.2, 21.2, 20.2, 19.2, 18.2, 17.3, 16.5],
           [1.60, 1.90, 2.06, 2.14, 2.16, 2.12, 2.05, 1.98, 1.92, 1.86, 1.80, 1.76],     # top tucks inside the skirt
           [2.36, 2.46, 2.50, 2.48, 2.42, 2.34, 2.26, 2.18, 2.10, 2.02, 1.94, 1.88],
           zoff=[0.15, 0.1, 0.05, 0.0, -0.04, -0.07, -0.08, -0.08, -0.06, -0.04, -0.02, 0.0],
           xoff=[0, 0, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.13, 0.14, 0.15], **AR)
    B.dome(hip, (x, 19.3, -2.12), 2.6, 4.8, 0.3, (2, 0, 0), layers=1, **PL)                              # quad plate
    B.dome(hip, (x, 20.9, 2.25), 2.5, 6.0, 0.22, (-3, 0, 0), axis=(0, 0, 1), layers=1, round=0.4, **AR)   # hamstring
    B.ring(hip, (x - 0.07, 22.2, -0.07), 2.22, 2.44, 0.8, bands=2, **LE)                                              # thigh strap
    B.ring(hip, (x - 0.07, 22.72, -0.07), 2.27, 2.49, 0.28, mat='gold', round=0.3)
    B.box(hip, (x + 2.2, 22.2, -0.1), (0.3, 1.0, 0.9), mat='gold', round=0.5, rim=1, rimw=0.12)                            # strap buckle
    # knee
    B.ring(shin, (x, 15.9, 0.0), 1.86, 1.96, 1.9, **AR)
    B.rbox(shin, (x, 15.95, -1.72), (2.75, 2.7, 1.45), (4, 0, 0), k=0.78, round=0.45, blob=(2.75, 2.7, 1.45), rim=1, rimw=0.18, **AR)   # knee cop
    B.box(shin, (x, 17.45, -1.95), (1.7, 1.7, 0.42), (6, 0, 45), mat='gold', round=0.4, rim=1, rimw=0.14)             # upper lame point
    B.box(shin, (x, 14.5, -1.95), (1.35, 1.35, 0.42), (-2, 0, 45), mat='gold', round=0.4, rim=1, rimw=0.12)
    for s in (1, -1):
        B.box(shin, (x + 1.72 * s, 15.9, -0.2), (0.5, 1.5, 1.5), (45, 0, 0), mat='gold', round=0.5)                   # side rondels
    # shin and calf
    B.limb(shin, x, 0.05, [15.3, 14.4, 13.5, 12.6, 11.7, 10.8, 9.9, 9.0, 8.1, 7.2, 6.3, 5.5],
           [1.78, 1.84, 1.86, 1.82, 1.74, 1.62, 1.50, 1.38, 1.28, 1.21, 1.19, 1.22],
           [1.86, 1.98, 2.06, 2.06, 1.98, 1.84, 1.68, 1.54, 1.42, 1.34, 1.32, 1.36],
           zoff=[0.0, 0.14, 0.26, 0.32, 0.32, 0.28, 0.22, 0.18, 0.14, 0.1, 0.1, 0.1], **AR)
    B.dome(shin, (x, 12.6, 2.2), 2.4, 4.4, 0.26, (-4, 0, 0), axis=(0, 0, 1), layers=1, round=0.42, **AR)  # calf
    B.dome(shin, (x, 10.6, -1.66), 2.1, 7.6, 0.34, (-2.5, 0, 0), layers=1, **PL)                         # greave
    B.box(shin, (x, 10.6, -2.0), (0.36, 7.0, 0.3), (-2.5, 0, 0), mat='gold', round=0.5)                               # greave ridge
    B.ring(shin, (x, 13.75, 0.28), 1.98, 2.18, 0.62, bands=1, **LE)                                                   # garter strap
    B.box(shin, (x + 0.9, 13.75, -1.95), (0.7, 0.8, 0.3), mat='gold', round=0.5, rim=1, rimw=0.1)
    B.ring(shin, (x, 7.05, 0.12), 1.36, 1.50, 0.55, bands=1, **LE)                                                    # ankle strap
    B.ring(shin, (x, 6.45, 0.12), 1.42, 1.56, 0.34, mat='gold', round=0.3)
    # boot
    B.ring(foot, (x, 5.5, 0.3), 1.30, 1.45, 1.3, **AR)
    B.rbox(foot, (x, 4.42, 0.85), (2.42, 2.3, 2.5), k=0.84, round=0.35, blob=(2.42, 2.3, 2.5), **AR)                  # heel
    B.rbox(foot, (x, 4.25, -1.25), (2.5, 2.0, 2.7), k=0.86, round=0.3, blob=(2.5, 2.0, 2.7), **AR)                    # mid foot
    B.box(foot, (x, 5.2, -1.3), (1.9, 0.5, 2.6), (-24, 0, 0), mat='gold', round=0.4, rim=1, rimw=0.14)                 # instep plate
    B.rbox(foot, (x, 4.02, -3.25), (2.3, 1.55, 2.2), k=0.84, round=0.35, blob=(2.3, 1.55, 2.2), **AR)                 # toe box
    B.rbox(foot, (x, 3.98, -4.65), (1.6, 1.2, 1.6), (12, 0, 0), k=0.8, round=0.4, blob=(1.6, 1.2, 1.6), **AR)
    B.box(foot, (x, 4.25, -5.55), (0.95, 0.8, 1.1), (30, 0, 0), round=0.5, **AR)                                       # curled tip
    B.box(foot, (x, 4.72, -6.05), (0.55, 0.5, 0.6), (55, 0, 0), mat='gold', round=0.5)
    B.box(foot, (x, 3.9, -3.3), (2.36, 0.3, 0.5), mat='gold', round=0.3)                                               # toe band
    B.box(foot, (x, SOLE + 0.19, -1.35), (2.56, 0.38, 6.3), mat='leather', tone=0.8)                                   # sole
    B.box(foot, (x, SOLE + 0.3, 1.3), (2.3, 0.6, 1.5), mat='leather', tone=0.8)                                        # heel block
