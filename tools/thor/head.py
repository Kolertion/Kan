"""Head and face. Face parts are baked at a higher texel density."""

FD = 44.0          # px per unit for facial cubes (authored relative to the default 20)
SK = dict(mat='skin')


def build(B):
    B.side = 1
    F = dict(region='face', dens=FD)
    h = 'head'
    # skull
    B.rbox(h, (0, 46.2, 0.55), (4.7, 4.4, 5.6), k=0.84, k2=0.82, round=0.25, blob=(4.7, 4.4, 5.6), dens=30, **SK)
    B.box(h, (0, 46.55, -1.85), (4.1, 3.1, 1.2), round=0.2, **SK, **F)                    # forehead
    B.ring(h, (0, 43.25, 0.8), 1.38, 1.46, 1.5, rot=(-6, 0, 0), **SK)                     # upper neck
    # lower face: cheeks narrowing into the jaw and chin
    f = 'head3'
    for y0, y1, w, zf, zb in ((44.7, 45.75, 4.05, -2.5, 1.4), (43.95, 44.7, 3.7, -2.5, 1.3), (43.35, 43.95, 3.1, -2.48, 0.9),
                              (42.85, 43.35, 2.4, -2.42, 0.2), (42.5, 42.85, 1.6, -2.3, -0.7)):
        B.box(f, (0, (y0 + y1) / 2, (zf + zb) / 2), (w, y1 - y0 + 0.06, zb - zf), **SK, **F)
    for s in (1, -1):
        B.box(f, (1.46 * s, 43.62, -0.95), (0.7, 2.6, 2.86), (0, 0, -29 * s), round=0.2, **SK, **F)       # jaw line
        B.box(h, (2.38 * s, 45.25, 0.45), (0.42, 1.25, 0.85), (0, 0, -8 * s), round=0.5, shade=0.08, **SK, dens=30)  # ear
        B.box(h, (2.5 * s, 45.3, 0.3), (0.2, 0.7, 0.4), mat='skin', shade=0.35, dens=30)
    # brow ridge
    B.box(h, (0, 46.18, -2.32), (3.9, 0.5, 0.5), round=0.3, shade=0.05, **SK, **F)
    # nose (legacy bone LeftEyelidBase)
    n = 'LeftEyelidBase'
    B.box(n, (0, 45.45, -2.72), (0.4, 1.5, 0.6), (13, 0, 0), round=0.5, **SK, **F)
    B.box(n, (0, 44.72, -2.98), (0.5, 0.45, 0.5), round=0.55, **SK, **F)
    for s in (1, -1):
        B.box(n, (0.3 * s, 44.6, -2.76), (0.26, 0.3, 0.4), round=0.6, shade=0.1, **SK, **F)
    B.box(n, (0, 44.44, -2.7), (0.42, 0.1, 0.3), mat='lip', base=(176, 118, 110), dens=FD)   # nostril shadow
    # mouth: thin, level, slightly down-turned
    m = 'thor_mouth'
    B.box(m, (0, 43.95, -2.56), (0.92, 0.13, 0.3), mat='lip', base=(198, 142, 134), dens=FD)
    B.box(m, (0, 43.76, -2.55), (0.70, 0.15, 0.3), mat='lip', base=(210, 158, 148), dens=FD)
    B.box(m, (0, 43.87, -2.53), (1.08, 0.07, 0.3), mat='lash', base=(104, 48, 50), dens=FD)
    for s in (1, -1):
        B.box(m, (0.58 * s, 43.845, -2.51), (0.14, 0.08, 0.2), (0, 0, -20 * s), mat='lash', base=(132, 74, 72), dens=FD)
    # eyes and brows, mirrored
    for side in (1, -1):
        B.side = side
        e, ir = 'RightEyelidBase2', 'new_eye_iris_1'
        B.box(e, (1.02, 45.5, -2.5), (1.18, 0.46, 0.1), mat='eye', dens=FD * 1.5)
        B.box(e, (1.05, 45.72, -2.58), (1.32, 0.15, 0.22), (0, 0, 7), mat='lash', dens=FD)             # upper lash line
        B.box(e, (1.05, 45.87, -2.53), (1.34, 0.2, 0.2), (0, 0, 7), shade=0.38, **SK, **F)             # upper lid
        B.box(e, (1.03, 45.25, -2.545), (1.22, 0.17, 0.16), (0, 0, 2), shade=0.14, **SK, **F)          # lower lid
        B.box(e, (0.43, 45.43, -2.54), (0.26, 0.26, 0.14), (0, 0, 45), shade=0.2, **SK, **F)           # tear duct corner
        B.box(e, (1.72, 45.74, -2.53), (0.44, 0.11, 0.18), (0, 0, 27), mat='lash', dens=FD)            # lash wing
        B.box(ir, (0.99, 45.5, -2.56), (0.38, 0.42, 0.05), mat='iris', dens=FD * 2.0)
        B.box('RightEyelidBase3', (1.12, 46.22, -2.62), (1.42, 0.16, 0.26), (0, 0, 10), mat='brow', dens=FD)
        B.box('RightEyelidBase3', (0.5, 46.06, -2.62), (0.4, 0.16, 0.26), (0, 0, 24), mat='brow', dens=FD)
    B.side = 1
