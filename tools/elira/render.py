"""Software renderer for previewing a model: orthographic, z-buffered, alpha-tested, nearest sampling."""
import math
import numpy as np
from PIL import Image


def _basis(yaw, pitch):
    ya, pa = math.radians(yaw), math.radians(pitch)
    right = np.array([-math.cos(ya), 0, -math.sin(ya)])
    fwd0 = np.array([-math.sin(ya), 0, math.cos(ya)])
    fwd = fwd0 * math.cos(pa) - np.array([0, 1, 0]) * math.sin(pa)
    up = -np.cross(fwd, right)
    up = up / np.linalg.norm(up)
    return right, (up if up[1] >= 0 else -up), fwd


def render(quads, tex, yaw=0, pitch=0, size=(700, 1100), center=None, scale=None, bg=(58, 62, 74), light='studio', ss=2):
    """yaw 0 looks at the model's front. light: 'studio', 'mc' (Minecraft's two entity lights) or None."""
    w, h = size[0] * ss, size[1] * ss
    right, up, fwd = _basis(yaw, pitch)
    allp = np.concatenate([q[0] for q in quads])
    center = (allp.min(0) + allp.max(0)) / 2 if center is None else np.array(center, float)
    if scale is None:
        px, py = (allp - center) @ right, (allp - center) @ up
        scale = 0.94 * min(w / (2 * np.abs(px).max() + 1e-6), h / (2 * np.abs(py).max() + 1e-6))
    else:
        scale = scale * ss
    T = np.asarray(tex)
    th, tw = T.shape[:2]
    col = np.zeros((h, w, 3), np.float32)
    col[:] = bg
    zb = np.full((h, w), np.inf, np.float32)
    Lw = right * -0.35 + up * 0.6 - fwd * 0.72
    Lw = Lw / np.linalg.norm(Lw)
    L0 = np.array([0.2, 1.0, -0.7]) / np.linalg.norm([0.2, 1.0, -0.7])
    L1 = np.array([-0.2, 1.0, 0.7]) / np.linalg.norm([-0.2, 1.0, 0.7])
    for wc, uv, _ in quads:
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
        if n @ fwd > 0.0:
            continue
        if light == 'mc':
            shade = min(1.0, (max(0.0, float(n @ L0)) + max(0.0, float(n @ L1))) * 0.6 + 0.4)
        elif light:
            shade = 0.62 + 0.38 * max(0.0, float(n @ Lw)) + 0.08 * float(n @ up)
        else:
            shade = 1.0
        umin, umax = uv[:, 0].min(), uv[:, 0].max()
        vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
        iu0, iv0 = int(math.floor(umin * tw + 1e-6)), int(math.floor(vmin * th + 1e-6))
        iu1, iv1 = max(int(math.ceil(umax * tw - 1e-6)) - 1, iu0), max(int(math.ceil(vmax * th - 1e-6)) - 1, iv0)
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
            ui = np.clip((u * tw).astype(int), iu0, iu1)
            vi = np.clip((v * th).astype(int), iv0, iv1)
            s = T[np.clip(vi, 0, th - 1), np.clip(ui, 0, tw - 1)]
            m &= s[..., 3] > 100
            if not m.any():
                continue
            sub[m] = z[m]
            col[y0:y1, x0:x1][m] = s[..., :3][m] * shade
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    return img.resize(size, Image.LANCZOS) if ss > 1 else img


def sheet(quads, tex, views, size=(420, 760), **kw):
    ims = [render(quads, tex, yaw=y, pitch=p, size=size, **kw) for y, p in views]
    out = Image.new('RGB', (size[0] * len(ims), size[1]))
    for i, im in enumerate(ims):
        out.paste(im, (i * size[0], 0))
    return out
