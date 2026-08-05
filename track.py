import numpy as np


def catmull_rom_closed(pts, samples=40):
    n = len(pts)
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


class Track:
    def __init__(self, seed=None, width=28.0, spacing=3.0):
        self.seed = seed
        self.width = width
        self.spacing = spacing
        rng = np.random.default_rng(seed)
        while not self._generate(rng):
            pass

    def _generate(self, rng):
        n = int(rng.integers(9, 14))
        step = 2 * np.pi / n
        ang = np.arange(n) * step + rng.uniform(-0.28, 0.28, n) * step
        rad = 200.0 * rng.uniform(0.45, 1.0, n)
        pts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
        dense = catmull_rom_closed(pts, 40)
        closed = np.vstack([dense, dense[:1]])
        seg = np.hypot(*np.diff(closed, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        m = int(total // self.spacing)
        if m < 220 or total > 3200:
            return False
        u = np.linspace(0.0, total, m, endpoint=False)
        c = np.stack([np.interp(u, s, closed[:, 0]), np.interp(u, s, closed[:, 1])], 1)
        tan = np.roll(c, -1, 0) - np.roll(c, 1, 0)
        theta = np.arctan2(tan[:, 1], tan[:, 0])
        dth = (np.roll(theta, -1) - theta + np.pi) % (2 * np.pi) - np.pi
        kappa = dth / self.spacing
        if np.abs(kappa).max() > 1.0 / (self.width * 0.75):
            return False
        dist = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        idx = np.arange(m)
        gap = np.abs(idx[:, None] - idx[None, :])
        gap = np.minimum(gap, m - gap)
        if dist[gap > 25].min() < self.width * 1.5:
            return False
        self.c = c
        self.m = m
        self.theta = theta
        self.kappa = kappa
        self.length = m * self.spacing
        self.normal = np.stack([-np.sin(theta), np.cos(theta)], 1)
        self.left = c + self.normal * self.width / 2
        self.right = c - self.normal * self.width / 2
        return True

    def window(self, idx, half):
        return (idx[:, None] + np.arange(-half, half + 1)[None, :]) % self.m
