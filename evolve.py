import numpy as np
from brain import Brains


class Evolution:
    def __init__(self, rng, elite_frac=0.05, parent_frac=0.25, random_frac=0.04,
                 rate=0.12, sigma=0.3, sigma_min=0.03, sigma_max=0.6):
        self.rng = rng
        self.elite_frac = elite_frac
        self.parent_frac = parent_frac
        self.random_frac = random_frac
        self.rate = rate
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prev_best = -np.inf

    def reset_track(self):
        self.prev_best = -np.inf

    def adapt(self, best):
        if best > self.prev_best * 1.01 + 1.0:
            self.sigma = max(self.sigma_min, self.sigma * 0.94)
        else:
            self.sigma = min(self.sigma_max, self.sigma * 1.10)
        self.prev_best = max(self.prev_best, best)

    def diversify(self, b, keep=1):
        for k in range(len(b.W)):
            mw = self.rng.random(b.W[k].shape) < self.rate
            mb = self.rng.random(b.B[k].shape) < self.rate
            nw = mw * self.rng.normal(0, self.sigma, b.W[k].shape)
            nb = mb * self.rng.normal(0, self.sigma, b.B[k].shape)
            nw[:keep] = 0
            nb[:keep] = 0
            b.W[k] += nw
            b.B[k] += nb

    def next(self, b, fit):
        n = b.n
        order = np.argsort(-fit)
        ne = max(2, int(round(n * self.elite_frac)))
        nr = int(round(n * self.random_frac))
        nc = n - ne - nr
        pcount = max(4, int(round(n * self.parent_frac)))
        pool = order[:pcount]
        prob = np.arange(pcount, 0, -1, dtype=float)
        prob /= prob.sum()
        pa = self.rng.choice(pool, nc, p=prob)
        pb = self.rng.choice(pool, nc, p=prob)
        nbrains = Brains(n, b.sizes, self.rng, init=True)
        elite = order[:ne]
        for k in range(len(b.W)):
            W, B = b.W[k], b.B[k]
            mask = self.rng.random((nc, 1, W.shape[2])) < 0.5
            cw = np.where(mask, W[pa], W[pb])
            cb = np.where(mask[:, 0, :], B[pa], B[pb])
            cw = cw + (self.rng.random(cw.shape) < self.rate) * self.rng.normal(0, self.sigma, cw.shape)
            cb = cb + (self.rng.random(cb.shape) < self.rate) * self.rng.normal(0, self.sigma, cb.shape)
            nbrains.W[k][:ne] = W[elite]
            nbrains.B[k][:ne] = B[elite]
            nbrains.W[k][ne:ne + nc] = cw
            nbrains.B[k][ne:ne + nc] = cb
        return nbrains
