import numpy as np


class Brains:
    def __init__(self, n, sizes, rng=None, init=True):
        self.n = n
        self.sizes = [int(s) for s in sizes]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = []
        self.B = []
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            lim = np.sqrt(6.0 / (i + o))
            if init:
                self.W.append(self.rng.uniform(-lim, lim, (n, i, o)))
            else:
                self.W.append(np.zeros((n, i, o)))
            self.B.append(np.zeros((n, o)))

    def forward(self, x):
        h = x
        for W, B in zip(self.W, self.B):
            h = np.tanh(np.matmul(h[:, None, :], W)[:, 0, :] + B)
        return h

    def trace(self, i, x):
        acts = [x]
        h = x
        for W, B in zip(self.W, self.B):
            h = np.tanh(h @ W[i] + B[i])
            acts.append(h)
        return acts

    def save(self, path, i):
        data = {"sizes": np.array(self.sizes)}
        for k, (W, B) in enumerate(zip(self.W, self.B)):
            data[f"W{k}"] = W[i]
            data[f"B{k}"] = B[i]
        np.savez(path, **data)

    @staticmethod
    def load(path, n, rng=None):
        data = np.load(path)
        sizes = [int(s) for s in data["sizes"]]
        b = Brains(n, sizes, rng, init=False)
        for k in range(len(sizes) - 1):
            b.W[k][:] = data[f"W{k}"][None]
            b.B[k][:] = data[f"B{k}"][None]
        return b
