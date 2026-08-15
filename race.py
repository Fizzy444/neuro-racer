import numpy as np
from physics import Cars, Params

N_RAYS = 11
RAY_FOV = np.deg2rad(200.0)
RAY_RANGE = 50.0
RAY_WIN = 18
OBS_SIZE = N_RAYS + 4
DT = 1.0 / 30.0
STALL_STEPS = 120
PARAMS = Params()
CAR_L = PARAMS.length
CAR_W = PARAMS.width
RAY_ANGLES = np.linspace(-RAY_FOV / 2, RAY_FOV / 2, N_RAYS)
CORNER_L = np.array([1, 1, -1, -1]) * CAR_L / 2
CORNER_W = np.array([1, -1, 1, -1]) * CAR_W / 2


def ray_cast(pos, d, A, B, max_range):
    e = B - A
    ap = A[:, None, :, :] - pos[:, None, None, :]
    dd = d[:, :, None, :]
    ee = e[:, None, :, :]
    den = dd[..., 0] * ee[..., 1] - dd[..., 1] * ee[..., 0]
    num_t = ap[..., 0] * ee[..., 1] - ap[..., 1] * ee[..., 0]
    num_u = ap[..., 0] * dd[..., 1] - ap[..., 1] * dd[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num_t / den
        u = num_u / den
    ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, max_range)
    return np.minimum(t.min(axis=2), max_range)


class Race:
    def __init__(self, track, n, max_steps):
        self.track = track
        self.n = n
        self.max_steps = max_steps
        self.cars = Cars(n, PARAMS)
        self.cars.reset(track.c[0], track.theta[0])
        self.idx = np.zeros(n, dtype=int)
        self.progress = np.zeros(n)
        self.max_progress = np.zeros(n)
        self.ref_progress = np.zeros(n)
        self.last_gain = np.zeros(n, dtype=int)
        self.alive = np.ones(n, dtype=bool)
        self.death = np.zeros(n, dtype=int)
        self.step_count = 0

    @property
    def done(self):
        return (not self.alive.any()) or self.step_count >= self.max_steps

    def fitness(self):
        dist = self.max_progress * self.track.spacing
        # Use step_count if alive, else the step they died on
        steps = np.where(self.alive, self.step_count, self.death)
        steps = np.maximum(steps, 1) # prevent div by zero
        
        # Calculate average speed in m/s
        avg_speed = dist / (steps * DT)
        
        # Primary reward is distance, secondary reward is speed. 
        # This creates evolutionary pressure to go fast, without prioritizing crashing quickly.
        return dist + avg_speed

    def observe(self):
        n = self.n
        t = self.track
        c = self.cars
        obs = np.zeros((n, OBS_SIZE))
        a = np.where(self.alive)[0]
        if a.size == 0:
            return obs
        pos = np.stack([c.x[a], c.y[a]], 1)
        ang = c.psi[a][:, None] + RAY_ANGLES[None, :]
        d = np.stack([np.cos(ang), np.sin(ang)], -1)
        w = t.window(self.idx[a], RAY_WIN)
        w1 = (w + 1) % t.m
        A = np.concatenate([t.left[w], t.right[w]], 1)
        B = np.concatenate([t.left[w1], t.right[w1]], 1)
        dist = ray_cast(pos, d, A, B, RAY_RANGE)
        obs[a, :N_RAYS] = dist / RAY_RANGE
        obs[a, N_RAYS] = c.vx[a] / 40.0
        obs[a, N_RAYS + 1] = c.vy[a] / 8.0
        obs[a, N_RAYS + 2] = c.w[a] / 2.0
        obs[a, N_RAYS + 3] = c.delta[a] / PARAMS.max_steer
        return obs

    def step(self, steer, throttle):
        self.cars.step(steer, throttle, self.alive, DT)
        self.step_count += 1
        self._update()

    def _update(self):
        t = self.track
        c = self.cars
        n = self.n
        pos = np.stack([c.x, c.y], 1)
        w = t.window(self.idx, 10)
        d2 = ((t.c[w] - pos[:, None, :]) ** 2).sum(-1)
        new_idx = w[np.arange(n), d2.argmin(1)]
        delta = (new_idx - self.idx + t.m // 2) % t.m - t.m // 2
        self.idx = new_idx
        self.progress = self.progress + delta
        self.max_progress = np.maximum(self.max_progress, self.progress)
        gain = self.progress >= self.ref_progress + 3
        self.ref_progress = np.where(gain, self.progress, self.ref_progress)
        self.last_gain = np.where(gain, self.step_count, self.last_gain)
        cs, sn = np.cos(c.psi)[:, None], np.sin(c.psi)[:, None]
        cx = c.x[:, None] + CORNER_L[None, :] * cs - CORNER_W[None, :] * sn
        cy = c.y[:, None] + CORNER_L[None, :] * sn + CORNER_W[None, :] * cs
        nrm = t.normal[new_idx]
        cp = t.c[new_idx]
        lat = (cx - cp[:, 0:1]) * nrm[:, 0:1] + (cy - cp[:, 1:2]) * nrm[:, 1:2]
        crashed = np.abs(lat).max(1) > t.width / 2
        stalled = (self.step_count - self.last_gain) > STALL_STEPS
        wrong = self.progress < self.max_progress - 10
        dead = self.alive & (crashed | stalled | wrong)
        self.death = np.where(dead, self.step_count, self.death)
        self.alive = self.alive & ~dead
