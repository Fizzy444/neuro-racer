from dataclasses import dataclass
import numpy as np

G = 9.81

@dataclass
class Params:
    mass: float = 1200.0
    inertia: float = 2500.0        # higher → resists spinning / drifting
    a: float = 1.25                # CG to front axle
    b: float = 1.45                # CG to rear axle
    h: float = 0.45                # CG height (weight transfer)
    mu: float = 6.0                # Massive grip (arcade style)
    pac_b: float = 3.0             # Pacejka B — very gradual grip curve
    pac_c: float = 1.1             # Pacejka C — 1.1 means almost no drop-off in grip (no snap oversteer)
    max_drive: float = 12000.0     # Double the low-end torque for faster acceleration
    max_power: float = 220000.0    # Massive power increase (~300hp) for higher top speed
    max_brake: float = 25000.0     # Stronger brakes to handle the higher speeds
    brake_bias: float = 0.62
    drag: float = 0.9
    roll: float = 0.015
    max_steer: float = 0.45        
    steer_rate: float = 1.6        
    yaw_damping: float = 2000.0    # Massive yaw damping to stop spinning
    lat_damping: float = 200.0     # Extreme lateral damping to completely kill drift
    length: float = 4.4
    width: float = 1.9


class Cars:
    def __init__(self, n, params=None):
        self.n = n
        self.p = params or Params()
        self.reset(np.zeros(2), 0.0)

    def reset(self, pos, heading):
        n = self.n
        self.x = np.full(n, float(pos[0]))
        self.y = np.full(n, float(pos[1]))
        self.psi = np.full(n, float(heading))
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.w = np.zeros(n)
        self.delta = np.zeros(n)
        self.ax = np.zeros(n)

    def step(self, steer, throttle, active, dt=1.0 / 30.0, substeps=4):
        p = self.p
        wb = p.a + p.b
        h = dt / substeps
        act = active.astype(float)
        steer = np.clip(steer, -1.0, 1.0)
        thr = np.clip(throttle, 0.0, 1.0)
        brk = np.clip(-throttle, 0.0, 1.0)
        for _ in range(substeps):
            vx, vy, w = self.vx, self.vy, self.w
            lim = p.max_steer / (1.0 + (vx / 30.0) ** 2)
            self.delta = self.delta + np.clip(steer * lim - self.delta, -p.steer_rate * h, p.steer_rate * h)
            d = self.delta
            fzf = np.maximum(p.mass * G * p.b / wb - p.mass * self.ax * p.h / wb, 800.0)
            fzr = np.maximum(p.mass * G * p.a / wb + p.mass * self.ax * p.h / wb, 800.0)
            vxe = np.maximum(vx, 3.0)
            af = np.arctan2(vy + p.a * w, vxe) - d
            ar = np.arctan2(vy - p.b * w, vxe)
            drive = thr * np.minimum(p.max_drive, p.max_power / np.maximum(vx, 1.0))
            brake = brk * p.max_brake * np.tanh(vx * 1.5)
            df = p.mu * fzf
            dr = p.mu * fzr
            fxr = np.clip(drive - (1.0 - p.brake_bias) * brake, -dr, dr)
            fxf = np.clip(-p.brake_bias * brake, -df, df)
            df_eff = np.sqrt(np.maximum(df ** 2 - fxf ** 2, 1.0))
            dr_eff = np.sqrt(np.maximum(dr ** 2 - fxr ** 2, 1.0))
            fyf = -df_eff * np.sin(p.pac_c * np.arctan(p.pac_b * af))
            fyr = -dr_eff * np.sin(p.pac_c * np.arctan(p.pac_b * ar))
            cd, sd = np.cos(d), np.sin(d)
            drag = -p.drag * vx * np.abs(vx)
            roll = -p.roll * p.mass * G * np.tanh(vx)
            fx = fxr + fxf * cd - fyf * sd + drag + roll
            fy = fyr + fxf * sd + fyf * cd
            mz = p.a * (fyf * cd + fxf * sd) - p.b * fyr
            
            # Artificial stabilizers
            mz = mz - p.yaw_damping * w
            fy = fy - p.mass * p.lat_damping * vy  # Kill lateral velocity (drifting)

            ax = fx / p.mass
            self.ax = ax * act
            self.vx = np.maximum(vx + (ax + vy * w) * h, 0.0) * act
            self.vy = (vy + (fy / p.mass - vx * w) * h) * act
            self.w = (w + mz / p.inertia * h) * act
            c, s = np.cos(self.psi), np.sin(self.psi)
            self.x = self.x + (self.vx * c - self.vy * s) * h
            self.y = self.y + (self.vx * s + self.vy * c) * h
            self.psi = self.psi + self.w * h

