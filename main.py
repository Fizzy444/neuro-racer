import argparse
import os
import sys
import time
import numpy as np

from track import Track
from brain import Brains
from evolve import Evolution
from race import Race, OBS_SIZE, N_RAYS, RAY_ANGLES, RAY_RANGE, CAR_L, CAR_W, DT


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--layers", type=str, default="32,24,24,16,12")
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--track-every", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--headless", type=int, default=0)
    ap.add_argument("--out", type=str, default="models/latest.npz")
    return ap.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        layers = [int(v) for v in args.layers.split(",") if v.strip()]
        self.sizes = [OBS_SIZE] + layers + [2]
        n = 1 if args.demo else args.pop
        self.evo = Evolution(self.rng)
        if args.load:
            self.brains = Brains.load(args.load, n, self.rng)
            self.sizes = self.brains.sizes
            if n > 1:
                self.evo.diversify(self.brains, keep=1)
        else:
            self.brains = Brains(n, self.sizes, self.rng)
        self.gen = 0
        self.track_no = 0
        self.track_gen = 0
        self.best_hist = []
        self.mean_hist = []
        self.last_best = 0.0
        self.last_laps = 0.0
        self.all_time_best = 0.0
        self.max_steps = int(args.time / DT)
        self.new_track()
        self.start_gen()

    def new_track(self):
        self.track = Track(seed=int(self.rng.integers(1 << 31)))
        self.track_no += 1
        self.track_gen = 0
        self.evo.reset_track()

    def start_gen(self):
        self.race = Race(self.track, self.brains.n, self.max_steps)
        self.obs = self.race.observe()

    def step(self):
        out = self.brains.forward(self.obs)
        self.race.step(out[:, 0], out[:, 1])
        self.obs = self.race.observe()
        if self.race.done:
            self.end_gen()

    def end_gen(self):
        fit = self.race.fitness()
        best = float(fit.max())
        champ = int(fit.argmax())
        
        # Reconstruct actual distance and speed for the champion
        champ_dist = self.race.max_progress[champ] * self.track.spacing
        champ_steps = self.race.step_count if self.race.alive[champ] else self.race.death[champ]
        champ_speed_mps = champ_dist / (max(champ_steps, 1) * 1.0/30.0)
        
        laps = champ_dist / self.track.length
        self.last_best = best
        self.last_laps = laps
        self.all_time_best = max(self.all_time_best, best)
        self.best_hist.append(best)
        self.mean_hist.append(float(fit.mean()))
        
        print(f"gen {self.gen:4d} | trk {self.track_no:3d} | fit {best:7.1f} | dist {champ_dist:6.0f}m | "
              f"spd {champ_speed_mps * 3.6:5.0f}km/h | laps {laps:4.2f} | alive {int(self.race.alive.sum()):3d} | sig {self.evo.sigma:.3f}", flush=True)
        if self.args.demo:
            self.new_track()
        else:
            out_dir = os.path.dirname(self.args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.brains.save(self.args.out, champ)
            self.evo.adapt(best)
            self.brains = self.evo.next(self.brains, fit)
            self.gen += 1
            self.track_gen += 1
            if self.track_gen >= self.args.track_every or laps >= 1.0:
                self.new_track()
        self.start_gen()

    def leader(self):
        r = self.race
        alive = np.where(r.alive)[0]
        if alive.size:
            return int(alive[np.argmax(r.progress[alive])])
        return int(np.argmax(r.max_progress))


class Camera:
    def __init__(self, vw, vh):
        self.vw = vw
        self.vh = vh
        self.center = np.zeros(2)
        self.scale = 1.0

    def fit(self, track):
        pts = np.vstack([track.left, track.right])
        mn, mx = pts.min(0), pts.max(0)
        self.center = (mn + mx) / 2
        self.scale = min((self.vw - 60) / (mx[0] - mn[0]), (self.vh - 60) / (mx[1] - mn[1]))

    def follow(self, pos, zoom=5.0):
        self.center = self.center + (pos - self.center) * 0.15
        self.scale = self.scale + (zoom - self.scale) * 0.1

    def tf(self, pts):
        out = np.empty(pts.shape)
        out[..., 0] = (pts[..., 0] - self.center[0]) * self.scale + self.vw / 2
        out[..., 1] = self.vh / 2 - (pts[..., 1] - self.center[1]) * self.scale
        return out


class Viewer:
    def __init__(self, tr):
        import pygame
        self.pg = pygame
        pygame.init()
        self.W, self.H, self.VW = 1440, 860, 1020
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("Neuroevolution Racing")
        names = "dejavusansmono,consolas,menlo,monospace"
        self.font = pygame.font.SysFont(names, 15)
        self.big = pygame.font.SysFont(names, 20, bold=True)
        self.tr = tr
        self.cam = Camera(self.VW, self.H)
        self.follow = False
        self.show_rays = True
        self.show_dead = True
        self.fast = False
        self.paused = False
        self.speeds = [0.5, 1, 2, 4, 8]
        self.speed_i = 1
        self.cached_no = -1
        self.cached_follow = None
        self.track_surf = None
        self.acc = 0.0

    def text(self, s, x, y, color=(215, 220, 230), font=None):
        surf = (font or self.font).render(s, True, color)
        self.screen.blit(surf, (x, y))

    def render_track(self, surf):
        pg = self.pg
        t = self.track
        L = self.cam.tf(t.left)
        R = self.cam.tf(t.right)
        m = t.m
        for i in range(m):
            j = (i + 1) % m
            pg.draw.polygon(surf, (56, 58, 66), [L[i], L[j], R[j], R[i]])
        for i in range(m):
            j = (i + 1) % m
            col = (200, 50, 50) if (i // 2) % 2 == 0 else (235, 235, 235)
            pg.draw.line(surf, col, L[i], L[j], 3)
            pg.draw.line(surf, col, R[i], R[j], 3)
        for i in range(0, m, 4):
            j = (i + 2) % m
            C0 = self.cam.tf(t.c[i])
            C1 = self.cam.tf(t.c[j])
            pg.draw.line(surf, (90, 92, 100), C0, C1, 1)
        pg.draw.line(surf, (255, 255, 255), L[0], R[0], 5)

    def draw_cars(self, lead):
        pg = self.pg
        tr = self.tr
        r = tr.race
        c = r.cars
        cs, sn = np.cos(c.psi)[:, None], np.sin(c.psi)[:, None]
        l = np.array([1, 1, -1, -1]) * CAR_L / 2
        w = np.array([1, -1, -1, 1]) * CAR_W / 2
        l = np.array([1, -1, -1, 1]) * CAR_L / 2
        w = np.array([1, 1, -1, -1]) * CAR_W / 2
        cx = c.x[:, None] + l[None, :] * cs - w[None, :] * sn
        cy = c.y[:, None] + l[None, :] * sn + w[None, :] * cs
        pts = self.cam.tf(np.stack([cx, cy], -1))
        alive = np.where(r.alive)[0]
        if self.show_dead:
            for i in np.where(~r.alive)[0]:
                pg.draw.polygon(self.screen, (95, 62, 62), pts[i])
        if alive.size:
            order = alive[np.argsort(-r.progress[alive])]
            for k, i in enumerate(order):
                f = k / max(1, len(order) - 1)
                col = (int(80 + 0 * f), int(235 - 110 * f), int(120 + 110 * f))
                pg.draw.polygon(self.screen, col, pts[i])
        if r.alive[lead] or alive.size == 0:
            pg.draw.polygon(self.screen, (255, 225, 60), pts[lead])
            pg.draw.polygon(self.screen, (20, 20, 20), pts[lead], 2)
        if self.show_rays and r.alive[lead]:
            dist = tr.obs[lead, :N_RAYS] * RAY_RANGE
            ang = c.psi[lead] + RAY_ANGLES
            o = self.cam.tf(np.array([c.x[lead], c.y[lead]]))
            ends = self.cam.tf(np.stack([c.x[lead] + np.cos(ang) * dist, c.y[lead] + np.sin(ang) * dist], 1))
            for e in ends:
                pg.draw.line(self.screen, (255, 240, 140), o, e, 1)
                pg.draw.circle(self.screen, (255, 120, 80), e, 3)

    def draw_network(self, rect, lead):
        pg = self.pg
        tr = self.tr
        b = tr.brains
        acts = b.trace(lead, tr.obs[lead])
        sizes = b.sizes
        nl = len(sizes)
        xs = np.linspace(rect.x + 30, rect.right - 40, nl)
        pos = []
        for k, s in enumerate(sizes):
            ys = np.linspace(rect.y + 20, rect.bottom - 20, s) if s > 1 else np.array([rect.centery])
            pos.append([(xs[k], y) for y in ys])
        for k in range(nl - 1):
            W = b.W[k][lead]
            flow = W * acts[k][:, None]
            ii, oo = np.nonzero(np.abs(flow) > 0.12)
            for i, o in zip(ii, oo):
                v = flow[i, o]
                s = min(1.0, abs(v) * 1.4)
                col = (int(40 + 215 * s), int(40 + 130 * s), 40) if v > 0 else (40, int(40 + 110 * s), int(40 + 215 * s))
                pg.draw.line(self.screen, col, pos[k][i], pos[k + 1][o], 1)
        for k in range(nl):
            rad = 4 if sizes[k] > 20 else 5
            for i, p in enumerate(pos[k]):
                a = max(-1.0, min(1.0, float(acts[k][i])))
                col = (int(60 + 195 * max(a, 0)), int(60 + 100 * abs(a)), int(60 + 195 * max(-a, 0)))
                pg.draw.circle(self.screen, col, p, rad)
        self.text("steer", pos[-1][0][0] + 10, pos[-1][0][1] - 8, (200, 200, 210))
        self.text("gas/brk", pos[-1][1][0] + 10, pos[-1][1][1] - 8, (200, 200, 210))
        self.text("sensors", rect.x + 4, rect.bottom - 2, (130, 135, 150))
        self.text(f"{len(sizes) - 2} hidden layers", rect.x + 150, rect.bottom - 2, (130, 135, 150))

    def draw_graph(self, rect):
        pg = self.pg
        tr = self.tr
        pg.draw.rect(self.screen, (28, 31, 40), rect)
        pg.draw.rect(self.screen, (60, 64, 78), rect, 1)
        best = tr.best_hist[-120:]
        mean = tr.mean_hist[-120:]
        if len(best) < 2:
            self.text("fitness / generation", rect.x + 8, rect.y + 6, (130, 135, 150))
            return
        top = max(best) * 1.1 + 1
        for series, col in ((mean, (90, 150, 255)), (best, (110, 235, 140))):
            pts = [(rect.x + 6 + i * (rect.w - 12) / (len(series) - 1), rect.bottom - 6 - v / top * (rect.h - 24)) for i, v in enumerate(series)]
            pg.draw.lines(self.screen, col, False, pts, 2)
        self.text("best", rect.x + 8, rect.y + 4, (110, 235, 140))
        self.text("mean", rect.x + 55, rect.y + 4, (90, 150, 255))
        self.text("distance (m)", rect.right - 120, rect.y + 4, (130, 135, 150))

    def draw(self):
        pg = self.pg
        tr = self.tr
        r = tr.race
        lead = tr.leader()
        self.screen.fill((24, 34, 28), (0, 0, self.VW, self.H))
        self.track = tr.track
        if self.follow:
            self.cam.follow(np.array([r.cars.x[lead], r.cars.y[lead]]))
            self.render_track(self.screen)
        else:
            if self.cached_no != tr.track_no or self.cached_follow:
                self.cam.fit(tr.track)
                self.track_surf = pg.Surface((self.VW, self.H))
                self.track_surf.fill((24, 34, 28))
                self.render_track(self.track_surf)
                self.cached_no = tr.track_no
            self.screen.blit(self.track_surf, (0, 0))
        self.cached_follow = self.follow
        self.draw_cars(lead)
        pg.draw.rect(self.screen, (18, 20, 26), (self.VW, 0, self.W - self.VW, self.H))
        x = self.VW + 16
        self.text("NEUROEVOLUTION RACING", x, 12, (255, 225, 60), self.big)
        alive = int(r.alive.sum())
        secs = r.step_count * DT
        mode = "demo" if tr.args.demo else "training"
        rows = [
            f"mode        {mode}",
            f"generation  {tr.gen}",
            f"track       #{tr.track_no}  (gen {tr.track_gen + 1}/{tr.args.track_every})",
            f"alive       {alive} / {r.n}",
            f"time        {secs:5.1f} / {tr.args.time:.0f} s",
            f"leader      {r.fitness()[lead]:7.1f} m  ({r.fitness()[lead] / tr.track.length:.2f} laps)",
            f"speed       {r.cars.vx[lead] * 3.6:5.0f} km/h",
            f"last gen    {tr.last_best:7.1f} m  ({tr.last_laps:.2f} laps)",
            f"all-time    {tr.all_time_best:7.1f} m",
            f"mutation    {tr.evo.sigma:.3f}",
            f"sim speed   {'FAST' if self.fast else str(self.speeds[self.speed_i]) + 'x'}{'  [PAUSED]' if self.paused else ''}",
        ]
        for i, s in enumerate(rows):
            self.text(s, x, 48 + i * 21)
        self.draw_network(pg.Rect(x, 310, self.W - self.VW - 40, 300), lead)
        self.draw_graph(pg.Rect(x, 650, self.W - self.VW - 32, max(50, self.H - 730)))
        keys = "SPACE pause  F fast  +/- speed  C camera  R rays  D dead  N track  S save  Q quit"
        self.text(keys[:44], x, self.H - 68, (120, 125, 140))
        self.text(keys[44:], x, self.H - 48, (120, 125, 140))
        pg.display.flip()

    def handle(self, ev):
        pg = self.pg
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.VIDEORESIZE:
            self.W, self.H = ev.w, ev.h
            self.VW = max(400, self.W - 420)
            self.screen = pg.display.set_mode((self.W, self.H), pg.RESIZABLE)
            self.cam.vw = self.VW
            self.cam.vh = self.H
            self.cached_no = -1
        if ev.type == pg.KEYDOWN:
            k = ev.key
            if k in (pg.K_ESCAPE, pg.K_q):
                return False
            if k == pg.K_SPACE:
                self.paused = not self.paused
            elif k == pg.K_f:
                self.fast = not self.fast
            elif k == pg.K_c:
                self.follow = not self.follow
            elif k == pg.K_r:
                self.show_rays = not self.show_rays
            elif k == pg.K_d:
                self.show_dead = not self.show_dead
            elif k == pg.K_n:
                self.tr.new_track()
                self.tr.start_gen()
            elif k == pg.K_s:
                os.makedirs("models", exist_ok=True)
                self.tr.brains.save("models/snapshot.npz", self.tr.leader())
            elif k in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                self.speed_i = min(len(self.speeds) - 1, self.speed_i + 1)
            elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                self.speed_i = max(0, self.speed_i - 1)
        return True

    def run(self):
        clock = self.pg.time.Clock()
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            for ev in self.pg.event.get():
                running = self.handle(ev) and running
            if not self.paused:
                if self.fast:
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 0.04:
                        self.tr.step()
                else:
                    self.acc += dt * self.speeds[self.speed_i] / DT
                    n = min(int(self.acc), 40)
                    self.acc = 0.0 if self.acc > 40 else self.acc - n
                    for _ in range(n):
                        self.tr.step()
            self.draw()
        self.pg.quit()


def main():
    args = parse()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        tr = Trainer(args)
        while tr.gen < args.headless:
            tr.step()
        return
    Viewer(Trainer(args)).run()


if __name__ == "__main__":
    main()
