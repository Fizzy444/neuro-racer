import pygame
import numpy as np
import sys
import os
import math
import random
from physics import Cars, Params
from track import Track
from race import ray_cast, N_RAYS, RAY_ANGLES, RAY_WIN, RAY_RANGE, OBS_SIZE
from brain import Brains

class Camera:
    def __init__(self, vw, vh):
        self.vw = vw
        self.vh = vh
        self.center = np.zeros(2)
        self.scale = 1.0

    def follow(self, pos, zoom=5.0):
        self.center = self.center + (pos - self.center) * 0.15
        self.scale = self.scale + (zoom - self.scale) * 0.1

    def tf(self, pts):
        out = np.empty(pts.shape)
        if len(pts.shape) == 1:
            out[0] = (pts[0] - self.center[0]) * self.scale + self.vw / 2
            out[1] = self.vh / 2 - (pts[1] - self.center[1]) * self.scale
        else:
            out[..., 0] = (pts[..., 0] - self.center[0]) * self.scale + self.vw / 2
            out[..., 1] = self.vh / 2 - (pts[..., 1] - self.center[1]) * self.scale
        return out

def manual_drive():
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    pygame.init()
    
    # Crisp high-res resolution
    W, H = 1200, 800
    screen = pygame.display.set_mode((W, H))
    
    pygame.display.set_caption("Retro Arcade Racer - Crisp UI")
    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.SysFont("monospace", 22, bold=True)
        big_font = pygame.font.SysFont("monospace", 42, bold=True)
    except:
        font = pygame.font.Font(None, 28)
        big_font = pygame.font.Font(None, 52)

    # Retro Colors
    GRASS = (20, 140, 60)
    TARMAC = (90, 90, 90)
    CURB_A = (255, 255, 255)
    CURB_B = (220, 20, 40)
    UI_TEXT = (255, 255, 255)

    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 0)] # P1 (Yellow), P2 (Cyan), AI (Red)

    app_running = True
    
    while app_running:
        # --- START MENU ---
        menu_state = 0 # 0: Players, 1: AI, 2: Laps, 3: Map, 4: Done
        num_players = 1
        use_ai = False
        target_laps = 3
        track_seed = 42
        
        button_w, button_h = 400, 50
        start_y = H // 2 - 20
        
        menu_trails = []
        for _ in range(200):
            speed = random.uniform(0.2, 1.5)
            c = int(40 + speed * 80)
            menu_trails.append([random.randint(0, W), random.randint(0, H), speed, random.randint(1, 3), (c, c, c + 30)])

        while menu_state < 4:
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    app_running = False
                    menu_state = 4 # Break menu loop
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_click = True
            
            if not app_running: break
            
            # --- Soothing Parallax Starfield Background ---
            screen.fill((5, 5, 12)) # Deep space dark
            
            for t in menu_trails:
                t[0] -= t[2] # drift slowly
                if t[0] < -10: 
                    t[0] = W + 10
                    t[1] = random.randint(0, H)
                
                pygame.draw.rect(screen, t[4], (int(t[0]), int(t[1]), t[3], t[3]))
            
            # Pulsating Title
            pulse = math.sin(pygame.time.get_ticks() / 300.0)
            title_col = (255, int(150 + 100 * pulse), 0)
            title_surf = big_font.render("--- ARCADE RACER ---", True, title_col)
            shadow_surf = big_font.render("--- ARCADE RACER ---", True, (0, 0, 0))
            
            screen.blit(shadow_surf, (W // 2 - title_surf.get_width() // 2 + 4, H // 2 - 180 + 4))
            screen.blit(title_surf, (W // 2 - title_surf.get_width() // 2, H // 2 - 180))
            
            if menu_state == 0:
                prompt_str = "Select Players:"
                buttons = [
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y, button_w, button_h), "text": "1 Player (Arrows)", "val": 1},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y + button_h + 20, button_w, button_h), "text": "2 Players (P2 WASD)", "val": 2}
                ]
            elif menu_state == 1:
                prompt_str = "Add AI Opponent?"
                buttons = [
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y, button_w, button_h), "text": "Yes (AI Enabled)", "val": True},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y + button_h + 20, button_w, button_h), "text": "No", "val": False}
                ]
            elif menu_state == 2:
                prompt_str = "Number of Laps:"
                buttons = [
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y, button_w, button_h), "text": "1 Lap", "val": 1},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y + button_h + 20, button_w, button_h), "text": "3 Laps", "val": 3},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y + button_h * 2 + 40, button_w, button_h), "text": "5 Laps", "val": 5}
                ]
            else:
                prompt_str = "Select Track:"
                buttons = [
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y, button_w, button_h), "text": "Track 1 (Beginner)", "val": 42},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y + button_h + 20, button_w, button_h), "text": "Track 2 (Advanced)", "val": 1337},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, start_y + button_h * 2 + 40, button_w, button_h), "text": "Random Track", "val": None}
                ]
                
            prompt_surf = font.render(prompt_str, True, (0, 255, 255))
            prompt_shadow = font.render(prompt_str, True, (0, 0, 0))
            screen.blit(prompt_shadow, (W // 2 - prompt_surf.get_width() // 2 + 2, start_y - 40 + 2))
            screen.blit(prompt_surf, (W // 2 - prompt_surf.get_width() // 2, start_y - 40))
            
            for btn in buttons:
                rect = btn["rect"]
                is_hover = rect.collidepoint(mouse_pos)
                
                bg_color = (255, 0, 255) if is_hover else (0, 0, 150)
                if is_hover and mouse_click:
                    if menu_state == 0:
                        num_players = btn["val"]
                        menu_state = 1
                    elif menu_state == 1:
                        use_ai = btn["val"]
                        menu_state = 2
                    elif menu_state == 2:
                        target_laps = btn["val"]
                        menu_state = 3
                    else:
                        track_seed = btn["val"]
                        menu_state = 4
                
                # Button Shadow
                shadow_rect = rect.copy()
                shadow_rect.y += 4
                shadow_rect.x += 4
                pygame.draw.rect(screen, (0, 0, 0), shadow_rect)
                
                # Button Base
                pygame.draw.rect(screen, bg_color, rect)
                pygame.draw.rect(screen, (255, 255, 255), rect, width=3)
                
                text_surf = font.render(btn["text"], True, (255, 255, 255))
                screen.blit(text_surf, (rect.centerx - text_surf.get_width() // 2, rect.centery - text_surf.get_height() // 2))
                
            pygame.display.flip()
            clock.tick(60)

        if not app_running:
            break
            
        # --- GAME INIT ---
        n_cars = num_players + (1 if use_ai else 0)
        ai_index = n_cars - 1 if use_ai else -1
        
        ai_brain = None
        if use_ai:
            if os.path.exists("models/snapshot.npz"):
                ai_brain = Brains.load("models/snapshot.npz", 1)
            elif os.path.exists("models/latest.npz"):
                ai_brain = Brains.load("models/latest.npz", 1)
            else:
                ai_brain = Brains(1, [OBS_SIZE, 32, 24, 24, 16, 12, 2])

        track = Track(seed=track_seed)
        cars = Cars(n_cars, Params())
        trails = [[] for _ in range(n_cars)]
        cars_progress = np.zeros(n_cars, dtype=float)
        winner_text = None
        win_timer = 0.0
        
        def reset_cars(t_obj):
            cars.reset(t_obj.c[0], t_obj.theta[0])
            if n_cars > 1:
                nrm = t_obj.normal[0]
                spread = 16.0 
                spacing = spread / (n_cars - 1)
                for i in range(n_cars):
                    offset = -spread/2 + i * spacing
                    cars.x[i] += nrm[0] * offset
                    cars.y[i] += nrm[1] * offset
            cars.idx = np.zeros(n_cars, dtype=int)
            cars_progress.fill(0)
            nonlocal winner_text, win_timer
            winner_text = None
            win_timer = 0.0
            for t in trails: t.clear()
            
        reset_cars(track)
        
        cam = Camera(W, H)
        running = True
        paused = False
        DT = 1.0 / 60.0
        countdown_timer = 3.99 # Start countdown

        # --- GAME LOOP ---
        while running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    app_running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_click = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        paused = not paused
                    elif not paused:
                        if event.key == pygame.K_r:
                            reset_cars(track)
                            countdown_timer = 3.99
                        elif event.key == pygame.K_t:
                            track = Track(seed=None)
                            reset_cars(track)
                            countdown_timer = 3.99
            
            if not paused:
                if winner_text is not None:
                    win_timer -= DT
                    if win_timer <= 0:
                        running = False # Return to main menu
                elif countdown_timer > 0:
                    countdown_timer -= DT
                else:
                    keys = pygame.key.get_pressed()
                    steer = np.zeros(n_cars)
                    throttle = np.zeros(n_cars)
                    
                    # P1 Control
                    if keys[pygame.K_LEFT]: steer[0] = 1.0
                    elif keys[pygame.K_RIGHT]: steer[0] = -1.0
                    
                    if keys[pygame.K_UP]: throttle[0] = 1.0
                    elif keys[pygame.K_DOWN]: throttle[0] = -1.0
                        
                    # P2 Control
                    if num_players >= 2:
                        if keys[pygame.K_w]: throttle[1] = 1.0
                        elif keys[pygame.K_s]: throttle[1] = -1.0
                        if keys[pygame.K_a]: steer[1] = 1.0
                        elif keys[pygame.K_d]: steer[1] = -1.0
                        
                    # AI Control
                    if use_ai and ai_brain is not None:
                        a = ai_index
                        obs = np.zeros((1, OBS_SIZE))
                        pos = np.stack([cars.x[a:a+1], cars.y[a:a+1]], 1)
                        ang = cars.psi[a:a+1][:, None] + RAY_ANGLES[None, :]
                        d = np.stack([np.cos(ang), np.sin(ang)], -1)
                        w = track.window(cars.idx[a:a+1], RAY_WIN)
                        w1 = (w + 1) % track.m
                        A = np.concatenate([track.left[w], track.right[w]], 1)
                        B = np.concatenate([track.left[w1], track.right[w1]], 1)
                        dist = ray_cast(pos, d, A, B, RAY_RANGE)
                        obs[0, :N_RAYS] = dist[0] / RAY_RANGE
                        obs[0, N_RAYS] = cars.vx[a] / 40.0
                        obs[0, N_RAYS + 1] = cars.vy[a] / 8.0
                        obs[0, N_RAYS + 2] = cars.w[a] / 2.0
                        obs[0, N_RAYS + 3] = cars.delta[a] / cars.p.max_steer
                        
                        ai_out = ai_brain.forward(obs)[0]
                        steer[a] = ai_out[0]
                        throttle[a] = ai_out[1]
                        
                    active = np.ones(n_cars, dtype=bool)
                    cars.step(steer, throttle, active, DT)
                    
                    # Collisions and Lap Progress
                    pos = np.stack([cars.x, cars.y], 1)
                    w = track.window(cars.idx, 10)
                    d2 = ((track.c[w] - pos[:, None, :]) ** 2).sum(-1)
                    new_idx = w[np.arange(n_cars), d2.argmin(1)]
                    
                    delta = (new_idx - cars.idx + track.m // 2) % track.m - track.m // 2
                    cars.idx = new_idx
                    cars_progress += delta
                    
                    # Check win conditions
                    for k in range(n_cars):
                        laps_completed = cars_progress[k] / track.m
                        if laps_completed >= target_laps and winner_text is None:
                            if k == 0: winner_text = "P1 WINS!"
                            elif k == 1 and num_players >= 2: winner_text = "P2 WINS!"
                            else: winner_text = "AI WINS!"
                            win_timer = 5.0 # Show win screen for 5 seconds
                    
                    nrm = track.normal[new_idx]
                    cp = track.c[new_idx]
                    lat = (cars.x - cp[:, 0]) * nrm[:, 0] + (cars.y - cp[:, 1]) * nrm[:, 1]
                    max_lat = track.width / 2.0 - cars.p.width / 2.0
                    
                    for k in range(n_cars):
                        if abs(lat[k]) > max_lat:
                            correction = (abs(lat[k]) - max_lat) * np.sign(lat[k])
                            cars.x[k] -= correction * nrm[k, 0]
                            cars.y[k] -= correction * nrm[k, 1]
                            cars.vx[k] *= 0.90
                            cars.vy[k] *= 0.5
                
                # Trails
                for k in range(n_cars):
                    speed = np.hypot(cars.vx[k], cars.vy[k])
                    t = trails[k]
                    if len(t) == 0 or np.hypot(t[-1][0] - cars.x[k], t[-1][1] - cars.y[k]) > 1.0:
                        t.append((cars.x[k], cars.y[k], speed))
                    if len(t) > 100: t.pop(0)
                
            # --- DRAWING ---
            screen.fill(GRASS) # Retro green
            
            # Dynamic Camera
            spread_x = cars.x.max() - cars.x.min()
            spread_y = cars.y.max() - cars.y.min()
            
            # Zoom adjusted for larger screen
            zoom_x = W / max(spread_x + 80.0, 10.0)
            zoom_y = H / max(spread_y + 80.0, 10.0)
            
            target_zoom = min(zoom_x, zoom_y)
            target_zoom = max(0.4, min(3.5, target_zoom)) 
            
            mid_x = (cars.x.max() + cars.x.min()) / 2
            mid_y = (cars.y.max() + cars.y.min()) / 2
            
            cam.follow(np.array([mid_x, mid_y]), zoom=target_zoom)
            
            L = cam.tf(track.left)
            R = cam.tf(track.right)
            
            # Fill track
            for i in range(track.m):
                j = (i + 1) % track.m
                pygame.draw.polygon(screen, TARMAC, [L[i], L[j], R[j], R[i]])
                
            # Draw curbs
            for i in range(track.m):
                j = (i + 1) % track.m
                col = CURB_A if (i // 2) % 2 == 0 else CURB_B
                pygame.draw.line(screen, col, L[i], L[j], 6)
                pygame.draw.line(screen, col, R[i], R[j], 6)
                
            # Draw grid lines
            for i in range(0, track.m, 6):
                j = (i + 3) % track.m
                pygame.draw.line(screen, (130, 130, 0), cam.tf(track.c[i]), cam.tf(track.c[j]), 3)
                
            # --- START/FINISH LINE ---
            num_squares = 14
            for s in range(num_squares):
                f1 = s / num_squares
                f2 = (s + 1) / num_squares
                pt1 = L[0] + (R[0] - L[0]) * f1
                pt2 = L[0] + (R[0] - L[0]) * f2
                color = (255, 255, 255) if s % 2 == 0 else (0, 0, 0)
                pygame.draw.line(screen, color, pt1, pt2, 10)
                
            # Draw Trails
            for k in range(n_cars):
                for p in trails[k]:
                    pt = cam.tf(np.array(p[:2]))
                    pygame.draw.rect(screen, colors[k], (int(pt[0])-2, int(pt[1])-2, 5, 5))

            # Render Cars
            CAR_L, CAR_W = cars.p.length, cars.p.width
            for k in range(n_cars):
                cs, sn = np.cos(cars.psi[k]), np.sin(cars.psi[k])
                
                def local_to_global(lx, ly):
                    gx = cars.x[k] + lx * cs - ly * sn
                    gy = cars.y[k] + lx * sn + ly * cs
                    return cam.tf(np.stack([gx, gy], -1))
                
                # Main body
                pts = local_to_global(np.array([1, 1, -1, -1]) * CAR_L / 2, np.array([1, -1, -1, 1]) * CAR_W / 2)
                pygame.draw.polygon(screen, colors[k], pts)
                pygame.draw.polygon(screen, (0, 0, 0), pts, 3)
                
                # Windshield
                w_pts = local_to_global(np.array([0.4, 0.4, -0.1, -0.1]) * CAR_L / 2, np.array([0.8, -0.8, -0.8, 0.8]) * CAR_W / 2)
                pygame.draw.polygon(screen, (20, 20, 30), w_pts)
                
                # Roof
                r_pts = local_to_global(np.array([0.0, 0.0, -0.6, -0.6]) * CAR_L / 2, np.array([0.7, -0.7, -0.7, 0.7]) * CAR_W / 2)
                pygame.draw.polygon(screen, colors[k], r_pts)
                pygame.draw.polygon(screen, (0, 0, 0), r_pts, 2)
                
                # Headlights
                h1 = local_to_global(np.array([0.9, 1.0, 1.0, 0.9]) * CAR_L / 2, np.array([0.9, 0.9, 0.5, 0.5]) * CAR_W / 2)
                h2 = local_to_global(np.array([0.9, 1.0, 1.0, 0.9]) * CAR_L / 2, np.array([-0.5, -0.5, -0.9, -0.9]) * CAR_W / 2)
                pygame.draw.polygon(screen, (255, 255, 150), h1)
                pygame.draw.polygon(screen, (255, 255, 150), h2)
                
            # Info HUD
            def get_lap_text(k):
                laps = int(cars_progress[k] / track.m) + 1
                return min(laps, target_laps)
                
            texts = [f"P1: {np.hypot(cars.vx[0], cars.vy[0]) * 3.6:5.0f} km/h  Lap: {get_lap_text(0)}/{target_laps}"]
            if num_players >= 2: texts.append(f"P2: {np.hypot(cars.vx[1], cars.vy[1]) * 3.6:5.0f} km/h  Lap: {get_lap_text(1)}/{target_laps}")
            if use_ai: texts.append(f"AI: {np.hypot(cars.vx[ai_index], cars.vy[ai_index]) * 3.6:5.0f} km/h  Lap: {get_lap_text(ai_index)}/{target_laps}")
            
            for i, text in enumerate(texts):
                surf = font.render(text, True, UI_TEXT)
                screen.blit(surf, (15, 15 + i * 25))
                
            # --- MINIMAP ---
            mm_size = 200
            mm_rect = pygame.Rect(W - mm_size - 15, 15, mm_size, mm_size)
            pygame.draw.rect(screen, (0, 0, 0), mm_rect)
            pygame.draw.rect(screen, (0, 255, 0), mm_rect, width=3)
            
            mn, mx = track.c.min(axis=0), track.c.max(axis=0)
            track_w = mx[0] - mn[0]
            track_h = mx[1] - mn[1]
            scale = (mm_size - 20) / max(track_w, track_h, 1.0)
            
            def to_mm(px, py):
                cx = (px - mn[0]) - track_w / 2
                cy = (py - mn[1]) - track_h / 2
                return (int(mm_rect.centerx + cx * scale), int(mm_rect.centery - cy * scale))
                
            mm_pts = [to_mm(p[0], p[1]) for p in track.c]
            pygame.draw.lines(screen, (0, 150, 0), False, mm_pts, 2)
            
            for k in range(n_cars):
                pos_mm = to_mm(cars.x[k], cars.y[k])
                pygame.draw.rect(screen, colors[k], (pos_mm[0]-3, pos_mm[1]-3, 7, 7))
                
            # --- WIN OVERLAY ---
            if winner_text is not None:
                overlay = pygame.Surface((W, H))
                overlay.set_alpha(150)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))
                
                win_surf = big_font.render(winner_text, True, (255, 215, 0))
                scaled_surf = pygame.transform.scale(win_surf, (win_surf.get_width() * 3, win_surf.get_height() * 3))
                screen.blit(scaled_surf, (W // 2 - scaled_surf.get_width() // 2, H // 2 - scaled_surf.get_height() // 2))
                
            # --- COUNTDOWN OVERLAY ---
            elif countdown_timer > 0:
                overlay = pygame.Surface((W, H))
                overlay.set_alpha(150)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))
                
                if countdown_timer > 1.0:
                    cd_text = str(int(countdown_timer))
                    color = (255, 50, 50)
                else:
                    cd_text = "GO!"
                    color = (50, 255, 50)
                    
                cd_surf = big_font.render(cd_text, True, color)
                scaled_surf = pygame.transform.scale(cd_surf, (cd_surf.get_width() * 3, cd_surf.get_height() * 3))
                screen.blit(scaled_surf, (W // 2 - scaled_surf.get_width() // 2, H // 2 - scaled_surf.get_height() // 2))
                
            # --- ESCAPE PAUSE MENU ---
            elif paused:
                # Dim background
                overlay = pygame.Surface((W, H))
                overlay.set_alpha(180)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))
                
                title_surf = big_font.render("--- PAUSED ---", True, (255, 255, 0))
                screen.blit(title_surf, (W // 2 - title_surf.get_width() // 2, H // 2 - 120))
                
                button_w, button_h = 400, 50
                esc_buttons = [
                    {"rect": pygame.Rect(W // 2 - button_w // 2, H // 2 - 30, button_w, button_h), "text": "Resume", "action": "resume"},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, H // 2 + 35, button_w, button_h), "text": "Main Menu", "action": "menu"},
                    {"rect": pygame.Rect(W // 2 - button_w // 2, H // 2 + 100, button_w, button_h), "text": "Quit Game", "action": "quit"}
                ]
                
                for btn in esc_buttons:
                    rect = btn["rect"]
                    is_hover = rect.collidepoint(mouse_pos)
                    bg_color = (255, 0, 255) if is_hover else (0, 0, 150)
                    
                    if is_hover and mouse_click:
                        if btn["action"] == "resume":
                            paused = False
                        elif btn["action"] == "menu":
                            running = False # Break inner game loop
                        elif btn["action"] == "quit":
                            running = False
                            app_running = False
                            
                    pygame.draw.rect(screen, bg_color, rect)
                    pygame.draw.rect(screen, (255, 255, 255), rect, width=3)
                    
                    text_surf = font.render(btn["text"], True, (255, 255, 255))
                    screen.blit(text_surf, (rect.centerx - text_surf.get_width() // 2, rect.centery - text_surf.get_height() // 2))
                
            pygame.display.flip()
            clock.tick(60)
            
    pygame.quit()

if __name__ == "__main__":
    manual_drive()
