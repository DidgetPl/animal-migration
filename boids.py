import numpy as np
import pygame
from mesa import Agent, Model
from mesa.space import ContinuousSpace

# --- KONFIGURACJA ---
SCREEN_W, SCREEN_H = 1600, 900
WORLD_W, WORLD_H = SCREEN_W * 3, SCREEN_H * 3
NUM_BOIDS = 120
VISUAL_RADIUS = 50
MIN_DISTANCE = 20
MAX_SPEED = 5
BORDER_MARGIN = 50

# Parametry Zoomu
MIN_ZOOM = 0.5   # Widzimy więcej świata (pomniejszenie)
MAX_ZOOM = 2.0   # Widzimy detale (powiększenie)
ZOOM_STEP = 0.1

class Boid(Agent):
    def __init__(self, model, pos, velocity):
        super().__init__(model)
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def step(self):
        neighbors = self.model.space.get_neighbors(self.pos, VISUAL_RADIUS, False)
        if neighbors:
            sep_vec = self.separation(neighbors)
            ali_vec = self.alignment(neighbors)
            coh_vec = self.cohesion(neighbors)
            self.velocity += (sep_vec * 1.5 + ali_vec * 1.0 + coh_vec * 1.0)

        # Odbicie od ścian
        if self.pos[0] < BORDER_MARGIN: self.velocity[0] += 0.5
        elif self.pos[0] > WORLD_W - BORDER_MARGIN: self.velocity[0] -= 0.5
        if self.pos[1] < BORDER_MARGIN: self.velocity[1] += 0.5
        elif self.pos[1] > WORLD_H - BORDER_MARGIN: self.velocity[1] -= 0.5
        
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = (self.velocity / speed) * MAX_SPEED

        new_pos = self.pos + self.velocity
        eps = 0.001
        new_pos[0] = np.clip(new_pos[0], eps, WORLD_W - eps)
        new_pos[1] = np.clip(new_pos[1], eps, WORLD_H - eps)
        
        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def separation(self, neighbors):
        steer = np.zeros(2)
        for n in neighbors:
            dist = np.linalg.norm(self.pos - n.pos)
            if 0 < dist < MIN_DISTANCE:
                steer += (self.pos - n.pos) / dist
        return steer

    def alignment(self, neighbors):
        avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
        return (avg_vel - self.velocity) * 0.1

    def cohesion(self, neighbors):
        avg_pos = np.mean([n.pos for n in neighbors], axis=0)
        return (avg_pos - self.pos) * 0.05

class BoidModel(Model):
    def __init__(self, n):
        super().__init__()
        self.space = ContinuousSpace(WORLD_W, WORLD_H, False)
        for _ in range(n):
            pos = np.array([self.random.random() * WORLD_W, self.random.random() * WORLD_H])
            vel = (np.random.rand(2) - 0.5) * 5
            boid = Boid(self, pos, vel)
            self.space.place_agent(boid, pos)

    def step(self):
        agents = list(self.agents)
        self.random.shuffle(agents)
        for boid in agents:
            boid.step()

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    model = BoidModel(NUM_BOIDS)

    cam_x, cam_y = 0, 0
    zoom_level = 1.0
    dragging = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- OBSŁUGA ZOOMU ---
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0: # Scroll w górę
                    zoom_level = min(MAX_ZOOM, zoom_level + ZOOM_STEP)
                else: # Scroll w dół
                    zoom_level = max(MIN_ZOOM, zoom_level - ZOOM_STEP)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx, dy = event.rel
                    # Przesunięcie musi brać pod uwagę zoom, żeby mysz "nadążała" za terenem
                    cam_x -= dx / zoom_level
                    cam_y -= dy / zoom_level
                    
                    cam_x = max(0, min(cam_x, WORLD_W - SCREEN_W / zoom_level))
                    cam_y = max(0, min(cam_y, WORLD_H - SCREEN_H / zoom_level))

        model.step()
        screen.fill((10, 10, 15))
        
        # Obliczanie widocznego obszaru świata
        visible_world_rect = pygame.Rect(
            (0 - cam_x) * zoom_level, 
            (0 - cam_y) * zoom_level, 
            WORLD_W * zoom_level, 
            WORLD_H * zoom_level
        )
        pygame.draw.rect(screen, (80, 20, 20), visible_world_rect, int(5 * zoom_level) + 1)

        for boid in model.agents:
            # Kluczowa formuła renderowania z zoomem:
            # Ekran = (PozycjaŚwiat - Kamera) * Zoom
            render_x = (boid.pos[0] - cam_x) * zoom_level
            render_y = (boid.pos[1] - cam_y) * zoom_level
            
            # Rysujemy tylko te, które są na ekranie
            if -20 < render_x < SCREEN_W + 20 and -20 < render_y < SCREEN_H + 20:
                vel_norm = np.linalg.norm(boid.velocity)
                direction = boid.velocity / vel_norm if vel_norm > 0 else np.array([1, 0])
                
                # Skalowanie wielkości boida
                size = 10 * zoom_level
                draw_pos = np.array([render_x, render_y])
                
                tip = draw_pos + direction * size
                base_l = draw_pos - direction * (size/2) + np.array([-direction[1], direction[0]]) * (size/2.5)
                base_r = draw_pos - direction * (size/2) - np.array([-direction[1], direction[0]]) * (size/2.5)
                
                pygame.draw.polygon(screen, (0, 255, 200), [tip, base_l, base_r])

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()