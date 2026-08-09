import numpy as np
import pygame
from boid_model import BoidModel
from boids.migrator import Migrator
from boids.predator import Predator
from variables import GRID_SIZE

SCREEN_W, SCREEN_H = 1600, 900
WORLD_W, WORLD_H = SCREEN_W * 2, SCREEN_H * 4

def clamp_camera(cam_x, cam_y, zoom):
    view_w = SCREEN_W / zoom
    view_h = SCREEN_H / zoom

    if view_w >= WORLD_W:
        max_x = 0
        cam_x = (WORLD_W - view_w) / 2
    else:
        max_x = WORLD_W - view_w
        cam_x = np.clip(cam_x, 0, max_x)

    if view_h >= WORLD_H:
        max_y = 0
        cam_y = (WORLD_H - view_h) / 2
    else:
        max_y = WORLD_H - view_h
        cam_y = np.clip(cam_y, 0, max_y)

    return cam_x, cam_y

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    model = BoidModel(160, WORLD_W, WORLD_H, num_obstacles=35)

    cam_x, cam_y = 0.0, 0.0
    zoom = 1.0
    dragging = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return
            
            elif event.type == pygame.MOUSEWHEEL:
                old_zoom = zoom
                zoom = float(np.clip(zoom + event.y * 0.05, 0.3, 2.5))
                
                cam_x, cam_y = clamp_camera(cam_x, cam_y, zoom)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: 
                    dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                cam_x -= dx / zoom
                cam_y -= dy / zoom
                
                cam_x, cam_y = clamp_camera(cam_x, cam_y, zoom)

        model.step()
        screen.fill((20, 20, 25))

        start_col = max(0, int(cam_x // GRID_SIZE))
        end_col = min(model.cols, int((cam_x + SCREEN_W / zoom) // GRID_SIZE) + 2)
        
        start_row = max(0, int(cam_y // GRID_SIZE))
        end_row = min(model.rows, int((cam_y + SCREEN_H / zoom) // GRID_SIZE) + 2)

        for obs in model.obstacles:
            r = obs.get_rect()
            draw_rect = pygame.Rect(
                (r.x - cam_x) * zoom, (r.y - cam_y) * zoom,
                r.width * zoom, r.height * zoom
            )
            pygame.draw.rect(screen, (60, 60, 70), draw_rect)
            pygame.draw.rect(screen, (100, 100, 110), draw_rect, 2)

        time_factor = pygame.time.get_ticks() * 0.003

        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                val = model.terrain_height[r][c]

                if hasattr(model, 'river_map') and model.river_map[r][c]:
                    wave = np.sin(r * 0.4 + time_factor) * 0.1
                    
                    water_depth = np.clip(val + wave, 0.0, 1.0)

                    c_shallow = pygame.Color(70, 150, 200)
                    c_deep = pygame.Color(20, 60, 130)

                    color = c_shallow.lerp(c_deep, water_depth)

                elif val > model.mountain_threshold:
                    factor = (val - model.mountain_threshold) / (1.0 - model.mountain_threshold)
                    c_low = pygame.Color(110, 100, 90)
                    c_high = pygame.Color(45, 40, 35)
                    color = c_low.lerp(c_high, factor)
                    
                elif val > model.forest_threshold:
                    factor = (val - model.forest_threshold) / (model.mountain_threshold - model.forest_threshold)
                    c_low = pygame.Color(45, 150, 45)
                    c_high = pygame.Color(20, 75, 20)
                    color = c_low.lerp(c_high, factor)
                    
                else:
                    factor = val / model.forest_threshold
                    c_low = pygame.Color(185, 245, 185)
                    c_high = pygame.Color(115, 215, 115)
                    color = c_low.lerp(c_high, factor)

                rect = pygame.Rect(
                    (c * GRID_SIZE - cam_x) * zoom, 
                    (r * GRID_SIZE - cam_y) * zoom, 
                    GRID_SIZE * zoom + 1, 
                    GRID_SIZE * zoom + 1
                )

                pygame.draw.rect(screen, color, rect)



        for agent in model.agents:
            if agent.pos is not None:
                rx = (agent.pos[0] - cam_x) * zoom
                ry = (agent.pos[1] - cam_y) * zoom
                
                if -50 <= rx <= SCREEN_W + 50 and -50 <= ry <= SCREEN_H + 50:
                    size = 8 * zoom
                    vel = agent.velocity
                    speed = np.linalg.norm(vel)
                    angle = np.arctan2(vel[1], vel[0]) if speed > 0 else 0.0
                    
                    if isinstance(agent, Migrator):
                        if getattr(agent, 'scared', False):
                            color = (255, 140, 0)
                        elif getattr(agent, 'is_feeding', False):
                            color = (30, 110, 90)
                        else:
                            color = (200, 200, 0)

                        p1 = (rx + np.cos(angle) * size, ry + np.sin(angle) * size)
                        p2 = (rx + np.cos(angle + 2.5) * size/2, ry + np.sin(angle + 2.5) * size/2)
                        p3 = (rx + np.cos(angle - 2.5) * size/2, ry + np.sin(angle - 2.5) * size/2)
                        pygame.draw.polygon(screen, color, [p1, p2, p3])
                        
                    elif isinstance(agent, Predator):
                        p_size = 14 * zoom
                        p1 = (rx + np.cos(angle) * p_size, ry + np.sin(angle) * p_size)
                        p2 = (rx + np.cos(angle + 2.3) * p_size/2, ry + np.sin(angle + 2.3) * p_size/2)
                        p3 = (rx + np.cos(angle - 2.3) * p_size/2, ry + np.sin(angle - 2.3) * p_size/2)
                        pygame.draw.polygon(screen, (255, 0, 50), [p1, p2, p3])

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()