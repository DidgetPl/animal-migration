import numpy as np
import pygame
from boids_logic import GRID_SIZE, BoidModel, Migrator, Predator

SCREEN_W, SCREEN_H = 1600, 900
WORLD_W, WORLD_H = SCREEN_W * 2, SCREEN_H * 2

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    model = BoidModel(160, WORLD_W, WORLD_H, num_obstacles=35)

    cam_x, cam_y = 0, 0
    zoom = 1.0
    dragging = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            elif event.type == pygame.MOUSEWHEEL:
                zoom = np.clip(zoom + event.y * 0.1, 0.4, 2.5)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                cam_x = np.clip(cam_x - dx/zoom, 0, WORLD_W - SCREEN_W/zoom)
                cam_y = np.clip(cam_y - dy/zoom, 0, WORLD_H - SCREEN_H/zoom)

        model.step()
        screen.fill((20, 20, 25))

        for obs in model.obstacles:
            r = obs.get_rect()
            draw_rect = pygame.Rect(
                (r.x - cam_x) * zoom, (r.y - cam_y) * zoom,
                r.width * zoom, r.height * zoom
            )
            pygame.draw.rect(screen, (60, 60, 70), draw_rect)
            pygame.draw.rect(screen, (100, 100, 110), draw_rect, 2)

        for r in range(model.rows):
            for c in range(model.cols):
                val = model.terrain_height[r][c]
                
                if val > 0.65:
                    factor = (val - 0.65) / (1.0 - 0.65)
                    c_low = pygame.Color(110, 100, 90)
                    c_high = pygame.Color(45, 40, 35)
                    color = c_low.lerp(c_high, factor)
                    
                elif val > 0.55:
                    factor = (val - 0.55) / (0.65 - 0.55)
                    c_low = pygame.Color(45, 150, 45)
                    c_high = pygame.Color(20, 75, 20)
                    color = c_low.lerp(c_high, factor)
                    
                else:
                    factor = val / 0.55
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

        # Rysowanie agentów
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
                        # Kolor zależy od strachu: przerażony = pomarańczowy, bezpieczny = turkusowy
                        color = (255, 140, 0) if getattr(agent, 'scared', False) else (0, 255, 200)
                        
                        p1 = (rx + np.cos(angle) * size, ry + np.sin(angle) * size)
                        p2 = (rx + np.cos(angle + 2.5) * size/2, ry + np.sin(angle + 2.5) * size/2)
                        p3 = (rx + np.cos(angle - 2.5) * size/2, ry + np.sin(angle - 2.5) * size/2)
                        pygame.draw.polygon(screen, color, [p1, p2, p3])
                        
                    elif isinstance(agent, Predator):
                        # Drapieżnik: Większy, czerwony trójkąt
                        p_size = 14 * zoom
                        p1 = (rx + np.cos(angle) * p_size, ry + np.sin(angle) * p_size)
                        p2 = (rx + np.cos(angle + 2.3) * p_size/2, ry + np.sin(angle + 2.3) * p_size/2)
                        p3 = (rx + np.cos(angle - 2.3) * p_size/2, ry + np.sin(angle - 2.3) * p_size/2)
                        pygame.draw.polygon(screen, (255, 0, 50), [p1, p2, p3])
        
        #UI
        hunting_count = sum(
            1 for obj in model.agents
            if getattr(obj, "is_hunting", False)
        )

        font = pygame.font.SysFont("Arial", 24)
        text = font.render(
            f"Polujące drapieżniki: {hunting_count}",
            True,
            (255, 255, 255)
        )
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()