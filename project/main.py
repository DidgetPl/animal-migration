import numpy as np
import pygame
from boids_logic import GRID_SIZE, BoidModel, Migrator

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
                cost = model.terrain_map[r][c]
                if cost > 5.0: color = (80, 70, 60)
                elif cost > 2.0: color = (34, 139, 34)
                else: color = (144, 238, 144)

                rect = pygame.Rect((c*GRID_SIZE - cam_x)*zoom, (r*GRID_SIZE - cam_y)*zoom, 
                                GRID_SIZE*zoom + 1, GRID_SIZE*zoom + 1)
                pygame.draw.rect(screen, color, rect)

        for agent in model.agents:
            if isinstance(agent, Migrator) and agent.pos is not None:
                rx = (agent.pos[0] - cam_x) * zoom
                ry = (agent.pos[1] - cam_y) * zoom

                if -50 <= rx <= SCREEN_W + 50 and -50 <= ry <= SCREEN_H + 50:
                    size = 8 * zoom
                    vel = agent.velocity
                    speed = np.linalg.norm(vel)

                    angle = np.arctan2(vel[1], vel[0]) if speed > 0 else 0.0

                    p1 = (rx + np.cos(angle) * size, ry + np.sin(angle) * size)
                    p2 = (rx + np.cos(angle + 2.5) * size/2, ry + np.sin(angle + 2.5) * size/2)
                    p3 = (rx + np.cos(angle - 2.5) * size/2, ry + np.sin(angle - 2.5) * size/2)
                    pygame.draw.polygon(screen, (0, 255, 200), [p1, p2, p3])
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()