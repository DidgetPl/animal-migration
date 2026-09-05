import numpy as np
import pygame
from simulation.boids.migrator import Migrator
from simulation.boids.obstacle import Obstacle
from simulation.boids.predator import Predator
from variables import GRID_SIZE

RIVER_SHALLOW_COLOR = (70, 150, 200)
RIVER_DEEP_COLOR = (20, 60, 130)
MOUNTAIN_LOW_COLOR = (110, 100, 90)
MOUNTAIN_HIGH_COLOR = (45, 40, 35)
FOREST_LOW_COLOR = (141, 143, 41)
FOREST_HIGH_COLOR = (65, 75, 20)
GRASS_LOW_COLOR = (220, 188, 104)
GRASS_HIGH_COLOR = (174, 127, 77)
GNU_COLOR = (101, 67, 33)
SCARED_GNU_COLOR = (202, 72, 21)
FEEDING_GNU_COLOR = (196, 164, 132)
PREDATOR_COLOR = (255, 50, 50)
TREE_INTERIOR_COLOR = (101, 110, 12)
TREE_OUTLINE_COLOR = (64, 72, 8)
ROCK_INTERIOR_COLOR = (80, 80, 85)
ROCK_OUTLINE_COLOR = (50, 50, 55)
FLOW_ARROW_COLOR = (240, 240, 255, 120)

class WorldRenderer:
    def __init__(self, screen_w, screen_h, flip=True):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.flip = flip
        self.overlay_surface = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)

    def world_to_screen(self, pos, cam_x, cam_y, zoom):
        sx = (pos[0] - cam_x) * zoom
        sy = (pos[1] - cam_y) * zoom
        return sx, sy

    def is_visible(self, sx, sy, margin=50):
        return -margin <= sx <= self.screen_w + margin and -margin <= sy <= self.screen_h + margin

    def draw_terrain(self, screen, model, cam_x, cam_y, zoom):
        start_col = max(0, int(cam_x // GRID_SIZE))
        end_col = min(model.cols, int((cam_x + self.screen_w / zoom) // GRID_SIZE) + 2)
        
        start_row = max(0, int(cam_y // GRID_SIZE))
        end_row = min(model.rows, int((cam_y + self.screen_h / zoom) // GRID_SIZE) + 2)

        time_factor = pygame.time.get_ticks() * 0.003

        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                val = model.terrain_height[r][c]

                if hasattr(model, 'river_map') and model.river_map[r][c]:
                    wave = np.sin(r * 0.4 + time_factor) * 0.1
                    water_depth = np.clip(val + wave, 0.0, 1.0)

                    c_shallow = pygame.Color(RIVER_SHALLOW_COLOR)
                    c_deep = pygame.Color(RIVER_DEEP_COLOR)
                    color = c_shallow.lerp(c_deep, water_depth)

                elif val > model.mountain_threshold:
                    factor = (val - model.mountain_threshold) / (1.0 - model.mountain_threshold)
                    c_low = pygame.Color(MOUNTAIN_LOW_COLOR)
                    c_high = pygame.Color(MOUNTAIN_HIGH_COLOR)
                    color = c_low.lerp(c_high, factor)
                    
                elif val > model.forest_threshold:
                    factor = (val - model.forest_threshold) / (model.mountain_threshold - model.forest_threshold)
                    c_low = pygame.Color(FOREST_LOW_COLOR)
                    c_high = pygame.Color(FOREST_HIGH_COLOR)
                    color = c_low.lerp(c_high, factor)
                    
                else:
                    factor = val / model.forest_threshold
                    c_low = pygame.Color(GRASS_LOW_COLOR)
                    c_high = pygame.Color(GRASS_HIGH_COLOR)
                    color = c_low.lerp(c_high, factor)

                rect = pygame.Rect(
                    (c * GRID_SIZE - cam_x) * zoom,
                    (r * GRID_SIZE - cam_y) * zoom,
                    GRID_SIZE * zoom + 1,
                    GRID_SIZE * zoom + 1
                )
                pygame.draw.rect(screen, color, rect)

    def draw_flow_field(self, screen, model, cam_x, cam_y, zoom):
        if not hasattr(model, 'flow_field') or model.flow_field is None:
            return

        self.overlay_surface.fill((0, 0, 0, 0))

        start_col = max(0, int(cam_x // GRID_SIZE))
        end_col = min(model.cols, int((cam_x + self.screen_w / zoom) // GRID_SIZE) + 2)
        start_row = max(0, int(cam_y // GRID_SIZE))
        end_row = min(model.rows, int((cam_y + self.screen_h / zoom) // GRID_SIZE) + 2)

        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                world_x = (c + 0.5) * GRID_SIZE
                world_y = (r + 0.5) * GRID_SIZE
                
                force = model.flow_field.get_force_at((world_x, world_y))
                norm = np.linalg.norm(force)
                if norm == 0:
                    continue

                angle = np.arctan2(force[1], force[0])
                center_sx, center_sy = self.world_to_screen((world_x, world_y), cam_x, cam_y, zoom)

                arrow_len = (GRID_SIZE * 0.4) * zoom
                head_len = arrow_len * 0.35

                start_p = (
                    int(center_sx - np.cos(angle) * (arrow_len * 0.5)),
                    int(center_sy - np.sin(angle) * (arrow_len * 0.5))
                )
                end_p = (
                    int(center_sx + np.cos(angle) * (arrow_len * 0.5)),
                    int(center_sy + np.sin(angle) * (arrow_len * 0.5))
                )

                pygame.draw.line(self.overlay_surface, FLOW_ARROW_COLOR, start_p, end_p, width=max(1, int(2 * zoom)))

                left_wing = (
                    int(end_p[0] - head_len * np.cos(angle - 0.5)),
                    int(end_p[1] - head_len * np.sin(angle - 0.5))
                )
                right_wing = (
                    int(end_p[0] - head_len * np.cos(angle + 0.5)),
                    int(end_p[1] - head_len * np.sin(angle + 0.5))
                )
                pygame.draw.line(self.overlay_surface, FLOW_ARROW_COLOR, end_p, left_wing, width=max(1, int(2 * zoom)))
                pygame.draw.line(self.overlay_surface, FLOW_ARROW_COLOR, end_p, right_wing, width=max(1, int(2 * zoom)))

        screen.blit(self.overlay_surface, (0, 0))

    def draw_agents(self, screen, model, cam_x, cam_y, zoom):
        for agent in model.agents:
            if agent.pos is None:
                continue

            rx, ry = self.world_to_screen(agent.pos, cam_x, cam_y, zoom)

            if not self.is_visible(rx, ry):
                continue

            if isinstance(agent, Obstacle):
                radius = getattr(agent, 'radius', 8.0) * zoom
                
                if agent.obstacle_type == "rock":
                    pygame.draw.circle(screen, ROCK_OUTLINE_COLOR, (int(rx), int(ry)), max(2, int(radius)))
                    pygame.draw.circle(screen, ROCK_INTERIOR_COLOR, (int(rx), int(ry)), max(2, int(radius)), width=max(1, int(2 * zoom)))
                
                elif agent.obstacle_type == "tree":
                    pygame.draw.circle(screen, TREE_OUTLINE_COLOR, (int(rx), int(ry)), max(3, int(radius)))
                    pygame.draw.circle(screen, TREE_INTERIOR_COLOR, (int(rx), int(ry)), max(1, int(radius * 0.6)))

            elif isinstance(agent, (Migrator, Predator)):
                size = 8 * zoom
                vel = agent.velocity
                speed = np.linalg.norm(vel)
                angle = np.arctan2(vel[1], vel[0]) if speed > 0 else 0.0
                
                if isinstance(agent, Migrator):
                    if getattr(agent, 'scared', False):
                        color = SCARED_GNU_COLOR
                    elif getattr(agent, 'is_feeding', False):
                        color = FEEDING_GNU_COLOR
                    else:
                        color = GNU_COLOR

                    p1 = (rx + np.cos(angle) * size, ry + np.sin(angle) * size)
                    p2 = (rx + np.cos(angle + 2.5) * size / 2, ry + np.sin(angle + 2.5) * size / 2)
                    p3 = (rx + np.cos(angle - 2.5) * size / 2, ry + np.sin(angle - 2.5) * size / 2)
                    pygame.draw.polygon(screen, color, [p1, p2, p3])

                elif isinstance(agent, Predator):
                    p_size = 14 * zoom
                    p1 = (rx + np.cos(angle) * p_size, ry + np.sin(angle) * p_size)
                    p2 = (rx + np.cos(angle + 2.3) * p_size / 2, ry + np.sin(angle + 2.3) * p_size / 2)
                    p3 = (rx + np.cos(angle - 2.3) * p_size / 2, ry + np.sin(angle - 2.3) * p_size / 2)
                    pygame.draw.polygon(screen, PREDATOR_COLOR, [p1, p2, p3])

    def render(self, screen, model, cam_x, cam_y, zoom, show_flow_field=False):
        screen.fill((20, 20, 25))
        self.draw_terrain(screen, model, cam_x, cam_y, zoom)
        
        if show_flow_field:
            self.draw_flow_field(screen, model, cam_x, cam_y, zoom)

        self.draw_agents(screen, model, cam_x, cam_y, zoom)
        if self.flip:
            pygame.display.flip()