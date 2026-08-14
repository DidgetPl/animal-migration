import numpy as np
from boids.obstacle import Obstacle
from variables import GRID_SIZE


class MigrationFlowField:
    def __init__(self, model):
        self.model = model
        self.rows = model.rows
        self.cols = model.cols
        self.field = np.zeros((self.rows, self.cols, 2), dtype=np.float32)

        self.w_north = 1.0
        self.w_grass = 0.8
        self.w_terrain = 3.0
        self.w_boundary = 3.5

    def update_field(self):
        grass_map = getattr(self.model, 'grass_map', np.ones((self.rows, self.cols)))
        
        effective_terrain_map = np.copy(self.model.terrain_cost_map)

        if hasattr(self.model, 'agents'):
            for agent in self.model.agents:
                if isinstance(agent, Obstacle):
                    gx = int(agent.pos[0] // GRID_SIZE)
                    gy = int(agent.pos[1] // GRID_SIZE)
                    
                    radius_cells = max(1, int((agent.radius + 15.0) // GRID_SIZE))
                    
                    r_min, r_max = max(0, gy - radius_cells), min(self.rows, gy + radius_cells + 1)
                    c_min, c_max = max(0, gx - radius_cells), min(self.cols, gx + radius_cells + 1)
                    
                    for r in range(r_min, r_max):
                        for c in range(c_min, c_max):
                            dist = np.hypot(c - gx, r - gy) * GRID_SIZE
                            if dist < agent.radius + 20.0:
                                effective_terrain_map[r, c] += 12.0

        gy, gx = np.gradient(grass_map)
        ty, tx = np.gradient(effective_terrain_map)

        margin_x = 4
        margin_y = 5

        f_north = np.array([0.0, -1.0], dtype=np.float32)

        for r in range(self.rows):
            for c in range(self.cols):
                f_grass = np.array([gx[r, c], gy[r, c]], dtype=np.float32)
                norm_g = np.linalg.norm(f_grass)
                if norm_g > 0:
                    f_grass /= norm_g

                f_terrain = np.array([-tx[r, c], -ty[r, c]], dtype=np.float32)
                norm_t = np.linalg.norm(f_terrain)
                if norm_t > 0:
                    f_terrain /= norm_t

                if effective_terrain_map[r, c] > 2.0:
                    f_terrain *= (effective_terrain_map[r, c] / 2.0)

                f_boundary = np.array([0.0, 0.0], dtype=np.float32)

                if c < margin_x:
                    dist_factor = (margin_x - c) / margin_x
                    f_boundary[0] += dist_factor ** 2
                elif c >= self.cols - margin_x:
                    dist_factor = (c - (self.cols - margin_x - 1)) / margin_x
                    f_boundary[0] -= dist_factor ** 2

                if r >= self.rows - margin_y:
                    dist_factor = (r - (self.rows - margin_y - 1)) / margin_y
                    f_boundary[1] -= dist_factor ** 2

                combined = (
                    self.w_north * f_north +
                    self.w_grass * f_grass +
                    self.w_terrain * f_terrain +
                    self.w_boundary * f_boundary
                )

                norm_c = np.linalg.norm(combined)
                if norm_c > 0:
                    combined /= norm_c

                self.field[r, c] = combined

    def get_force_at(self, pos):
        grid_x = int(np.clip(pos[0] // GRID_SIZE, 0, self.cols - 1))
        grid_y = int(np.clip(pos[1] // GRID_SIZE, 0, self.rows - 1))
        return self.field[grid_y, grid_x]