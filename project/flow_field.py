import numpy as np
from variables import GRID_SIZE


class MigrationFlowField:
    def __init__(self, model):
        self.model = model
        self.rows = model.rows
        self.cols = model.cols
        self.field = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        
        self.w_north = 1.0
        self.w_grass = 1.8
        self.w_terrain = 2.5
        self.w_boundary = 3.5

    def update_field(self):
        grass_map = getattr(self.model, 'grass_map', np.ones((self.rows, self.cols)))
        terrain_map = self.model.terrain_cost_map

        gy, gx = np.gradient(grass_map)
        ty, tx = np.gradient(terrain_map)

        margin_x = 4
        margin_y = 5

        for r in range(self.rows):
            for c in range(self.cols):
                f_north = np.array([0.0, -1.0])

                f_grass = np.array([gx[r, c], gy[r, c]])
                norm_g = np.linalg.norm(f_grass)
                if norm_g > 0:
                    f_grass /= norm_g

                f_terrain = np.array([-tx[r, c], -ty[r, c]])
                norm_t = np.linalg.norm(f_terrain)
                if norm_t > 0:
                    f_terrain /= norm_t

                if terrain_map[r, c] > 3.0:
                    f_terrain *= (terrain_map[r, c] / 2.0)

                f_boundary = np.array([0.0, 0.0])

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