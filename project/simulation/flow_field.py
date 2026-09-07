import collections
import heapq

import numpy as np
from simulation.boids.obstacle import Obstacle
from variables import GRID_SIZE


class MigrationFlowField:
    def __init__(self, model):
        self.model = model
        self.rows = model.rows
        self.cols = model.cols
        self.field = np.zeros((self.rows, self.cols, 2), dtype=np.float32)

    def update_field(self):
        cost_map = np.copy(self.model.terrain_cost_map).astype(np.float32)

        if hasattr(self.model, 'grass_map'):
            cost_map += (1.0 - self.model.grass_map) * 0.5

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
                                cost_map[r, c] += 15.0

        margin_x, margin_y = 4, 5
        for r in range(self.rows):
            for c in range(self.cols):
                if c < margin_x:
                    cost_map[r, c] += (margin_x - c) * 3.0
                elif c >= self.cols - margin_x:
                    cost_map[r, c] += (c - (self.cols - margin_x - 1)) * 3.0
                if r >= self.rows - margin_y:
                    cost_map[r, c] += (r - (self.rows - margin_y - 1)) * 3.0

        dist_map = np.full((self.rows, self.cols), fill_value=1e9, dtype=np.float32)
        pq = []

        for c in range(self.cols):
            dist_map[0, c] = 0.0
            heapq.heappush(pq, (0.0, 0, c))

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

        while pq:
            d, r, c = heapq.heappop(pq)

            if d > dist_map[r, c]:
                continue

            for dr, dc, weight in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    move_cost = 0.5 * (cost_map[r, c] + cost_map[nr, nc]) * weight
                    new_d = d + move_cost

                    if new_d < dist_map[nr, nc]:
                        dist_map[nr, nc] = new_d
                        heapq.heappush(pq, (new_d, nr, nc))

        dy, dx = np.gradient(dist_map)

        fx = -dx
        fy = -dy

        combined = np.stack((fx, fy), axis=-1)

        norms = np.linalg.norm(combined, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        self.field = combined / norms

    def get_force_at(self, pos):
        grid_x = int(np.clip(pos[0] // GRID_SIZE, 0, self.cols - 1))
        grid_y = int(np.clip(pos[1] // GRID_SIZE, 0, self.rows - 1))
        return self.field[grid_y, grid_x]