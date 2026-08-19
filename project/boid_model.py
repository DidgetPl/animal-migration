import numpy as np
from boids.migrator import Migrator
from boids.obstacle import Obstacle
from boids.predator import Predator
from flow_field import MigrationFlowField
from mesa import Model
from mesa.space import ContinuousSpace
from noise import snoise2
from variables import GRID_SIZE


class BoidModel(Model):
    def __init__(self, num_migrators, num_predators, width, height, river_cost=3.0, forest_cost=2.0,
                grass_regrowth=0.001, num_obstacles=120, mountain_threshold=0.56, forest_threshold=0.28,
                enable_river=True, migrator_speed=3.0, river_speed_mod=0.3, river_stream=0.02, hunger_rate=0.035):
        super().__init__()
        self.width, self.height = width, height
        self.space = ContinuousSpace(width, height, False)
        self.grid_to_remove = []
        self.steps = 0

        self.rows = height // GRID_SIZE
        self.cols = width // GRID_SIZE
        self.terrain_cost_map = np.zeros((self.rows, self.cols))
        self.terrain_height = np.zeros((self.rows, self.cols))

        self.mountain_threshold = mountain_threshold
        self.forest_threshold = forest_threshold

        seed = self.random.uniform(0.0, 1000.0)

        self.grass_map = np.ones((self.rows, self.cols), dtype=float)
        self.grass_regrowth_rate = grass_regrowth

        self.river_map = np.zeros((self.rows, self.cols), dtype=bool)
        self.river_speed_mod = river_speed_mod
        self.river_stream = river_stream

        if enable_river:
            self._generate_river()
        
        for i in range(self.rows):
            for j in range(self.cols):
                nx = i / 18.0
                ny = j / 18.0
                
                v1 = snoise2(nx, ny, octaves=1, base=int(seed))
                v2 = snoise2(nx * 3.5, ny * 3.5, octaves=2, base=int(seed) + 1)
                v3 = snoise2(nx * 8.0, ny * 8.0, octaves=1, base=int(seed) + 2)
                
                total_noise = (1.0 * v1) + (0.35 * v2) + (0.1 * v3)
                val = (total_noise + 1.45) / 2.9
                val = np.clip(val, 0.0, 1.0) ** 2.2
                
                self.terrain_height[i][j] = val

                if self.river_map[i][j]:
                    self.terrain_cost_map[i][j] = river_cost
                elif val > self.mountain_threshold:
                    self.terrain_cost_map[i][j] = 8.0
                elif val > self.forest_threshold:
                    self.terrain_cost_map[i][j] = forest_cost
                else:
                    self.terrain_cost_map[i][j] = 1.0

        self.flow_field = MigrationFlowField(self)
        self.flow_field.update_field()

        available_starts = np.where(self.terrain_cost_map[self.rows - 3] < 10.0)[0]
        if len(available_starts) == 0:
            available_starts = np.arange(1, self.cols - 1)

        for _ in range(num_migrators):
            col = self.random.choice(available_starts)
            row = self.rows - 2 + self.random.uniform(-0.5, 0.5)
            
            start_pos = np.array([
                col * GRID_SIZE + GRID_SIZE / 2,
                row * GRID_SIZE + GRID_SIZE / 2
            ])
            
            migrator = Migrator(self, max_speed=migrator_speed, hunger_rate=hunger_rate)
            self.space.place_agent(migrator, start_pos)
            self.agents.add(migrator)

        self.num_predators = num_predators
        for _ in range(self.num_predators):
            rx = self.random.uniform(100, self.width - 100)
            ry = self.random.uniform(self.height * 0.2, self.height * 0.8)

            predator = Predator(self, [rx, ry])
            self.space.place_agent(predator, [rx, ry])
            self.agents.add(predator)

        self.num_obstacles = num_obstacles
        for _ in range(self.num_obstacles):
            rx = self.random.uniform(20, self.width - 20)
            ry = self.random.uniform(20, self.height - 20)
            
            grid_x = int(rx // GRID_SIZE)
            grid_y = int(ry // GRID_SIZE)

            if not self.river_map[grid_y][grid_x] and self.terrain_cost_map[grid_y][grid_x] < 5.0:
                o_type = self.random.choice(["rock", "tree"])
                radius = 6.0 if o_type == "rock" else 10.0
                
                obstacle = Obstacle(self, obstacle_type=o_type, radius=radius)
                self.space.place_agent(obstacle, [rx, ry])
                self.agents.add(obstacle)

    def step_environment(self):
        self.grass_map = np.clip(
            self.grass_map + self.grass_regrowth_rate, 0.0, 1.0
        )

    def step(self):
        self.step_environment()

        for agent in self.grid_to_remove:
            if agent in self.agents:
                self.space.remove_agent(agent)
                agent.remove()
        self.grid_to_remove = []
        
        for agent in list(self.agents):
            if isinstance(agent, (Migrator, Predator)):
                agent.step()
        
        if self.steps % 10 == 0:
            self.flow_field.update_field()
            
        self.steps += 1

    def _generate_river(self):
        river_center_x = self.cols // 2
        width = 2

        for r in range(self.rows):
            offset = int(np.sin(r * 0.1) * 6 + np.sin(r * 0.03) * 12)
            c_center = river_center_x + offset
            
            for c in range(c_center - width, c_center + width + 1):
                if 0 <= c < self.cols:
                    self.river_map[r][c] = True