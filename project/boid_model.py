import numpy as np
from boids.migrator import Migrator
from boids.predator import Predator
from mesa import Model
from mesa.space import ContinuousSpace
from noise import snoise2
from pathfinding import astar
from variables import GRID_SIZE


class BoidModel(Model):
    def __init__(self, n, width, height, num_obstacles=15):
        super().__init__()
        self.width, self.height = width, height
        self.space = ContinuousSpace(width, height, False)
        self.grid_to_remove = []

        self.rows = height // GRID_SIZE
        self.cols = width // GRID_SIZE
        self.terrain_map = np.zeros((self.rows, self.cols))
        self.terrain_height = np.zeros((self.rows, self.cols))

        self.mountain_threshold = 0.56
        self.forest_threshold = 0.28
        
        seed = self.random.uniform(0.0, 1000.0)

        self.grass_map = np.ones((self.rows, self.cols), dtype=float)
        self.grass_regrowth_rate = 0.001

        for i in range(self.rows):
            for j in range(self.cols):
                nx = i / 18.0
                ny = j / 18.0
                
                v1 = snoise2(nx, ny, octaves=1, base=int(seed))
                
                v2 = snoise2(nx * 3.5, ny * 3.5, octaves=2, base=int(seed) + 1)
                
                v3 = snoise2(nx * 8.0, ny * 8.0, octaves=1, base=int(seed) + 2)
                
                total_noise = (1.0 * v1) + (0.35 * v2) + (0.1 * v3)
                
                val = (total_noise + 1.45) / 2.9
                val = np.clip(val, 0.0, 1.0)
                
                val = val ** 2.2
                
                self.terrain_height[i][j] = val

                if val > self.mountain_threshold:
                    self.terrain_map[i][j] = 7.0
                elif val > self.forest_threshold:
                    self.terrain_map[i][j] = 3.5
                else: 
                    self.terrain_map[i][j] = 1.0


        self.obstacles = []

        goal_node = (self.rows - 2, self.cols // 2)
        self.terrain_map[goal_node[0]][goal_node[1]] = 1.0

        available_starts = np.where(self.terrain_map[2] == 1.0)[0]
        if len(available_starts) == 0:
            available_starts = np.where(self.terrain_map[2] < 10.0)[0]
            
        possible_nodes = [(2, col) for col in available_starts if 1 < col < self.cols - 2]
        self.random.shuffle(possible_nodes)
        nodes_to_use = possible_nodes[:n]
        path_cache = {}

        for start_node in nodes_to_use:
            if start_node not in path_cache:
                path_cache[start_node] = astar(self.terrain_map, start_node, goal_node)
            
            path = path_cache[start_node]

            if path:
                start_pos = np.array([
                    start_node[1] * GRID_SIZE + GRID_SIZE/2,
                    start_node[0] * GRID_SIZE + GRID_SIZE/2
                ])
                
                migrator = Migrator(self, path)
                self.space.place_agent(migrator, start_pos)
                if migrator not in self.agents:
                    self.agents.add(migrator)

        self.num_predators = 7
        for _ in range(self.num_predators):
            rx = self.random.uniform(100, self.width - 100)
            ry = self.random.uniform(self.height * 0.2, self.height * 0.8)

            predator = Predator(self, [rx, ry])
            self.space.place_agent(predator, [rx, ry])
            self.agents.add(predator)

    def step_environment(self):
        self.grass_map = np.clip(
            self.grass_map + self.grass_regrowth_rate, 0.0, 1.0
        )

    def step(self):
        for agent in self.grid_to_remove:
            if agent in self.agents:
                self.space.remove_agent(agent)
                agent.remove()
        self.grid_to_remove = []

        for agent in list(self.agents):
            if isinstance(agent, Migrator) or isinstance(agent, Predator):
                agent.step()