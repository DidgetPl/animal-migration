import numpy as np
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from noise import pnoise2
from pathfinding import astar

GRID_SIZE = 30

class Migrator(Agent):
    def __init__(self, model, path):
        super().__init__(model)
        self.path = path
        self.current_target_idx = 0
        self.max_speed = 3.0
        self.max_force = 0.2
        self.velocity = np.array([model.random.uniform(-1, 1), 1.0])
        self.pos = None

    def step(self):
        if not self.path or self.current_target_idx >= len(self.path):
            self.model.grid_to_remove.append(self)
            return

        grid_x = int(self.pos[0] // GRID_SIZE)
        grid_y = int(self.pos[1] // GRID_SIZE)
        
        grid_x = np.clip(grid_x, 0, self.model.cols - 1)
        grid_y = np.clip(grid_y, 0, self.model.rows - 1)
        
        terrain_cost = self.model.terrain_map[grid_y][grid_x]
        current_max_speed = self.max_speed * (1.0 / terrain_cost)

        target_grid = self.path[self.current_target_idx]
        target_pos = np.array([
            target_grid[1] * GRID_SIZE + GRID_SIZE/2, 
            target_grid[0] * GRID_SIZE + GRID_SIZE/2
        ])

        dist_to_target = np.linalg.norm(target_pos - self.pos)
        if dist_to_target < 20:
            self.current_target_idx += 1
            return

        desired = (target_pos - self.pos)
        desired = (desired / np.linalg.norm(desired)) * current_max_speed
        seek_force = desired - self.velocity

        neighbors = self.model.space.get_neighbors(self.pos, 60, False)
        migrator_neighbors = [n for n in neighbors if isinstance(n, Migrator) and n != self]

        sep = self.separation(migrator_neighbors) * 1.5
        ali = self.alignment(migrator_neighbors) * 1.0
        coh = self.cohesion(migrator_neighbors) * 0.5

        total_force = seek_force + sep + ali + coh
        
        if np.linalg.norm(total_force) > self.max_force:
            total_force = (total_force / np.linalg.norm(total_force)) * self.max_force
            
        self.velocity += total_force
        
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > current_max_speed and current_speed > 0:
            self.velocity = (self.velocity / current_speed) * current_max_speed

        new_pos = self.pos + self.velocity
        new_pos[0] = np.clip(new_pos[0], 0, self.model.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.model.height - 1)

        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def separation(self, neighbors):
        steer = np.zeros(2)
        for n in neighbors:
            diff = self.pos - n.pos
            dist = np.linalg.norm(diff)
            if 0 < dist < 25:
                steer += diff / dist
        return steer

    def alignment(self, neighbors):
        if not neighbors: return np.zeros(2)
        avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
        norm = np.linalg.norm(avg_vel)
        desired = (avg_vel / norm) * self.max_speed if norm > 0 else np.zeros(2)
        return desired - self.velocity

    def cohesion(self, neighbors):
        if not neighbors: return np.zeros(2)
        avg_pos = np.mean([n.pos for n in neighbors], axis=0)
        desired = (avg_pos - self.pos)
        dist = np.linalg.norm(desired)
        if dist > 0:
            desired = (desired / dist) * self.max_speed
        return desired - self.velocity

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
        scale = 10.0
        octaves = 4
        seed = self.random.randint(0, 1000)
        
        for i in range(self.rows):
            for j in range(self.cols):
                noise_val = pnoise2(i/scale, j/scale, octaves=octaves, base=seed)
                val = (noise_val + 1) / 2
                
                if val > 0.65: self.terrain_map[i][j] = 10.0
                elif val > 0.55: self.terrain_map[i][j] = 5.0
                else: self.terrain_map[i][j] = 1.0

                self.terrain_height[i][j] = val

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

    def step(self):
        for agent in self.grid_to_remove:
            if agent in self.agents:
                self.space.remove_agent(agent)
                agent.remove()
        self.grid_to_remove = []
        
        for agent in list(self.agents):
            if isinstance(agent, Migrator):
                agent.step()