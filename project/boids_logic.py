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
        self.scared = False
        self.predators = []
        self.last_known_predator_pos = None
        self.memory_timer = 0

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

        neighbors = self.model.space.get_neighbors(self.pos, 80, False)
        self.predators = [n for n in neighbors if isinstance(n, Predator)]
        migrator_neighbors = [n for n in neighbors if isinstance(n, Migrator) and n != self]

        self.scared = False
        flee_force = np.zeros(2)

        target_grid = self.path[self.current_target_idx]
        target_pos = np.array([
            target_grid[1] * GRID_SIZE + GRID_SIZE/2, 
            target_grid[0] * GRID_SIZE + GRID_SIZE/2
        ])

        look_ahead = min(len(self.path), self.current_target_idx + 8)
        for idx in range(self.current_target_idx + 1, look_ahead):
            chk_grid = self.path[idx]
            chk_pos = np.array([chk_grid[1] * GRID_SIZE + GRID_SIZE/2, chk_grid[0] * GRID_SIZE + GRID_SIZE/2])
            if np.linalg.norm(chk_pos - self.pos) < 40:
                self.current_target_idx = idx
                break

        if self.predators:
            self.scared = True
            closest_predator = min(self.predators, key=lambda p: np.linalg.norm(p.pos - self.pos))
            
            self.last_known_predator_pos = np.copy(closest_predator.pos)
            self.memory_timer = 120

            line_to_target = target_pos - closest_predator.pos
            norm_target = np.linalg.norm(line_to_target)
            
            if norm_target > 0:
                line_to_target /= norm_target
                perpendicular = np.array([-line_to_target[1], line_to_target[0]])
                
                agent_vec = self.pos - closest_predator.pos
                dot_product = np.dot(agent_vec, perpendicular)
                if dot_product < 0:
                    perpendicular *= -1
                
                desired_flee = (line_to_target * 0.5 + perpendicular * 1.2)
                norm_flee = np.linalg.norm(desired_flee)
                if norm_flee > 0:
                    desired_flee = (desired_flee / norm_flee) * (current_max_speed * 1.4)
                
                flee_force = desired_flee - self.velocity
            
            for n in migrator_neighbors:
                if np.linalg.norm(n.pos - self.pos) < 60:
                    n.scared = True
                    n.velocity += flee_force * 0.3
                    if not n.predators:
                        n.last_known_predator_pos = np.copy(closest_predator.pos)
                        n.memory_timer = 90
        
        memory_force = np.zeros(2)
        if not self.predators and self.last_known_predator_pos is not None and self.memory_timer > 0:
            self.memory_timer -= 1
            dist_to_danger_zone = np.linalg.norm(self.pos - self.last_known_predator_pos)
            
            if dist_to_danger_zone < 130:
                diff = self.pos - self.last_known_predator_pos
                if dist_to_danger_zone > 0:
                    memory_force = (diff / dist_to_danger_zone) * current_max_speed * 1.2
            else:
                if self.memory_timer <= 0:
                    self.last_known_predator_pos = None

        dist_to_target = np.linalg.norm(target_pos - self.pos)
        if dist_to_target < 20:
            self.current_target_idx += 1
            return

        desired = (target_pos - self.pos)
        desired = (desired / np.linalg.norm(desired)) * current_max_speed
        seek_force = desired - self.velocity

        sep = self.separation(migrator_neighbors) * 2.0  
        ali = self.alignment(migrator_neighbors) * 1.5
        coh = self.cohesion(migrator_neighbors) * 0.6

        if self.scared:
            total_force = seek_force * 0.2 + sep + ali + coh + flee_force * 3.5
        else:
            total_force = seek_force + sep + ali + coh + memory_force * 2.0
        
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

class Predator(Agent):
    def __init__(self, model, pos):
        super().__init__(model)
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 4.5
        self.detection_radius = 150.0

        self.hunger = model.random.uniform(0, 40)
        self.hunger_rate = 0.15
        self.is_hunting = False

    def step(self):
        self.hunger = min(100.0, self.hunger + self.hunger_rate)

        if self.hunger > 50.0:
            self.is_hunting = True
        elif self.hunger < 10.0:
            self.is_hunting = False

        neighbors = self.model.space.get_neighbors(self.pos, self.detection_radius, False)
        migrators = [n for n in neighbors if isinstance(n, Migrator)]

        if migrators and self.is_hunting:
            closest_prey = min(migrators, key=lambda m: np.linalg.norm(m.pos - self.pos))
            direction = closest_prey.pos - self.pos
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                self.velocity = (direction / dist) * self.max_speed
                
            if dist < 12:
                self.model.grid_to_remove.append(closest_prey)
                self.hunger = 0.0
                self.is_hunting = False
                if self.model.random.random() < 0.05:
                    self.velocity = np.array([self.model.random.uniform(-1, 1), self.model.random.uniform(-1, 1)])
                    self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * (self.max_speed * 0.3)
        else:
            if self.model.random.random() < 0.05:
                self.velocity = np.array([self.model.random.uniform(-1, 1), self.model.random.uniform(-1, 1)])
                self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * (self.max_speed * 0.3)

        new_pos = self.pos + self.velocity
        new_pos[0] = np.clip(new_pos[0], 0, self.model.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.model.height - 1)
        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

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

        self.num_predators = 12
        for _ in range(self.num_predators):
            rx = self.random.uniform(100, self.width - 100)
            ry = self.random.uniform(self.height * 0.2, self.height * 0.8)

            predator = Predator(self, [rx, ry])
            self.space.place_agent(predator, [rx, ry])
            self.agents.add(predator)

    def step(self):
        for agent in self.grid_to_remove:
            if agent in self.agents:
                self.space.remove_agent(agent)
                agent.remove()
        self.grid_to_remove = []

        for agent in list(self.agents):
            if isinstance(agent, Migrator) or isinstance(agent, Predator):
                agent.step()