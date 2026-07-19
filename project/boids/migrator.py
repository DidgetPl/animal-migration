import numpy as np
from boids.base_boid import BaseBoid
from variables import GRID_SIZE


class Migrator(BaseBoid):
    def __init__(self, model, path):
        super().__init__(model)
        self.path = path
        self.current_target_idx = 0
        self.max_speed = 3.0
        self.max_force = 0.2
        self.velocity = np.array([model.random.uniform(-1, 1), 1.0])
        self.scared = False
        self.predators = []
        self.last_known_predator_pos = None
        self.memory_timer = 0

        self.hunger = model.random.uniform(0, 30)
        self.hunger_rate = 0.08
        self.is_feeding = False

    @property
    def is_predator(self) -> bool:
        return False

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
        
        self.predators = [n for n in neighbors if getattr(n, 'is_predator', False)]
        migrator_neighbors = [n for n in neighbors if not getattr(n, 'is_predator', False) and n != self]

        self.hunger = min(100.0, self.hunger + self.hunger_rate)
        is_fertile_ground = (terrain_cost == 1.0)

        eating_neighbors = [n for n in migrator_neighbors if getattr(n, 'is_feeding', False)]
        
        if self.hunger > 40.0 and is_fertile_ground:
            if self.hunger > 75.0 or len(eating_neighbors) >= 2:
                self.is_feeding = True
        
        if self.hunger <= 0.0:
            self.hunger = 0.0
            self.is_feeding = False

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
            self.is_feeding = False

            haunting_predators = [p for p in self.predators if getattr(p, 'is_hunting', False)]
            closest_predator = min(self.predators if not haunting_predators else haunting_predators,
                                   key=lambda p: np.linalg.norm(p.pos - self.pos))
            
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
                    n.is_feeding = False
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

        if self.is_feeding and not self.scared:
            current_max_speed *= 0.15 
            self.hunger = max(0.0, self.hunger - 0.4) 

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
        elif self.is_feeding:
            total_force = seek_force * 0.1 + sep * 3.0 + ali * 0.2 + coh * 2.0
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