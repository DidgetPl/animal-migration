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

        self.hunger = model.random.uniform(10, 40)
        self.hunger_rate = 0.035
        self.is_feeding = False

    @property
    def is_predator(self) -> bool:
        return False

    def _get_grid_coords(self):
        gx = int(self.pos[0] // GRID_SIZE)
        gy = int(self.pos[1] // GRID_SIZE)
        gx = np.clip(gx, 0, self.model.cols - 1)
        gy = np.clip(gy, 0, self.model.rows - 1)
        return gx, gy

    def _update_terrain_and_speed(self, grid_x, grid_y):
        terrain_cost = self.model.terrain_cost_map[grid_y][grid_x]
        current_max_speed = self.max_speed * (1.0 / terrain_cost)
        return terrain_cost, current_max_speed

    def _update_hunger_and_feeding(self, grid_x, grid_y, terrain_cost, migrator_neighbors, current_max_speed):
        current_grass = getattr(self.model, 'grass_map', np.ones((self.model.rows, self.model.cols)))[grid_y][grid_x]
        
        self.hunger = min(100.0, self.hunger + self.hunger_rate)
        is_fertile_ground = (terrain_cost == 1.0) and (current_grass > 0.3)
        eating_neighbors = [n for n in migrator_neighbors if getattr(n, 'is_feeding', False)]

        if (self.hunger > 40.0 or len(eating_neighbors) >= 2) and is_fertile_ground:
            self.is_feeding = True

        if self.hunger <= 1.0 or current_grass < 0.15:
            self.is_feeding = False
            if self.hunger <= 1.0:
                self.hunger = 0.0

        if self.is_feeding and not self.scared:
            eaten_amount = min(current_grass, 0.006)
            if hasattr(self.model, 'grass_map'):
                self.model.grass_map[grid_y][grid_x] -= eaten_amount
            
            nutrition_value = eaten_amount * 45.0
            self.hunger = max(0.0, self.hunger - nutrition_value)
            current_max_speed *= 0.1

        return current_max_speed

    def _handle_river_effects(self, grid_x, grid_y, current_max_speed):
        is_in_river = getattr(self.model, 'river_map', np.zeros((self.model.rows, self.model.cols)))[grid_y][grid_x]
        
        if is_in_river:
            current_max_speed = self.max_speed * 0.25
            self.is_feeding = False
            self.velocity[1] += 0.03

        return current_max_speed

    def _update_path_target(self, target_pos):
        look_ahead = min(len(self.path), self.current_target_idx + 8)
        for idx in range(self.current_target_idx + 1, look_ahead):
            chk_grid = self.path[idx]
            chk_pos = np.array([chk_grid[1] * GRID_SIZE + GRID_SIZE/2, chk_grid[0] * GRID_SIZE + GRID_SIZE/2])
            if np.linalg.norm(chk_pos - self.pos) < 40:
                self.current_target_idx = idx
                break

    def _calculate_predator_and_flee_forces(self, target_pos, migrator_neighbors, current_max_speed):
        flee_force = np.zeros(2)

        if self.predators:
            self.scared = True
            self.is_feeding = False

            haunting_predators = [p for p in self.predators if getattr(p, 'is_hunting', False)]
            closest_predator = min(
                self.predators if not haunting_predators else haunting_predators,
                key=lambda p: np.linalg.norm(p.pos - self.pos)
            )
            
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

        return flee_force

    def _calculate_memory_force(self, current_max_speed):
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

        return memory_force

    def _apply_movement_and_physics(self, target_pos, migrator_neighbors, flee_force, memory_force, current_max_speed):
        desired = (target_pos - self.pos)
        norm_desired = np.linalg.norm(desired)
        desired = (desired / norm_desired) * current_max_speed if norm_desired > 0 else np.zeros(2)
        seek_force = desired - self.velocity

        sep = self.separation(migrator_neighbors) * 2.0
        ali = self.alignment(migrator_neighbors) * 1.5
        coh = self.cohesion(migrator_neighbors) * 0.6

        if self.scared:
            total_force = seek_force * 0.2 + sep + ali + coh + flee_force * 3.5
        elif self.is_feeding:
            total_force = seek_force * 0.05 + sep * 2.5 + ali * 0.1 + coh * 1.2
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

    def step(self):
        if not self.path or self.current_target_idx >= len(self.path):
            self.model.grid_to_remove.append(self)
            return

        grid_x, grid_y = self._get_grid_coords()
        terrain_cost, current_max_speed = self._update_terrain_and_speed(grid_x, grid_y)

        neighbors = self.model.space.get_neighbors(self.pos, 80, False)
        self.predators = [n for n in neighbors if getattr(n, 'is_predator', False)]
        migrator_neighbors = [n for n in neighbors if not getattr(n, 'is_predator', False) and n != self]

        current_max_speed = self._update_hunger_and_feeding(grid_x, grid_y, terrain_cost, migrator_neighbors, current_max_speed)
        self.scared = False
        current_max_speed = self._handle_river_effects(grid_x, grid_y, current_max_speed)

        target_grid = self.path[self.current_target_idx]
        target_pos = np.array([
            target_grid[1] * GRID_SIZE + GRID_SIZE / 2,
            target_grid[0] * GRID_SIZE + GRID_SIZE / 2
        ])

        if np.linalg.norm(target_pos - self.pos) < 20:
            self.current_target_idx += 1
            return

        self._update_path_target(target_pos)

        flee_force = self._calculate_predator_and_flee_forces(target_pos, migrator_neighbors, current_max_speed)
        memory_force = self._calculate_memory_force(current_max_speed)

        self._apply_movement_and_physics(target_pos, migrator_neighbors, flee_force, memory_force, current_max_speed)

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