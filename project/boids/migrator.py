import numpy as np
from boids.base_boid import BaseBoid
from boids.obstacle import Obstacle
from variables import GRID_SIZE


class Migrator(BaseBoid):
    def __init__(self, model):
        super().__init__(model)
        self.max_speed = 3.0
        self.max_force = 0.2 #TODO: rozpisać to
        self.velocity = np.array([model.random.uniform(-1, 1), -1.0], dtype=np.float32)
        
        self.scared = False
        self.predators = []
        self.last_known_predator_pos = None
        self.memory_timer = 0

        self.hunger = model.random.uniform(10, 40)
        self.max_hunger = 100.0
        self.hunger_rate = 0.035
        self.is_feeding = False

        self.merit_score = 1.0
        self.age = 0
        self.last_y = None
        self.stuck_counter = 0

    @property
    def is_predator(self) -> bool:
        return False

    def _get_grid_coords(self):
        gx = int(np.clip(self.pos[0] // GRID_SIZE, 0, self.model.cols - 1))
        gy = int(np.clip(self.pos[1] // GRID_SIZE, 0, self.model.rows - 1))
        return gx, gy

    def _update_terrain_and_speed(self, grid_x, grid_y):
        terrain_cost = self.model.terrain_cost_map[grid_y][grid_x]
        current_max_speed = self.max_speed * (1.0 / terrain_cost)
        return terrain_cost, current_max_speed

    def _update_hunger_and_feeding(self, grid_x, grid_y, terrain_cost, migrator_neighbors, current_max_speed):
        current_grass = getattr(self.model, 'grass_map', np.ones((self.model.rows, self.model.cols)))[grid_y][grid_x]
        
        self.hunger = min(self.max_hunger, self.hunger + self.hunger_rate)
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
            current_max_speed = self.max_speed * 0.3
            self.is_feeding = False
            self.velocity[1] += 0.02

        return current_max_speed

    def _calculate_predator_and_flee_forces(self, target_pos, migrator_neighbors, current_max_speed):
        flee_force = np.zeros(2, dtype=np.float32)

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
                perpendicular = np.array([-line_to_target[1], line_to_target[0]], dtype=np.float32)
                
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
        memory_force = np.zeros(2, dtype=np.float32)
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

    def _is_scout_at_front(self, migrator_neighbors):
        if not migrator_neighbors:
            return True

        speed = np.linalg.norm(self.velocity)
        if speed == 0:
            return False

        heading = self.velocity / speed
        for other in migrator_neighbors:
            vec_to_other = other.pos - self.pos
            dist = np.linalg.norm(vec_to_other)
            if 0 < dist < 50:
                if np.dot(heading, vec_to_other / dist) > 0.5:
                    return False
        return True

    def avoid_obstacles(self, neighbors):
        avoid_force = np.zeros(2, dtype=np.float32)
        obstacles = [n for n in neighbors if isinstance(n, Obstacle)]
        
        if not obstacles:
            return avoid_force

        look_ahead = 15.0

        for obstacle in obstacles:
            to_obstacle = obstacle.pos - self.pos
            dist = np.linalg.norm(to_obstacle)
            effective_dist = dist - obstacle.radius

            if effective_dist < look_ahead:
                heading = self.velocity / (np.linalg.norm(self.velocity) + 1e-5)
                dot = np.dot(heading, to_obstacle)

                if dot > 0:
                    repulsion = -to_obstacle / (dist + 1e-5)
                    strength = (look_ahead - max(0.0, effective_dist)) / look_ahead
                    avoid_force += repulsion * (strength ** 2) * self.max_speed * 2.0

        return avoid_force

    def _update_merit_score(self, migrator_neighbors, terrain_cost, grid_x, grid_y):
        self.age += 1
        
        is_in_river = getattr(self.model, 'river_map', np.zeros((self.model.rows, self.model.cols)))[grid_y][grid_x]
        is_in_forest = terrain_cost > 1.5

        self.merit_score *= 0.999

        if self.last_y is None:
            self.last_y = self.pos[1]
            return

        delta_y = self.last_y - self.pos[1]
        self.last_y = self.pos[1]
        speed = np.linalg.norm(self.velocity)

        if delta_y > 0.05:
            is_scout = self._is_scout_at_front(migrator_neighbors)
            terrain_multiplier = 3.0 if (is_in_river or is_in_forest) else 1.0
            bonus = (0.04 if is_scout else 0.015) * terrain_multiplier
            
            self.merit_score += bonus
            self.stuck_counter = 0

        elif speed < 0.2 and not self.is_feeding and not is_in_river and not is_in_forest:
            self.stuck_counter += 1
            if self.stuck_counter > 40:
                self.merit_score -= 0.03
        else:
            if not is_in_river and not is_in_forest:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        if (is_in_river or is_in_forest) and speed > 0.1:
            self.merit_score += 0.01

        self.merit_score = np.clip(self.merit_score, 0.1, 5.0)

    def _apply_movement_and_physics(self, flow_vector, migrator_neighbors, flee_force, memory_force, obstacle_force, current_max_speed):
        desired = flow_vector * current_max_speed
        seek_force = desired - self.velocity

        grid_x, grid_y = self._get_grid_coords()
        terrain_cost = self.model.terrain_cost_map[grid_y][grid_x]
        is_in_forest = terrain_cost > 1.5

        cohesion_weight = 0.3 if is_in_forest else 0.6

        sep = self.separation(migrator_neighbors) * 2.2
        ali = self.alignment(migrator_neighbors) * 1.1
        coh = self.cohesion(migrator_neighbors) * cohesion_weight

        time_seed = (self.age * 0.05) + (id(self) % 100)
        lateral_wiggle = np.array([np.sin(time_seed) * 0.25, 0.0], dtype=np.float32)

        my_merit = self.merit
        leadership_factor = np.clip(my_merit, 0.85, 1.3)

        if self.scared:
            total_force = seek_force * 0.2 + sep + ali + coh + flee_force * 3.5 + obstacle_force * 3.0
        elif self.is_feeding:
            total_force = seek_force * 0.05 + sep * 2.5 + ali * 0.1 + coh * 1.2 + obstacle_force * 3.0
        else:
            total_force = (seek_force * leadership_factor) + sep + ali + coh + memory_force * 1.5 + obstacle_force * 3.0 + lateral_wiggle
        
        speed_ratio = current_max_speed / self.max_speed
        effective_max_force = self.max_force * speed_ratio

        if np.linalg.norm(total_force) > effective_max_force:
            total_force = (total_force / np.linalg.norm(total_force)) * effective_max_force
            
        self.velocity += total_force
        
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > current_max_speed and current_speed > 0:
            self.velocity = (self.velocity / current_speed) * current_max_speed

        new_pos = self.pos + self.velocity

        if new_pos[0] <= 5 or new_pos[0] >= self.model.width - 5:
            self.velocity[0] = 0.0

        if new_pos[1] >= self.model.height - 5:
            self.velocity[1] = min(0.0, self.velocity[1])

        new_pos[0] = np.clip(new_pos[0], 0, self.model.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.model.height - 1)

        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def step(self):
        if self.pos[1] < 20:
            self.model.grid_to_remove.append(self)
            return

        grid_x, grid_y = self._get_grid_coords()
        terrain_cost, current_max_speed = self._update_terrain_and_speed(grid_x, grid_y)

        neighbors = self.model.space.get_neighbors(self.pos, 60, False)
        self.predators = [n for n in neighbors if getattr(n, 'is_predator', False)]
        migrator_neighbors = [n for n in neighbors if isinstance(n, Migrator) and not getattr(n, 'is_predator', False) and n != self]

        current_max_speed = self._update_hunger_and_feeding(grid_x, grid_y, terrain_cost, migrator_neighbors, current_max_speed)
        self.scared = False
        current_max_speed = self._handle_river_effects(grid_x, grid_y, current_max_speed)

        self._update_merit_score(migrator_neighbors, terrain_cost, grid_x, grid_y)

        flow_vector = self.model.flow_field.get_force_at(self.pos)
        obstacle_force = self.avoid_obstacles(neighbors)

        flee_force = self._calculate_predator_and_flee_forces(self.pos, migrator_neighbors, current_max_speed)
        memory_force = self._calculate_memory_force(current_max_speed)

        self._apply_movement_and_physics(flow_vector, migrator_neighbors, flee_force, memory_force, obstacle_force, current_max_speed)

    @property
    def merit(self):
        satiation = 1.0 - (self.hunger / self.max_hunger)
        return np.clip(self.merit_score * (0.6 + 0.4 * satiation), 0.1, 5.0)

    def separation(self, neighbors):
        steer = np.zeros(2, dtype=np.float32)
        for n in neighbors:
            diff = self.pos - n.pos
            dist = np.linalg.norm(diff)
            if 0 < dist < 25:
                steer += diff / dist
        return steer

    def alignment(self, neighbors):
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        weighted_velocity_sum = np.zeros(2, dtype=np.float32)
        total_weight = 0.0

        for neighbor in neighbors:
            w = getattr(neighbor, 'merit', 1.0)
            weighted_velocity_sum += neighbor.velocity * w
            total_weight += w

        if total_weight > 0:
            avg_velocity = weighted_velocity_sum / total_weight
            norm = np.linalg.norm(avg_velocity)
            if norm > 0:
                return (avg_velocity / norm) * self.max_speed - self.velocity
        return np.zeros(2, dtype=np.float32)

    def cohesion(self, neighbors):
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        weighted_pos_sum = np.zeros(2, dtype=np.float32)
        total_weight = 0.0

        for neighbor in neighbors:
            w = getattr(neighbor, 'merit', 1.0)
            weighted_pos_sum += neighbor.pos * w
            total_weight += w

        if total_weight > 0:
            center_of_mass = weighted_pos_sum / total_weight
            desired = center_of_mass - self.pos
            norm = np.linalg.norm(desired)
            if norm > 0:
                return (desired / norm) * self.max_speed - self.velocity
        return np.zeros(2, dtype=np.float32)