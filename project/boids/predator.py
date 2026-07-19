import numpy as np
from boids.base_boid import BaseBoid


class Predator(BaseBoid):
    def __init__(self, model, pos):
        super().__init__(model)
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 4.5
        self.detection_radius = 150.0

        self.hunger = model.random.uniform(0, 40)
        self.hunger_rate = 0.15
        self.is_hunting = False

    @property
    def is_predator(self) -> bool:
        return True

    def step(self):
        self.hunger = min(100.0, self.hunger + self.hunger_rate)

        if self.hunger > 50.0:
            self.is_hunting = True
        elif self.hunger < 10.0:
            self.is_hunting = False

        neighbors = self.model.space.get_neighbors(self.pos, self.detection_radius, False)
        
        migrators = [n for n in neighbors if not getattr(n, 'is_predator', True)]

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