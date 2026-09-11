from mesa import Agent


class Obstacle(Agent):
    def __init__(self, model, obstacle_type="rock", radius=8.0):
        super().__init__(model)
        self.obstacle_type = obstacle_type
        self.radius = radius

    def step(self):
        pass