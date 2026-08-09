import numpy as np
from mesa import Agent


class BaseBoid(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.pos = None
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = 3.0

    @property
    def is_predator(self) -> bool:
        raise NotImplementedError("Brak zaimplementowanej metody w podklasie")