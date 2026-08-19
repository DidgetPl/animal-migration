import numpy as np
from simulation.boids.migrator import Migrator
from simulation.boids.obstacle import Obstacle
from simulation.boids.predator import Predator


class DummyModel:
    def __init__(self, metadata):
        self.cols = metadata["cols"]
        self.rows = metadata["rows"]
        self.mountain_threshold = metadata["mountain_threshold"]
        self.forest_threshold = metadata["forest_threshold"]
        self.terrain_height = metadata["terrain_height"]
        self.river_map = metadata.get("river_map")
        self.agents = []

    def update_frame(self, frame_agent_list):
        self.agents = []
        for a in frame_agent_list:
            a_type = a["type"]

            if a_type == 0:
                agent_obj = Migrator.__new__(Migrator)
            elif a_type == 1:
                agent_obj = Predator.__new__(Predator)
            else:
                agent_obj = Obstacle.__new__(Obstacle)

            agent_obj.pos = np.array([a["x"], a["y"]])
            agent_obj.velocity = np.array([np.cos(a["angle"]), np.sin(a["angle"])])

            if a_type in (2, 3):
                agent_obj.obstacle_type = "rock" if a_type == 2 else "tree"
                agent_obj.radius = a["flags"] / 10.0
            else:
                agent_obj.scared = bool(a["flags"] & 1)
                agent_obj.is_feeding = bool(a["flags"] & 2)

            self.agents.append(agent_obj)