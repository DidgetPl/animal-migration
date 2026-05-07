import numpy as np
from environment import MazeGenerator, Obstacle
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from pathfinding import astar

GRID_SIZE = 30 # Rozmiar komórki siatki w pikselach

class Migrator(Agent):
    def __init__(self, model, path):
        super().__init__(model)
        self.path = path
        self.current_target_idx = 0
        self.max_speed = 2.0
        self.max_force = 0.2
        self.velocity = np.array([model.random.uniform(-1, 1), 1.0])
        self.pos = None

    def step(self):
        if not self.path or self.current_target_idx >= len(self.path):
            self.model.grid_to_remove.append(self)
            return

        # 1. Obliczanie siły dążenia do aktualnego punktu A* (Autostrada)
        target_grid = self.path[self.current_target_idx]
        target_pos = np.array([
            target_grid[1] * GRID_SIZE + GRID_SIZE/2,
            target_grid[0] * GRID_SIZE + GRID_SIZE/2
        ])

        dist_to_target = np.linalg.norm(target_pos - self.pos)
        if dist_to_target < 20: # Zwiększamy promień akceptacji punktu
            self.current_target_idx += 1
            return

        # Siła dążenia (Steer toward target)
        desired = (target_pos - self.pos)
        desired = (desired / np.linalg.norm(desired)) * self.max_speed
        seek_force = desired - self.velocity

        # 2. Siły Boids (Lokalne stado)
        neighbors = self.model.space.get_neighbors(self.pos, 60, False)
        migrator_neighbors = [n for n in neighbors if isinstance(n, Migrator) and n != self]

        sep = self.separation(migrator_neighbors) * 1.5
        ali = self.alignment(migrator_neighbors) * 1.0
        coh = self.cohesion(migrator_neighbors) * 0.5

        # 3. Omijanie przeszkód (Raycasting / Sensory)
        avoid = self.avoid_obstacles_sensory() * 3.0

        # Sumowanie sił
        total_force = seek_force + sep + ali + coh + avoid
        
        # Ograniczenie siły i aktualizacja prędkości
        if np.linalg.norm(total_force) > self.max_force:
            total_force = (total_force / np.linalg.norm(total_force)) * self.max_force
            
        self.velocity += total_force
        
        # Ograniczenie prędkości maksymalnej
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Ruch
        new_pos = self.pos + self.velocity
        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def avoid_obstacles_sensory(self):
        steer = np.zeros(2)
        # Pobieramy tylko bliskie przeszkody
        neighbors = self.model.space.get_neighbors(self.pos, 40, False)
        obstacles = [n for n in neighbors if isinstance(n, Obstacle)]

        for obs in obstacles:
            rect = obs.get_rect()
            # Jeśli agent zbliża się do prostokąta, generuj siłę odpychającą od jego krawędzi
            diff = self.pos - np.array([rect.centerx, rect.centery])
            dist = np.linalg.norm(diff)
            if dist < 50:
                steer += (diff / (dist**2))
        return steer

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
        desired = (avg_vel / np.linalg.norm(avg_vel)) * self.max_speed
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
        
        from environment import MazeGenerator
        gen = MazeGenerator(width, height, min_room_size=700)
        self.obstacles = gen.generate(self)
        
        self.rows = height // GRID_SIZE
        self.cols = width // GRID_SIZE
        self.nav_grid = np.zeros((self.rows, self.cols))
        
        for obs in self.obstacles:
            r = obs.get_rect()
            for i in range(max(0, int(r.top // GRID_SIZE)), min(self.rows, int(r.bottom // GRID_SIZE) + 1)):
                for j in range(max(0, int(r.left // GRID_SIZE)), min(self.cols, int(r.right // GRID_SIZE) + 1)):
                    self.nav_grid[i][j] = 1

        goal_node = (self.rows - 2, self.cols // 2)
        self.nav_grid[goal_node[0]][goal_node[1]] = 0

        available_starts = np.where(self.nav_grid[2] == 0)[0]
        possible_nodes = [(2, col) for col in available_starts if 1 < col < self.cols - 2]
        self.random.shuffle(possible_nodes)
        nodes_to_use = possible_nodes[:n]
        path_cache = {}

        for start_node in nodes_to_use:
            if start_node not in path_cache:
                path_cache[start_node] = astar(self.nav_grid, start_node, goal_node)
            
            path = path_cache[start_node]

            if path:
                start_pos = np.array([
                    start_node[1] * GRID_SIZE + GRID_SIZE/2, 
                    start_node[0] * GRID_SIZE + GRID_SIZE/2
                ])
                
                migrator = Migrator(self, path)
                self.space.place_agent(migrator, start_pos)

    def step(self):
        for agent in self.grid_to_remove:
            self.space.remove_agent(agent)
            self.agents.remove(agent)
        self.grid_to_remove = []
        
        for agent in list(self.agents):
            if hasattr(agent, 'step'):
                agent.step()


class Boid(Agent):
    def __init__(self, model, pos, velocity):
        super().__init__(model)
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_speed = 5

    def step(self):
        # 1. Sąsiedzi (inni agenci, w tym przeszkody)
        all_neighbors = self.model.space.get_neighbors(self.pos, 50, False)
        boid_neighbors = [n for n in all_neighbors if isinstance(n, Boid)]
        obstacles = [n for n in all_neighbors if isinstance(n, Obstacle)]

        # 2. Klasyczne zasady Boids
        if boid_neighbors:
            self.velocity += self.separation(boid_neighbors) * 1.5
            self.velocity += self.alignment(boid_neighbors) * 1.0
            self.velocity += self.cohesion(boid_neighbors) * 1.0

        # 3. Omijanie przeszkód (Traktujemy je jak bardzo silne odpychanie)
        for obs in obstacles:
            self.velocity += self.avoid_obstacle(obs) * 2.5

        # 4. Granice mapy
        margin = 50
        if self.pos[0] < margin: self.velocity[0] += 0.6
        elif self.pos[0] > self.model.width - margin: self.velocity[0] -= 0.6
        if self.pos[1] < margin: self.velocity[1] += 0.6
        elif self.pos[1] > self.model.height - margin: self.velocity[1] -= 0.6

        # 5. Aktualizacja fizyki
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        new_pos = self.pos + self.velocity
        eps = 0.01
        new_pos[0] = np.clip(new_pos[0], eps, self.model.width - eps)
        new_pos[1] = np.clip(new_pos[1], eps, self.model.height - eps)

        # Sprawdzenie czy nowa pozycja nie jest wewnątrz jakiejś przeszkody
        # (Uproszczone: jeśli wejdzie, cofnij i odbij prędkość)
        for obs in self.model.obstacles:
            if obs.get_rect().collidepoint(new_pos):
                self.velocity *= -0.5 # Odbicie
                new_pos = self.pos # Zatrzymanie
                break

        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def avoid_obstacle(self, obs):
        # Prosty wektor odpychania od środka przeszkody
        obs_center = np.array([obs.pos[0] + obs.width/2, obs.pos[1] + obs.height/2])
        diff = self.pos - obs_center
        dist = np.linalg.norm(diff)
        return (diff / dist**2) * 100 if dist > 0 else np.zeros(2)

    def separation(self, neighbors):
        steer = np.zeros(2)
        for n in neighbors:
            dist = np.linalg.norm(self.pos - n.pos)
            if 0 < dist < 20:
                steer += (self.pos - n.pos) / dist
        return steer

    def alignment(self, neighbors):
        avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
        return (avg_vel - self.velocity) * 0.1

    def cohesion(self, neighbors):
        avg_pos = np.mean([n.pos for n in neighbors], axis=0)
        return (avg_pos - self.pos) * 0.05