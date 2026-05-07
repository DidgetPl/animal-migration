import numpy as np
from environment import MazeGenerator, Obstacle
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from pathfinding import astar

GRID_SIZE = 30 

class Migrator(Agent):
    def __init__(self, model, path):
            super().__init__(model)
            self.pos = None
            self.path = path
            self.current_target_idx = 0
            self.speed = 3.0
            self.velocity = np.array([0.0, 0.0])

    def step(self):
        if not self.path or self.current_target_idx >= len(self.path):
            self.model.grid_to_remove.append(self)
            return

        target_grid = self.path[self.current_target_idx]
        target_pos = np.array([
            target_grid[1] * GRID_SIZE + GRID_SIZE/2, 
            target_grid[0] * GRID_SIZE + GRID_SIZE/2
        ])

        direction = target_pos - self.pos
        dist = np.linalg.norm(direction)

        if dist < 5:
            self.current_target_idx += 1
        else:
            velocity = (direction / dist) * self.speed
            new_pos = self.pos + velocity
            self.model.space.move_agent(self, new_pos)
            self.pos = new_pos

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
        all_neighbors = self.model.space.get_neighbors(self.pos, 50, False)
        boid_neighbors = [n for n in all_neighbors if isinstance(n, Boid)]
        obstacles = [n for n in all_neighbors if isinstance(n, Obstacle)]

        if boid_neighbors:
            self.velocity += self.separation(boid_neighbors) * 1.5
            self.velocity += self.alignment(boid_neighbors) * 1.0
            self.velocity += self.cohesion(boid_neighbors) * 1.0

        for obs in obstacles:
            self.velocity += self.avoid_obstacle(obs) * 2.5

        margin = 50
        if self.pos[0] < margin: self.velocity[0] += 0.6
        elif self.pos[0] > self.model.width - margin: self.velocity[0] -= 0.6
        if self.pos[1] < margin: self.velocity[1] += 0.6
        elif self.pos[1] > self.model.height - margin: self.velocity[1] -= 0.6

        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        new_pos = self.pos + self.velocity
        eps = 0.01
        new_pos[0] = np.clip(new_pos[0], eps, self.model.width - eps)
        new_pos[1] = np.clip(new_pos[1], eps, self.model.height - eps)

        for obs in self.model.obstacles:
            if obs.get_rect().collidepoint(new_pos):
                self.velocity *= -0.5
                new_pos = self.pos
                break

        self.model.space.move_agent(self, new_pos)
        self.pos = new_pos

    def avoid_obstacle(self, obs):
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
    
import pygame
from mesa import Agent


class Obstacle(Agent):
    def __init__(self, model, pos, width, height):
        super().__init__(model)
        self.pos = pos
        self.width = width
        self.height = height

    def get_rect(self):
        return pygame.Rect(self.pos[0], self.pos[1], self.width, self.height)

class MazeGenerator:
    def __init__(self, width, height, min_room_size=200):
        self.width = width
        self.height = height
        self.min_room_size = min_room_size
        self.obstacles = []
        self.gap_size = 100 

    def generate(self, model):
        self.obstacles = []
        self._divide(0, 0, self.width, self.height, model)
        return self.obstacles

    def _divide(self, x, y, w, h, model):
        if w < self.min_room_size or h < self.min_room_size:
            return

        horizontal = h > w 
        if horizontal:
            split_y = model.random.randint(y + 50, y + h - 50)
            gap_x = model.random.randint(x, x + w - self.gap_size)
            
            left_wall = Obstacle(model, (x, split_y), gap_x - x, 15)
            right_wall = Obstacle(model, (gap_x + self.gap_size, split_y), w - (gap_x - x + self.gap_size), 15)
            
            self.obstacles.extend([left_wall, right_wall])
            
            self._divide(x, y, w, split_y - y, model)
            self._divide(x, split_y + 15, w, h - (split_y - y + 15), model)
        else:
            split_x = model.random.randint(x + 50, x + w - 50)
            gap_y = model.random.randint(y, y + h - self.gap_size)
            
            top_wall = Obstacle(model, (split_x, y), 15, gap_y - y)
            bottom_wall = Obstacle(model, (split_x, gap_y + self.gap_size), 15, h - (gap_y - y + self.gap_size))
            
            self.obstacles.extend([top_wall, bottom_wall])
            
            self._divide(x, y, split_x - x, h, model)
            self._divide(split_x + 15, y, w - (split_x - x + 15), h, model)

import heapq


def get_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def astar(grid, start, goal, limit=10000): 
    rows = len(grid)
    cols = len(grid[0])
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if current in closed_set:
            continue
        closed_set.add(current)
        
        if len(closed_set) > limit: break

        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                
                tentative_g = g_score[current] + get_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + get_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return None

import numpy as np
import pygame
from boids_logic import Boid, BoidModel, Migrator
from environment import Obstacle

SCREEN_W, SCREEN_H = 1600, 900
WORLD_W, WORLD_H = SCREEN_W * 3, SCREEN_H * 3

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    model = BoidModel(240, WORLD_W, WORLD_H, num_obstacles=35)

    cam_x, cam_y = 0, 0
    zoom = 1.0
    dragging = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            elif event.type == pygame.MOUSEWHEEL:
                zoom = np.clip(zoom + event.y * 0.1, 0.4, 2.5)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                cam_x = np.clip(cam_x - dx/zoom, 0, WORLD_W - SCREEN_W/zoom)
                cam_y = np.clip(cam_y - dy/zoom, 0, WORLD_H - SCREEN_H/zoom)

        model.step()
        screen.fill((20, 20, 25))

        for obs in model.obstacles:
            r = obs.get_rect()
            draw_rect = pygame.Rect(
                (r.x - cam_x) * zoom, (r.y - cam_y) * zoom,
                r.width * zoom, r.height * zoom
            )
            pygame.draw.rect(screen, (60, 60, 70), draw_rect)
            pygame.draw.rect(screen, (100, 100, 110), draw_rect, 2)

        for agent in model.agents:
            if isinstance(agent, Migrator):
                if agent.pos is not None:
                    rx = (agent.pos[0] - cam_x) * zoom
                    ry = (agent.pos[1] - cam_y) * zoom
                    
                    if 0 <= rx <= SCREEN_W and 0 <= ry <= SCREEN_H:
                        size = 6 * zoom
                        pygame.draw.circle(screen, (0, 255, 200), (int(rx), int(ry)), int(size))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
