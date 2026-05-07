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
        self.gap_size = 100  # Szerokość przejścia dla zwierząt

    def generate(self, model):
        self.obstacles = []
        self._divide(0, 0, self.width, self.height, model)
        return self.obstacles

    def _divide(self, x, y, w, h, model):
        # Warunek stopu - pomieszczenie jest za małe
        if w < self.min_room_size or h < self.min_room_size:
            return

        # Decyzja: dzielimy pionowo czy poziomo?
        horizontal = h > w # Dzielimy dłuższą krawędź

        if horizontal:
            # Losujemy linię podziału (y)
            split_y = model.random.randint(y + 50, y + h - 50)
            # Losujemy gdzie będzie dziura w ścianie
            gap_x = model.random.randint(x, x + w - self.gap_size)
            
            # Tworzymy dwie przeszkody (lewa część ściany i prawa część ściany)
            left_wall = Obstacle(model, (x, split_y), gap_x - x, 15)
            right_wall = Obstacle(model, (gap_x + self.gap_size, split_y), w - (gap_x - x + self.gap_size), 15)
            
            self.obstacles.extend([left_wall, right_wall])
            
            # Rekurencja dla góry i dołu
            self._divide(x, y, w, split_y - y, model)
            self._divide(x, split_y + 15, w, h - (split_y - y + 15), model)
        else:
            # Losujemy linię podziału (x)
            split_x = model.random.randint(x + 50, x + w - 50)
            # Losujemy dziurę
            gap_y = model.random.randint(y, y + h - self.gap_size)
            
            # Ściana górna i dolna
            top_wall = Obstacle(model, (split_x, y), 15, gap_y - y)
            bottom_wall = Obstacle(model, (split_x, gap_y + self.gap_size), 15, h - (gap_y - y + self.gap_size))
            
            self.obstacles.extend([top_wall, bottom_wall])
            
            # Rekurencja dla lewej i prawej strony
            self._divide(x, y, split_x - x, h, model)
            self._divide(split_x + 15, y, w - (split_x - x + 15), h, model)