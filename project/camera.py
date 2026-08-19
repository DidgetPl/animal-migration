import numpy as np
import pygame


class Camera:
    def __init__(self, screen_w: int, screen_h: int, world_w: int, world_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.world_w = world_w
        self.world_h = world_h
        
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL if False else event.type == 1027:
            self.zoom = float(np.clip(self.zoom + event.y * 0.05, 0.3, 2.5))
            self.clamp()

        elif event.type == 1025:
            if event.button == 1:
                self.dragging = True

        elif event.type == 1026:
            if event.button == 1:
                self.dragging = False

        elif event.type == 1024 and self.dragging:
            dx, dy = event.rel
            self.x -= dx / self.zoom
            self.y -= dy / self.zoom
            self.clamp()

    def clamp(self):
        view_w = self.screen_w / self.zoom
        view_h = self.screen_h / self.zoom

        if view_w >= self.world_w:
            self.x = (self.world_w - view_w) / 2
        else:
            self.x = float(np.clip(self.x, 0, self.world_w - view_w))

        if view_h >= self.world_h:
            self.y = (self.world_h - view_h) / 2
        else:
            self.y = float(np.clip(self.y, 0, self.world_h - view_h))