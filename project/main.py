import sys

import numpy as np
import pygame
from boid_model import BoidModel
from renderer import WorldRenderer
from simulation_menu import get_simulation_config

SCREEN_W, SCREEN_H = 1600, 900
WORLD_W, WORLD_H = SCREEN_W * 2, SCREEN_H * 4

def clamp_camera(cam_x, cam_y, zoom):
    view_w = SCREEN_W / zoom
    view_h = SCREEN_H / zoom

    if view_w >= WORLD_W:
        cam_x = (WORLD_W - view_w) / 2
    else:
        cam_x = np.clip(cam_x, 0, WORLD_W - view_w)

    if view_h >= WORLD_H:
        cam_y = (WORLD_H - view_h) / 2
    else:
        cam_y = np.clip(cam_y, 0, WORLD_H - view_h)

    return cam_x, cam_y

def main():
    config = get_simulation_config()

    if not config:
        print("Symulacja anulowana przez użytkownika.")
        sys.exit(0)

    print("Uruchamianie symulacji z parametrami:", config)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Symulacja Migracji Gnu")
    clock = pygame.time.Clock()
    
    model = BoidModel(
        num_migrators=config["num_migrators"],
        num_predators=config["num_predators"],
        river_cost=config["river_cost"],
        forest_cost=config["forest_cost"],
        grass_regrowth=config["grass_regrowth"],
        width=int(SCREEN_W * config["world_w_mult"]),
        height=int(SCREEN_H * config["world_h_mult"]),
        mountain_threshold=config["mountain_threshold"],
        forest_threshold=config["forest_threshold"],
        enable_river=config["enable_river"],
        num_obstacles=config["num_obstacles"]
    )
    renderer = WorldRenderer(SCREEN_W, SCREEN_H)

    cam_x, cam_y = 0.0, 0.0
    zoom = 1.0
    dragging = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            elif event.type == pygame.MOUSEWHEEL:
                zoom = float(np.clip(zoom + event.y * 0.05, 0.3, 2.5))
                cam_x, cam_y = clamp_camera(cam_x, cam_y, zoom)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: 
                    dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                cam_x -= dx / zoom
                cam_y -= dy / zoom
                cam_x, cam_y = clamp_camera(cam_x, cam_y, zoom)

        model.step()

        renderer.render(screen, model, cam_x, cam_y, zoom)
        
        clock.tick(60)

if __name__ == "__main__":
    main()