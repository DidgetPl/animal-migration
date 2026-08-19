import sys

import pygame
from boid_model import BoidModel
from camera import Camera
from renderer import WorldRenderer
from simulation_menu import get_simulation_config
from simulation_saver import SimulationRecorder

SCREEN_W, SCREEN_H = 1600, 900

def main():
    config = get_simulation_config()

    if not config:
        print("Symulacja anulowana przez użytkownika.")
        sys.exit(0)

    print("Uruchamianie symulacji z parametrami:", config)

    world_w = int(SCREEN_W * config["world_w_mult"])
    world_h = int(SCREEN_H * config["world_h_mult"])

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Symulacja Migracji Gnu")
    clock = pygame.time.Clock()

    recorder = None
    if config.get("record_simulation", False):
        recorder = SimulationRecorder()
        recorder.set_metadata(config, world_w, world_h)

    model = BoidModel(
        num_migrators=config["num_migrators"],
        num_predators=config["num_predators"],
        river_cost=config["river_cost"],
        forest_cost=config["forest_cost"],
        grass_regrowth=config["grass_regrowth"],
        width=world_w,
        height=world_h,
        mountain_threshold=config["mountain_threshold"],
        forest_threshold=config["forest_threshold"],
        enable_river=config["enable_river"],
        num_obstacles=config["num_obstacles"]
    )
    
    renderer = WorldRenderer(SCREEN_W, SCREEN_H)
    camera = Camera(SCREEN_W, SCREEN_H, world_w, world_h)

    frame_count = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                camera.handle_event(event)

        model.step()
    
        if recorder:
            recorder.capture_frame(frame_count, model.agents)

        renderer.render(screen, model, camera.x, camera.y, camera.zoom)
        frame_count += 1

        clock.tick(60)

    if recorder:
        recorder.save_to_file()

if __name__ == "__main__":
    main()