import numpy as np
import pygame
from camera import Camera
from renderer import WorldRenderer
from simulation.boid_model import BoidModel
from simulation.simulation_recorder import SimulationRecorder
from variables import SCREEN_H, SCREEN_W


def draw_hud(screen, font, model, current_frame, is_paused):
    status_str = "PAUZA" if is_paused else "SYMULACJA"
    
    hud_lines = [
        f"Stan: {status_str}",
        f"Klatka: {current_frame:.0f}",
        f"Liczba migratorów: {len(model.get_migrators())}",
        f"Liczba drapieżników: {len(model.get_predators())}",
        "-----------------------------------------",
        "[F] Pokaż pole przepływu",
        "[SPACJA] Pauza / Wznowienie",
        "[MYSZ] Przeciąganie i Zoom kamery",
        "[ESC] Wyjście do menu"
    ]

    padding = 10
    line_height = 22
    box_w = 460
    box_h = len(hud_lines) * line_height + padding * 2

    hud_surface = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 180))
    screen.blit(hud_surface, (10, 10))

    for i, line in enumerate(hud_lines):
        color = (255, 215, 0) if i == 0 else (220, 220, 220)

        txt = font.render(line, True, color)
        screen.blit(txt, (10 + padding, 10 + padding + i * line_height))


def run_live_simulation(config):
    world_w = int(SCREEN_W * config["world_w_mult"])
    world_h = int(SCREEN_H * config["world_h_mult"])

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Symulacja Migracji Gnu")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14, bold=True)

    recorder = None
    if config.get("record_simulation", False):
        recorder = SimulationRecorder()

    model = BoidModel(
        num_migrators=config["num_migrators"],
        num_predators=config["num_predators"],
        river_cost=config["river_cost"],
        forest_cost=config["forest_cost"],
        mountain_cost=config["mountain_cost"],
        grass_regrowth=config["grass_regrowth"],
        width=world_w,
        height=world_h,
        mountain_threshold=config["mountain_threshold"],
        forest_threshold=config["forest_threshold"],
        enable_river=config["enable_river"],
        num_obstacles=config["num_obstacles"],
        migrator_speed=config["migrator_speed"],
        river_speed_mod=config["river_speed_mod"],
        river_stream=config["river_current"],
        hunger_rate=config["hunger_rate"]
    )

    if recorder:
        recorder.set_metadata(config, model)

    renderer = WorldRenderer(SCREEN_W, SCREEN_H, flip=False)
    camera = Camera(SCREEN_W, SCREEN_H, world_w, world_h)

    frame_count = 0
    running = True
    is_paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
            else:
                camera.handle_event(event)

        if not is_paused:
            model.step()

            if recorder:
                recorder.capture_frame(frame_count, model.agents)

            frame_count += 1

        show_flow_field = pygame.key.get_pressed()[pygame.K_f]

        renderer.render(screen, model, camera.x, camera.y, camera.zoom, show_flow_field=show_flow_field)
        draw_hud(screen, font, model, frame_count, is_paused)

        pygame.display.flip()
        clock.tick(60)

    if recorder:
        recorder.save_to_file()

    pygame.quit()