import sys

import numpy as np
import pygame
from camera import Camera
from renderer import WorldRenderer
from replay.simulation_saver import SimulationLoader
from simulation.boids.migrator import Migrator
from simulation.boids.obstacle import Obstacle
from simulation.boids.predator import Predator
from variables import SCREEN_H, SCREEN_W


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


def draw_hud(screen, font, frame_idx, total_frames, is_paused, speed):
    status_str = "PAUZA" if is_paused else ("WSTECZ" if speed < 0 else "ODTWARZANIE")
    speed_str = f"{abs(speed):.2f}x"
    
    hud_lines = [
        f"Stan: {status_str} | Prędkość: {speed_str}",
        f"Klatka: {frame_idx + 1} / {total_frames} ({(frame_idx / max(1, total_frames - 1)) * 100:.1f}%)",
        "-----------------------------------------",
        "[SPACJA] Pauza/Wznowienie | [R] Reset prędkości",
        "[GÓRA/DÓŁ] Zmiana prędkości (w tym odtwarzanie w tył)",
        "[LEWO/PRAWO] Krok w tył / Krok w przód",
        "[MYSZ] Przeciąganie i Zoom kamery",
        "[ESC] Wyjście z odtwarzacza"
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


def run_replay(filename):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Odtwarzacz Powtórek Symulacji - Kontrola Czasu")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14, bold=True)

    loader = SimulationLoader(filename)
    metadata, frames = loader.load_all()

    dummy_model = DummyModel(metadata)
    renderer = WorldRenderer(SCREEN_W, SCREEN_H, flip=False)
    camera = Camera(SCREEN_W, SCREEN_H, metadata["width"], metadata["height"])

    total_frames = len(frames)
    current_frame = 0.0
    is_paused = False
    playback_speed = 1.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused

                elif event.key == pygame.K_RIGHT:
                    current_frame = min(total_frames - 1, int(current_frame) + 1)
                elif event.key == pygame.K_LEFT:
                    current_frame = max(0, int(current_frame) - 1)

                elif event.key in (pygame.K_UP, pygame.K_PLUS, pygame.K_KP_PLUS):
                    if playback_speed < 0:
                        playback_speed /= 1.5
                        if abs(playback_speed) < 0.25:
                            playback_speed = 0.25
                    else:
                        playback_speed = min(16.0, playback_speed * 1.5)

                elif event.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                    if playback_speed > 0:
                        playback_speed /= 1.5
                        if playback_speed < 0.25:
                            playback_speed = -0.25
                    else:
                        playback_speed = max(-16.0, playback_speed * 1.5)

                elif event.key == pygame.K_r:
                    playback_speed = 1.0

            camera.handle_event(event)

        if not is_paused:
            current_frame += playback_speed
            
            if current_frame >= total_frames:
                current_frame = total_frames - 1
                is_paused = True
            elif current_frame < 0:
                current_frame = 0.0
                is_paused = True

        idx = int(np.clip(current_frame, 0, total_frames - 1))

        dummy_model.update_frame(frames[idx]["agents"])
        renderer.render(screen, dummy_model, camera.x, camera.y, camera.zoom)

        draw_hud(screen, font, idx, total_frames, is_paused, playback_speed)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()