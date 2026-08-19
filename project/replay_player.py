import sys

import pygame
from simulation_saver import SimulationLoader


class Camera:
    def __init__(self, screen_width, screen_height, world_width, world_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_width = world_width
        self.world_height = world_height
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.speed = 15

    def handle_input(self, keys):
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.offset_x -= self.speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.offset_x += self.speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.offset_y -= self.speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.offset_y += self.speed

        self.offset_x = max(0, min(self.offset_x, self.world_width - self.screen_width))
        self.offset_y = max(0, min(self.offset_y, self.world_height - self.screen_height))

    def apply(self, x, y):
        """Przekształca pozycję ze świata na pozycję na ekranie."""
        return x - self.offset_x, y - self.offset_y


def run_replay(filename, screen_width=1600, screen_height=900):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Odtwarzacz Symulacji (Replay)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    print(f"Wczytywanie nagrania z {filename}...")
    loader = SimulationLoader(filename)
    frames = loader.read_frames()
    
    if not frames:
        print("Plik nie zawiera żadnych klatek!")
        return

    total_frames = len(frames)
    print(f"Wczytano {total_frames} klatek.")

    max_x = max(a["x"] for f in frames for a in f["agents"]) if frames else screen_width
    max_y = max(a["y"] for f in frames for a in f["agents"]) if frames else screen_height
    world_w = max(screen_width, int(max_x + 100))
    world_h = max(screen_height, int(max_y + 100))

    camera = Camera(screen_width, screen_height, world_w, world_h)

    current_frame_idx = 0
    is_paused = False
    playback_speed = 1.0
    base_fps = 60

    COLOR_MIGRATOR = (50, 150, 255)
    COLOR_PREDATOR = (220, 50, 50)
    COLOR_OBSTACLE = (100, 100, 100)
    COLOR_BG = (30, 30, 30)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_RIGHT and is_paused:
                    current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
                elif event.key == pygame.K_LEFT and is_paused:
                    current_frame_idx = max(0, current_frame_idx - 1)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    playback_speed = min(8.0, playback_speed * 1.5)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    playback_speed = max(0.25, playback_speed / 1.5)

        keys = pygame.key.get_pressed()
        camera.handle_input(keys)

        if not is_paused:
            current_frame_idx += 1
            if current_frame_idx >= total_frames:
                current_frame_idx = total_frames - 1
                is_paused = True

        screen.fill(COLOR_BG)

        frame_data = frames[current_frame_idx]

        for agent in frame_data["agents"]:
            screen_x, screen_y = camera.apply(agent["x"], agent["y"])

            if -20 <= screen_x <= screen_width + 20 and -20 <= screen_y <= screen_height + 20:
                a_type = agent["type"]
                
                if a_type == 0:
                    pygame.draw.circle(screen, COLOR_MIGRATOR, (int(screen_x), int(screen_y)), 6)
                elif a_type == 1:
                    pygame.draw.circle(screen, COLOR_PREDATOR, (int(screen_x), int(screen_y)), 8)
                elif a_type == 2:
                    pygame.draw.rect(screen, COLOR_OBSTACLE, (int(screen_x) - 8, int(screen_y) - 8, 16, 16))

        status_str = "PAUZA" if is_paused else "ODTWARZANIE"
        hud_text = f"Klatka: {current_frame_idx + 1}/{total_frames} | Status: {status_str} | Prędkość: {playback_speed:.2f}x"
        controls_text = "Sterowanie: WSAD/Strzałki - Kamera | Spacja - Pauza | +/- Prędkość | Left/Right (na pauzie) - Krok"

        txt_surface = font.render(hud_text, True, (255, 255, 255))
        txt_ctrl = font.render(controls_text, True, (180, 180, 180))

        pygame.draw.rect(screen, (0, 0, 0, 150), (10, 10, 650, 50))
        screen.blit(txt_surface, (15, 15))
        screen.blit(txt_ctrl, (15, 38))

        pygame.display.flip()

        clock.tick(int(base_fps * playback_speed))

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "simulation_record_20260819_200847.bin.gz"

    run_replay(target_file)