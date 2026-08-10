import numpy as np
from variables import GRID_SIZE


class MigrationFlowField:
    def __init__(self, model):
        self.model = model
        self.rows = model.rows
        self.cols = model.cols
        self.field = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        
        # Wagi sił
        self.w_north = 1.0       # Siła dążenia na Północ
        self.w_grass = 1.8       # Siła szukania pożywienia
        self.w_terrain = 2.5     # Unikanie trudnego terenu
        self.w_boundary = 3.5    # Siła odpychania od brzegów mapy

    def update_field(self):
        grass_map = getattr(self.model, 'grass_map', np.ones((self.rows, self.cols)))
        terrain_map = self.model.terrain_cost_map

        gy, gx = np.gradient(grass_map)
        ty, tx = np.gradient(terrain_map)

        # Szerokość stref buforowych (w kafelkach)
        margin_x = 4  # Dla krawędzi lewej / prawej
        margin_y = 5  # Dla dolnej krawędzi (nieco większy bufor, by mocno nadawać kierunek w górę)

        for r in range(self.rows):
            for c in range(self.cols):
                # 1. Dryf Północny
                f_north = np.array([0.0, -1.0])

                # 2. Wektor Trawy
                f_grass = np.array([gx[r, c], gy[r, c]])
                norm_g = np.linalg.norm(f_grass)
                if norm_g > 0:
                    f_grass /= norm_g

                # 3. Wektor Terenu
                f_terrain = np.array([-tx[r, c], -ty[r, c]])
                norm_t = np.linalg.norm(f_terrain)
                if norm_t > 0:
                    f_terrain /= norm_t

                if terrain_map[r, c] > 3.0:
                    f_terrain *= (terrain_map[r, c] / 2.0)

                # --- 4. ODPYCHANIE OD KRAWĘDZI (W/Z ORAZ POŁUDNIE) ---
                f_boundary = np.array([0.0, 0.0])

                # Lewa krawędź (Zachód) -> spychaj w prawo (+X)
                if c < margin_x:
                    dist_factor = (margin_x - c) / margin_x
                    f_boundary[0] += dist_factor ** 2

                # Prawa krawędź (Wschód) -> spychaj w lewo (-X)
                elif c >= self.cols - margin_x:
                    dist_factor = (c - (self.cols - margin_x - 1)) / margin_x
                    f_boundary[0] -= dist_factor ** 2

                # Dolna krawędź (Południe) -> spychaj w górę (-Y na planszy)
                if r >= self.rows - margin_y:
                    dist_factor = (r - (self.rows - margin_y - 1)) / margin_y
                    f_boundary[1] -= dist_factor ** 2  # Wypychamy ku górze (ujemne Y)

                # --- SUMOWANIE SIŁ ---
                combined = (
                    self.w_north * f_north +
                    self.w_grass * f_grass +
                    self.w_terrain * f_terrain +
                    self.w_boundary * f_boundary
                )

                norm_c = np.linalg.norm(combined)
                if norm_c > 0:
                    combined /= norm_c

                self.field[r, c] = combined

    def get_force_at(self, pos):
        grid_x = int(np.clip(pos[0] // GRID_SIZE, 0, self.cols - 1))
        grid_y = int(np.clip(pos[1] // GRID_SIZE, 0, self.rows - 1))
        return self.field[grid_y, grid_x]