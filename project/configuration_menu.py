import os
import re
import sys
from datetime import datetime

import six
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)
from replay.replay_utils import get_available_replays

REPLAYS_DIR = "replays"

DEFAULTS = {
    "world_w_mult": 2.0,
    "world_h_mult": 4.0,
    "mountain_threshold": 0.56,
    "forest_threshold": 0.28,
    "enable_river": True,
    "num_obstacles": 120,
    "num_migrators": 80,
    "num_predators": 3,
    "max_speed": 3.0,
    "river_speed_mod": 0.3,
    "river_current": 0.02,
    "hunger_rate": 0.035,
    "river_cost": 3.0,
    "forest_cost": 2.0,
    "grass_regrowth": 0.001,
    "record_simulation": False
}

class ConfigurationMenu(QDialog):
    def __init__(self):
        super().__init__()
        self.config = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Symulator Migracji - Menu & Konfiguracja")
        self.setMinimumSize(880, 680)

        root_layout = QVBoxLayout(self)

        title = QLabel("Ustawienia Symulacji i Odtwarzacz Powtórek")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-top: 5px; margin-bottom: 5px;")
        root_layout.addWidget(title)

        group_replay = QGroupBox("Odtwarzanie Nagranych Powtórek")
        layout_replay = QHBoxLayout()

        self.combo_replays = QComboBox()
        self.btn_refresh_replays = QPushButton("🔄 Odśwież")
        self.btn_refresh_replays.clicked.connect(self.refresh_replays)

        self.btn_start_replay = QPushButton("▶ Odtwórz Wybraną Powtórkę")
        self.btn_start_replay.setStyleSheet("font-weight: bold; padding: 6px 15px; background-color: #1976d2; color: white;")
        self.btn_start_replay.clicked.connect(self.start_replay)

        layout_replay.addWidget(QLabel("Wybierz powtórkę:"))
        layout_replay.addWidget(self.combo_replays, stretch=1)
        layout_replay.addWidget(self.btn_refresh_replays)
        layout_replay.addWidget(self.btn_start_replay)
        group_replay.setLayout(layout_replay)

        root_layout.addWidget(group_replay)

        self.refresh_replays()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()

        columns_layout = QHBoxLayout(scroll_content)
        left_column = QVBoxLayout()

        group_world = QGroupBox("Wymiary Świata (Wielokrotności Ekranu)")
        layout_world = QFormLayout()

        self.spin_world_w = QDoubleSpinBox()
        self.spin_world_w.setRange(1.0, 2.0)
        self.spin_world_w.setSingleStep(0.1)

        self.spin_world_h = QDoubleSpinBox()
        self.spin_world_h.setRange(1.0, 6.0)
        self.spin_world_h.setSingleStep(0.1)

        layout_world.addRow("Mnożnik szerokości (World W):", self.spin_world_w)
        layout_world.addRow("Mnożnik wysokości (World H):", self.spin_world_h)
        group_world.setLayout(layout_world)
        left_column.addWidget(group_world)

        group_terrain = QGroupBox("Ukształtowanie Terenu i Przeszkody")
        layout_terrain = QFormLayout()

        self.spin_mountain_thresh = QDoubleSpinBox()
        self.spin_mountain_thresh.setRange(0.0, 1.0)
        self.spin_mountain_thresh.setSingleStep(0.01)

        self.spin_forest_thresh = QDoubleSpinBox()
        self.spin_forest_thresh.setRange(0.0, 1.0)
        self.spin_forest_thresh.setSingleStep(0.01)

        self.chk_river = QCheckBox("Generuj rzekę w świecie")

        self.spin_obstacles = QSpinBox()
        self.spin_obstacles.setRange(0, 1000)

        layout_terrain.addRow("Próg dla gór:", self.spin_mountain_thresh)
        layout_terrain.addRow("Próg dla lasu:", self.spin_forest_thresh)
        layout_terrain.addRow("Rzeka w świecie:", self.chk_river)
        layout_terrain.addRow("Liczba przeszkód:", self.spin_obstacles)
        group_terrain.setLayout(layout_terrain)
        left_column.addWidget(group_terrain)

        group_costs = QGroupBox("Opory Terenu i Regeneracja")
        layout_costs = QFormLayout()

        self.spin_river_cost = QDoubleSpinBox()
        self.spin_river_cost.setRange(1.0, 10.0)
        self.spin_river_cost.setSingleStep(0.5)

        self.spin_forest_cost = QDoubleSpinBox()
        self.spin_forest_cost.setRange(1.0, 10.0)
        self.spin_forest_cost.setSingleStep(0.5)

        self.spin_grass_regrowth = QDoubleSpinBox()
        self.spin_grass_regrowth.setRange(0.0001, 0.01)
        self.spin_grass_regrowth.setDecimals(4)

        layout_costs.addRow("Opór rzeki:", self.spin_river_cost)
        layout_costs.addRow("Opór lasu:", self.spin_forest_cost)
        layout_costs.addRow("Szybkość odrastania trawy:", self.spin_grass_regrowth)
        group_costs.setLayout(layout_costs)
        left_column.addWidget(group_costs)

        columns_layout.addLayout(left_column)

        right_column = QVBoxLayout()

        group_pop = QGroupBox("Populacja Agentów")
        layout_pop = QFormLayout()

        self.spin_migrators = QSpinBox()
        self.spin_migrators.setRange(1, 500)

        self.spin_predators = QSpinBox()
        self.spin_predators.setRange(0, 50)

        layout_pop.addRow("Liczba Migratorów:", self.spin_migrators)
        layout_pop.addRow("Liczba Drapieżników:", self.spin_predators)
        group_pop.setLayout(layout_pop)
        right_column.addWidget(group_pop)

        group_movement = QGroupBox("Parametry Ruchu i Wody")
        layout_movement = QFormLayout()

        self.spin_max_speed = QDoubleSpinBox()
        self.spin_max_speed.setRange(0.5, 10.0)
        self.spin_max_speed.setSingleStep(0.5)

        self.spin_river_speed_mod = QDoubleSpinBox()
        self.spin_river_speed_mod.setRange(0.05, 1.0)
        self.spin_river_speed_mod.setSingleStep(0.05)

        self.spin_river_current = QDoubleSpinBox()
        self.spin_river_current.setRange(0.0, 0.5)
        self.spin_river_current.setDecimals(3)
        self.spin_river_current.setSingleStep(0.005)

        layout_movement.addRow("Max prędkość migratora:", self.spin_max_speed)
        layout_movement.addRow("Modyfikator prędkości w rzece:", self.spin_river_speed_mod)
        layout_movement.addRow("Siła prądu rzeki:", self.spin_river_current)
        group_movement.setLayout(layout_movement)
        right_column.addWidget(group_movement)

        group_metabolism = QGroupBox("Metabolizm")
        layout_metabolism = QFormLayout()

        self.spin_hunger_rate = QDoubleSpinBox()
        self.spin_hunger_rate.setRange(0.0, 0.15)
        self.spin_hunger_rate.setDecimals(3)
        self.spin_hunger_rate.setSingleStep(0.005)

        layout_metabolism.addRow("Współczynnik głodu (Hunger Rate):", self.spin_hunger_rate)
        group_metabolism.setLayout(layout_metabolism)
        right_column.addWidget(group_metabolism)

        group_record = QGroupBox("Rejestracja i Zapis")
        layout_record = QFormLayout()

        self.chk_record_sim = QCheckBox("Włącz rejestrację symulacji do pliku")
        layout_record.addRow("Zapis pozycji agentów:", self.chk_record_sim)
        group_record.setLayout(layout_record)
        right_column.addWidget(group_record)

        right_column.addStretch()

        columns_layout.addLayout(right_column)

        scroll_area.setWidget(scroll_content)
        root_layout.addWidget(scroll_area)

        btn_layout = QHBoxLayout()

        self.btn_reset = QPushButton("Przywróć domyślne")
        self.btn_reset.setStyleSheet("padding: 8px 15px; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_to_defaults)

        self.btn_cancel = QPushButton("Wyjście")
        self.btn_cancel.setStyleSheet("padding: 8px 15px;")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_start = QPushButton("Uruchom Nową Symulację")
        self.btn_start.setStyleSheet("font-weight: bold; padding: 8px 20px; background-color: #2e7d32; color: white;")
        self.btn_start.clicked.connect(self.start_simulation)

        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_start)

        root_layout.addLayout(btn_layout)

        self.reset_to_defaults()

    def refresh_replays(self):
        self.combo_replays.clear()
        replays = get_available_replays()

        if replays:
            for item in replays:
                self.combo_replays.addItem(item["label"], userData=item["path"])
            self.combo_replays.setEnabled(True)
            self.btn_start_replay.setEnabled(True)
        else:
            self.combo_replays.addItem("Brak zapisanych powtórek w folderze 'replays'")
            self.combo_replays.setEnabled(False)
            self.btn_start_replay.setEnabled(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def start_replay(self):
        selected_file = self.combo_replays.currentData()
        if selected_file:
            self.config = {
                "mode": "replay",
                "replay_file": selected_file
            }
            self.accept()

    def reset_to_defaults(self):
        self.spin_world_w.setValue(DEFAULTS["world_w_mult"])
        self.spin_world_h.setValue(DEFAULTS["world_h_mult"])
        self.spin_mountain_thresh.setValue(DEFAULTS["mountain_threshold"])
        self.spin_forest_thresh.setValue(DEFAULTS["forest_threshold"])
        self.chk_river.setChecked(DEFAULTS["enable_river"])
        self.spin_obstacles.setValue(DEFAULTS["num_obstacles"])

        self.spin_migrators.setValue(DEFAULTS["num_migrators"])
        self.spin_predators.setValue(DEFAULTS["num_predators"])

        self.spin_max_speed.setValue(DEFAULTS["max_speed"])
        self.spin_river_speed_mod.setValue(DEFAULTS["river_speed_mod"])
        self.spin_river_current.setValue(DEFAULTS["river_current"])
        self.spin_hunger_rate.setValue(DEFAULTS["hunger_rate"])

        self.spin_river_cost.setValue(DEFAULTS["river_cost"])
        self.spin_forest_cost.setValue(DEFAULTS["forest_cost"])
        self.spin_grass_regrowth.setValue(DEFAULTS["grass_regrowth"])

        self.chk_record_sim.setChecked(DEFAULTS["record_simulation"])

    def start_simulation(self):
        self.config = {
            "mode": "run",
            "world_w_mult": self.spin_world_w.value(),
            "world_h_mult": self.spin_world_h.value(),
            "mountain_threshold": self.spin_mountain_thresh.value(),
            "forest_threshold": self.spin_forest_thresh.value(),
            "enable_river": self.chk_river.isChecked(),
            "num_obstacles": self.spin_obstacles.value(),
            "num_migrators": self.spin_migrators.value(),
            "num_predators": self.spin_predators.value(),
            "max_speed": self.spin_max_speed.value(),
            "river_speed_mod": self.spin_river_speed_mod.value(),
            "river_current": self.spin_river_current.value(),
            "hunger_rate": self.spin_hunger_rate.value(),
            "river_cost": self.spin_river_cost.value(),
            "forest_cost": self.spin_forest_cost.value(),
            "grass_regrowth": self.spin_grass_regrowth.value(),
            "record_simulation": self.chk_record_sim.isChecked()
        }
        self.accept()


def get_simulation_config():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    menu = ConfigurationMenu()
    if menu.exec() == QDialog.Accepted:
        return menu.config
    return None


if __name__ == "__main__":
    cfg = get_simulation_config()
    print("Wygenerowany słownik konfiguracji:")
    print(cfg)