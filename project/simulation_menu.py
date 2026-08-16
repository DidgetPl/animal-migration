import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)


class SimulationMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.config = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Symulator Migracji - Konfiguracja")
        self.setMinimumWidth(400)

        main_layout = QVBoxLayout()

        title = QLabel("Ustawienia Symulacji Migracji")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        group_agents = QGroupBox("Populacja")
        layout_agents = QFormLayout()

        self.spin_migrators = QSpinBox()
        self.spin_migrators.setRange(10, 500)
        self.spin_migrators.setValue(80)
        layout_agents.addRow("Liczba Migratorów:", self.spin_migrators)

        self.spin_predators = QSpinBox()
        self.spin_predators.setRange(0, 20)
        self.spin_predators.setValue(3)
        layout_agents.addRow("Liczba Drapieżników:", self.spin_predators)

        group_agents.setLayout(layout_agents)
        main_layout.addWidget(group_agents)

        group_env = QGroupBox("Środowisko i Fizyka")
        layout_env = QFormLayout()

        self.spin_river_cost = QDoubleSpinBox()
        self.spin_river_cost.setRange(1.0, 10.0)
        self.spin_river_cost.setValue(3.0)
        self.spin_river_cost.setSingleStep(0.5)
        layout_env.addRow("Opór rzeki:", self.spin_river_cost)

        self.spin_forest_cost = QDoubleSpinBox()
        self.spin_forest_cost.setRange(1.0, 10.0)
        self.spin_forest_cost.setValue(2.0)
        self.spin_forest_cost.setSingleStep(0.5)
        layout_env.addRow("Opór lasu:", self.spin_forest_cost)

        self.spin_grass_regrowth = QDoubleSpinBox()
        self.spin_grass_regrowth.setRange(0.0001, 0.01)
        self.spin_grass_regrowth.setValue(0.001)
        self.spin_grass_regrowth.setDecimals(4)
        layout_env.addRow("Szybkość odrastania trawy:", self.spin_grass_regrowth)

        group_env.setLayout(layout_env)
        main_layout.addWidget(group_env)

        btn_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Uruchom Symulację")
        self.btn_start.setStyleSheet("font-weight: bold; padding: 8px; background-color: #2e7d32; color: white;")
        self.btn_start.clicked.connect(self.start_simulation)

        self.btn_cancel = QPushButton("Wyjście")
        self.btn_cancel.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_start)
        
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def start_simulation(self):
        self.config = {
            "num_migrators": self.spin_migrators.value(),
            "num_predators": self.spin_predators.value(),
            "river_cost": self.spin_river_cost.value(),
            "forest_cost": self.spin_forest_cost.value(),
            "grass_regrowth": self.spin_grass_regrowth.value(),
        }
        self.close()


def get_simulation_config():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    menu = SimulationMenu()
    menu.show()
    app.exec()

    return menu.config