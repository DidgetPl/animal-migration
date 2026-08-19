import sys

import mesa
from configuration_menu import get_simulation_config
from replay.replay_player import run_replay
from simulation.simulation_player import run_live_simulation


def main():
    while True:
        config = get_simulation_config()

        if not config:
            sys.exit(0)

        if config.get("mode") == "replay":
            replay_file = config.get("replay_file")
            run_replay(replay_file)

        elif config.get("mode") == "run":
            run_live_simulation(config)


if __name__ == "__main__":
    main()