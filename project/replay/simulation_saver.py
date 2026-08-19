import gzip
import os
import pickle
import struct
import time

import numpy as np

REPLAYS_DIR = "project\\replay\\replays"

def ensure_replays_dir():
    if not os.path.exists(REPLAYS_DIR):
        os.makedirs(REPLAYS_DIR)

class SimulationRecorder:
    def __init__(self, output_filename: str = None):
        ensure_replays_dir()
        
        if output_filename is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"replay_{timestamp}.bin.gz"
        
        if not output_filename.startswith(REPLAYS_DIR):
            self.output_filename = os.path.join(REPLAYS_DIR, output_filename)
        else:
            self.output_filename = output_filename
            
        self.frames = []
        self.metadata = {}

    def set_metadata(self, config: dict, model):
        self.metadata = {
            "width": model.space.width,
            "height": model.space.height,
            "cols": model.cols,
            "rows": model.rows,
            "mountain_threshold": model.mountain_threshold,
            "forest_threshold": model.forest_threshold,
            "terrain_height": model.terrain_height,
            "river_map": getattr(model, 'river_map', None),
            "config": config
        }

    def capture_frame(self, frame_index: int, agents: list):
        agent_data = []
        for agent in agents:
            if getattr(agent, 'pos', None) is None:
                continue

            pos_x, pos_y = float(agent.pos[0]), float(agent.pos[1])
            vel = getattr(agent, 'velocity', np.array([0.0, 0.0]))
            speed = np.linalg.norm(vel)
            angle = float(np.arctan2(vel[1], vel[0])) if speed > 0 else 0.0

            agent_type = 0
            flags = 0

            if agent.__class__.__name__ == 'Obstacle' or hasattr(agent, 'obstacle_type'):
                obs_type = getattr(agent, 'obstacle_type', 'rock')
                agent_type = 2 if obs_type == 'rock' else 3
                radius = getattr(agent, 'radius', 8.0)
                flags = int(np.clip(radius * 10, 0, 255))

            elif agent.__class__.__name__ == 'Predator' or getattr(agent, 'is_predator', False):
                agent_type = 1

            else:
                agent_type = 0
                if getattr(agent, 'scared', False):
                    flags |= 1
                if getattr(agent, 'is_feeding', False):
                    flags |= 2

            agent_data.append((agent.unique_id, pos_x, pos_y, angle, agent_type, flags))
            
        self.frames.append((frame_index, agent_data))

    def save_to_file(self):
        print(f"Zapisywanie {len(self.frames)} klatek symulacji do {self.output_filename}...")
        start_time = time.time()
        agent_struct = struct.Struct("<HfffBB")

        with gzip.open(self.output_filename, "wb") as f:
            meta_bytes = pickle.dumps(self.metadata)
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)

            f.write(struct.pack("<I", len(self.frames)))

            for frame_index, agent_list in self.frames:
                f.write(struct.pack("<IH", frame_index, len(agent_list)))
                for a_id, x, y, angle, a_type, flags in agent_list:
                    f.write(agent_struct.pack(a_id, x, y, angle, a_type, flags))

        elapsed = time.time() - start_time
        print(f"Zapis ukończony w {elapsed:.2f}s.")


class SimulationLoader:
    def __init__(self, input_filename: str):
        if not os.path.exists(input_filename) and not input_filename.startswith(REPLAYS_DIR):
            input_filename = os.path.join(REPLAYS_DIR, input_filename)
        self.input_filename = input_filename

    def load_all(self):
        agent_struct = struct.Struct("<HfffBB")
        agent_size = agent_struct.size

        with gzip.open(self.input_filename, "rb") as f:
            meta_len = struct.unpack("<I", f.read(4))[0]
            metadata = pickle.loads(f.read(meta_len))

            total_frames = struct.unpack("<I", f.read(4))[0]
            frames_data = []

            for _ in range(total_frames):
                frame_idx, num_agents = struct.unpack("<IH", f.read(6))
                agents = []
                
                for _ in range(num_agents):
                    a_bytes = f.read(agent_size)
                    a_id, x, y, angle, a_type, flags = agent_struct.unpack(a_bytes)
                    agents.append({
                        "id": a_id, "x": x, "y": y, 
                        "angle": angle, "type": a_type, "flags": flags
                    })
                
                frames_data.append({"frame": frame_idx, "agents": agents})

        return metadata, frames_data