import gzip
import struct
import time


class SimulationRecorder:
    def __init__(self, output_filename: str = None):
        if output_filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"simulation_record_{timestamp}.bin.gz"
        
        self.output_filename = output_filename
        self.frames = []
        self.metadata = {}

    def set_metadata(self, config: dict, map_width: int, map_height: int):
        self.metadata = {
            "width": map_width,
            "height": map_height,
            "config": config
        }

    def capture_frame(self, frame_index: int, agents: list):
        agent_data = []
        for agent in agents:
            agent_type = 0
            if hasattr(agent, 'is_predator') and agent.is_predator:
                agent_type = 1
            elif agent.__class__.__name__ == 'Obstacle':
                agent_type = 2

            pos_x, pos_y = float(agent.pos[0]), float(agent.pos[1])
            agent_data.append((agent.unique_id, pos_x, pos_y, agent_type))
            
        self.frames.append((frame_index, agent_data))

    def save_to_file(self):
        print(f"Zapisywanie {len(self.frames)} klatek symulacji do {self.output_filename}...")
        start_time = time.time()

        agent_struct = struct.Struct("<HffB")

        with gzip.open(self.output_filename, "wb") as f:
            f.write(struct.pack("<I", len(self.frames)))

            for frame_index, agent_list in self.frames:
                f.write(struct.pack("<IH", frame_index, len(agent_list)))
                
                for a_id, x, y, a_type in agent_list:
                    f.write(agent_struct.pack(a_id, x, y, a_type))

        elapsed = time.time() - start_time
        print(f"Zapis zakończony w {elapsed:.2f}s. Plik: {self.output_filename}")


class SimulationLoader:
    def __init__(self, input_filename: str):
        self.input_filename = input_filename

    def read_frames(self):
        agent_struct = struct.Struct("<HffB")
        agent_size = agent_struct.size

        frames_data = []

        with gzip.open(self.input_filename, "rb") as f:
            total_frames = struct.unpack("<I", f.read(4))[0]

            for _ in range(total_frames):
                frame_idx, num_agents = struct.unpack("<IH", f.read(6))
                agents = []
                
                for _ in range(num_agents):
                    agent_bytes = f.read(agent_size)
                    a_id, x, y, a_type = agent_struct.unpack(agent_bytes)
                    agents.append({"id": a_id, "x": x, "y": y, "type": a_type})
                
                frames_data.append({"frame": frame_idx, "agents": agents})

        return frames_data