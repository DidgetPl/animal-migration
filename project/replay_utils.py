import os
import re
from datetime import datetime

REPLAYS_DIR = "replays"

def format_replay_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", basename)
    
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M-%S")
            return dt.strftime("Nagranie z %d.%m.%Y r., godz. %H:%M:%S")
        except ValueError:
            pass
            
    return basename

def get_available_replays():
    if not os.path.exists(REPLAYS_DIR):
        os.makedirs(REPLAYS_DIR)
        
    replays = []
    for f in os.listdir(REPLAYS_DIR):
        if f.endswith(".bin.gz"):
            full_path = os.path.join(REPLAYS_DIR, f)
            label = format_replay_filename(f)
            replays.append({
                "path": full_path,
                "label": label,
                "mtime": os.path.getmtime(full_path)
            })
            
    replays.sort(key=lambda x: x["mtime"], reverse=True)
    return replays