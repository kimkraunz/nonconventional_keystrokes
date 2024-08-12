import pandas as pd
from datetime import datetime

__all__ = ["parse_input_data"]


def parse_input_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                key, action, timestamp = parts
                data.append({"key": key, "action": action, "timestamp": int(timestamp)})
    return pd.DataFrame(data)
