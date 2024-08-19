import pandas as pd
from datetime import datetime
import os
import glob

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


def read_baseline_files_to_dataframes(root_path):
    baseline_dataframes = {}
    for session in ["s0", "s1", "s2"]:
        session_path = os.path.join(root_path, session, "baseline")
        if os.path.exists(session_path):
            files = glob.glob(os.path.join(session_path, "*1.txt"))
            for file in files:
                df = parse_input_data(file)
                df["session"] = session
                file_name = os.path.basename(file)
                df["user_id"] = int(file_name[:3])
                df["keyboard"] = file_name[4]
                file_name = os.path.basename(file)
                baseline_dataframes[f"{session}_{file_name}"] = df
    return baseline_dataframes
