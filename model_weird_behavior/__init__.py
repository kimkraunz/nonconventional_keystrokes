import sys
import os

sys.path.append("/Users/kimkraunz/DataspellProjects/nonconventional_keystrokes")

from model_weird_behavior.calculate_features import KeystrokeFeatureExtractor

# Example usage
file_path = "/Users/kimkraunz/Library/CloudStorage/GoogleDrive-kim@mimoto.ai/Shared drives/ML/data/public_data/keystrokes/UB_keystroke_dataset/s0/baseline"
file_name = "001001.txt"
extractor = KeystrokeFeatureExtractor(os.path.join(file_path, file_name))
features = extractor.get_features()
print(features)

file_name = "002001.txt"
extractor = KeystrokeFeatureExtractor(os.path.join(file_path, file_name))
features = extractor.get_features()
print(features)
