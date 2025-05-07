import json

# --- Configuration ---
input_file = "/Users/panagiwths/Desktop/emoji_fuzzy_scores_exact.json"
output_file = "/Users/panagiwths/Desktop/emoji_fuzzy_scores.json"
feature_to_remove = "country"  # Change this to the feature you want to remove

# --- Load the JSON data ---
with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Count total entries ---
entry_count = len(data)
print(f"Total entries in the file: {entry_count}")

import torch

print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built()) 