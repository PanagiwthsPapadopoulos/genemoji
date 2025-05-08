import json

from tokenization import get_target_vector
from emoji_generation import run_genetic_algorithm
from image_generation import generate_emoji_image

with open("emoji_fuzzy_scores.json", "r", encoding="utf-8") as f:
    emoji_data = json.load(f)

all_keywords = set()
for emoji in emoji_data:
    all_keywords.update(emoji["fuzzy_scores"].keys())
all_keywords = sorted(list(all_keywords))

vocab = set()
for emoji in emoji_data:
    vocab.update(emoji["fuzzy_scores"].keys())

# Step 1: Tokenize the input prompt

prompt = "black penguin in the mountains"
# target_vector, role_map, spatial_map, background = get_target_vector(prompt, vocab)

# print("\n\n")
# print("🎯 Fuzzy Feature Vector:")
# print(target_vector)
# print("🔍 Role mapping:")
# print(role_map)
# print("🪐 Spatial mapping:")
# print(spatial_map)
# print("🌄 Background:")
# print(background)


# Step 2: Run the genetic algorithm to select best emojis
# chromosome = run_genetic_algorithm(target_vector, emoji_data, vocab)

# print("\n\n🏁 Best emoji combination:", chromosome)
# print("\n\n")

chromosome = ['🐧', '🧊', '⛰', '⚫️']
background = "mountain"
spatial_map = [('penguin', 'hold', 'ice')]
role_map = {'core': ['penguin'], 'modifier': ['black'], 'accessory': ['ice']}
target_vector = {'penguin': 1.0, 'black': 0.9, 'ice': 0.4, 'mountain': 0.5}

emoji_data_filtered = [
    entry for entry in emoji_data if entry["emoji"] in chromosome
]

# Step 3: Generate the image using DRL
generate_emoji_image(chromosome, role_map, spatial_map, background, target_vector, emoji_data_filtered)

