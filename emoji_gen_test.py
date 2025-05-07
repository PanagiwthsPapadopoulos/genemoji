from image_generation import EmojiCompositionEnv  # Adjust this to your actual file name
from PIL import Image
import os
import json

# Mock data
with open("emoji_fuzzy_scores.json", "r", encoding="utf-8") as f:
    emoji_data = json.load(f)

chromosome = ['🐧', '☄', '⛰', '🧊']
background = "mountain"
spatial_map = [('penguin', 'hold', 'ice')]
role_map = {'core': ['penguin'], 'modifier': ['black'], 'accessory': ['ice']}
target_vector = {'penguin': 1.0, 'black': 0.9, 'ice': 0.4, 'mountain': 0.5}

def unicode_to_filename(unicode_str):
    return "-".join(code.replace("U+", "").lower() for code in unicode_str.strip().split())


emoji_to_path = {
        entry["emoji"]: os.path.join("emoji_images", unicode_to_filename(entry["unicode"]) + ".png")
        for entry in emoji_data
    }

def unicode_to_filename(unicode_str):
    return "-".join(code.replace("U+", "").lower() for code in unicode_str.strip().split())

# Initialize environment
env = EmojiCompositionEnv(
            emoji_data=emoji_data,
            target_vector=target_vector,
            emoji_to_path=emoji_to_path,
            render_mode="rgb_array", 
            role_map = role_map,
            spatial_map = spatial_map,
            background=background,
        )

# Test loading and cropping
emoji = "🦒"
size = 64
x, y = 50, 50

animal = env._get_emoji_image(emoji)





eye_coords = env._detect_eyes_with_binary_conversion(animal)

# animal = env._remove_eyes(animal, eye_coords)

emoji = "🙄"

face = env._get_emoji_image(emoji)

fatial_features = env._extract_facial_features(face, emoji)

img = env._overlay_facial_features(animal, fatial_features, eye_coords)

env._set_background(env.canvas, img)
# img = env._color_image(img, 0, 255, 0, 0.6)
# img = env._resize_image(img, 30, 30)
# env._place_image(env.canvas, img, 0, 0)


env.canvas.show()
