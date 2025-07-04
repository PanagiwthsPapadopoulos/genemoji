import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.env_checker import check_env  
# from gymnasium.wrappers import EnvCompatibility
import imageio
import os
import torch
from typing import Union
import cv2
from collections import Counter
import time



class EmojiCompositionEnv(gym.Env):
    def __init__(self, emoji_data, target_vector, emoji_to_path, role_map, spatial_map, background, render_mode=None, canvas_size=(256, 256), max_steps=10):
        super().__init__()
        self.emoji_data = emoji_data
        self.target_vector = target_vector
        self.role_map = role_map
        self.spatial_map = spatial_map
        self.background = background
        self.emoji_to_path = emoji_to_path
        self.canvas_size = canvas_size
        self.max_steps = max_steps

        self.emoji_list = [e["emoji"] for e in emoji_data]

        # Accessory placement dictionary
        accessory_placement_hints = {
            # Headwear
            "hat": "top_head",
            "cap": "top_head",
            "crown": "top_head",
            "helmet": "top_head",
            "turban": "top_head",
            "headband": "top_head",
            "scarf": "neck",

            # Face Accessories
            "glasses": "eyes",
            "sunglasses": "eyes",
            "mask": "mouth",
            "mustache": "mouth",
            "beard": "mouth",

            # Hand Accessories
            "gloves": "hands",
            "watch": "hands",
            "bracelet": "hands",
            "ring": "hands",

            # Footwear
            "shoes": "feet",
            "boots": "feet",
            "sneakers": "feet",
            "socks": "feet",

            # Clothing
            "coat": "body",
            "shirt": "body",
            "jacket": "body",
            "dress": "body",
            "pants": "legs",
            "shorts": "legs",
            "skirt": "legs",
        }

        # Spatial verbs rules dictionary
        spatial_verb_rules = {
            "wear": "use_accessory_hints",
            "hold": "place_near_hands",
            "carry": "place_near_hands",
            "ride": "subject_on_object",
            "sit": "subject_on_object",
            "stand": "subject_on_object",
        }



        # Define action space: emoji index, x, y, scale, color_shift, crop_type, layer
        self.action_space = spaces.Box(
            low=np.array([
                0.0,                      # emoji index
                0.0, 0.0,                 # x, y positions normalized [0,1]
                0.0,                      # scale index
                0.0,                      # layer
                0.0,                      # color emoji index
                0.0,                      # crop type
                0.0                       # stop signal
            ], dtype=np.float32),
            high=np.array([
                len(self.emoji_list) - 1,  # emoji index
                0.8, 0.8,                  # x, y still [0,1]
                2.0,                       # scale (0 small, 1 medium, 2 large)
                3.0,                       # layer
                len(self.emoji_list),      # color reference (+1 = "no coloring")
                2.0,                       # crop (0 none, 1 accessory, 2 face)
                1.0                        # stop signal (1 = stop)
            ], dtype=np.float32)
        )

        # Observation: RGB image of canvas
        self.observation_space = spaces.Box(low=0, high=255, shape=(canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

        self.reset()

    def reset(self, seed=None, options=None):
        # print("Reset called!")
        super().reset(seed=seed)
        self.frames = []
        self.canvas = Image.new("RGBA", self.canvas_size, (255, 255, 255, 0))
        self.placed_emojis = []
        self.step_count = 0
        self.colored_emojis = set()
        return np.array(self.canvas.convert("RGB")), {}

    def render(self):
        if self.render_mode == "rgb_array":
            return np.array(self.canvas.convert("RGB"))
        else:
            self.canvas.show()

    def close(self):
        pass

    def _render_canvas_from_placed_emojis(self):
        # Reset the canvas (white transparent background)
        self.canvas = Image.new("RGBA", self.canvas_size, (255, 255, 255, 0))

        # Sort emojis by layer (z-index)
        self.placed_emojis.sort(key=lambda x: x[4])
        # print(f"🎨 Rendering canvas with {len(self.placed_emojis)} emojis...")

        for (emoji_symbol, x, y, size, layer) in self.placed_emojis:
            emoji_img = self._get_emoji_image(emoji_symbol)
            emoji_img = self._resize_image(emoji_img, size, size)
            if emoji_img.mode != "RGBA":
                emoji_img = emoji_img.convert("RGBA")

            visible_mask = emoji_img.split()[-1].point(lambda a: 255 if a > 0 else 0)
            self.canvas.paste(emoji_img, (x, y), visible_mask)

            # print(f"🔹 Emoji: {emoji_img} | Pos: ({x},{y}) | Size: {size} | Layer: {layer}")
            # print(f"📁 Path: {self.emoji_to_path.get(emoji_symbol, '❌ not found')}")

    def _get_emoji_image(self, emoji, extract_features=False):
        path = self.emoji_to_path.get(emoji)
        if not path or not os.path.exists(path):
            return Image.new("RGBA")

        img = Image.open(path).convert("RGBA")

        if extract_features:
            img = self._extract_facial_features(img, emoji)

        return img

    def _resize_image(self, image, height, width):
        return image.resize((height, width))

    def _color_image(self, image, R, G, B, a):
        tint_color=(R, G, B)
        img = image.convert("RGBA")
        r, g, b = tint_color

        # Split original image channels
        orig_r, orig_g, orig_b, alpha = img.split()

        # Blend each channel towards the tint
        blended_r = Image.blend(orig_r, Image.new("L", img.size, r), alpha=a)
        blended_g = Image.blend(orig_g, Image.new("L", img.size, g), alpha=a)
        blended_b = Image.blend(orig_b, Image.new("L", img.size, b), alpha=a)

        # Merge back with original alpha
        tinted = Image.merge("RGBA", (blended_r, blended_g, blended_b, alpha))
        return tinted

    def _apply_color_from_reference(self, emoji_img, reference_emoji):
        """
        Given a base emoji image and a reference emoji symbol,
        extract the dominant color from the reference and apply it to the base image.
        """
        ref_path = self.emoji_to_path.get(reference_emoji)
        if not ref_path or not os.path.exists(ref_path):
            print(f"⚠️ Reference emoji image for {reference_emoji} not found.")
            return emoji_img  # Return unchanged

        # Load reference emoji
        ref_img = Image.open(ref_path).convert("RGBA")
        np_ref = np.array(ref_img)
        alpha = np_ref[:, :, 3]

        # Only use non-transparent pixels
        mask = alpha > 0
        if not np.any(mask):
            print(f"⚠️ No visible pixels found in {reference_emoji}.")
            return emoji_img

        # Calculate average color
        avg_color = np.mean(np_ref[:, :, :3][mask], axis=0)
        r, g, b = [int(c) for c in avg_color]
        strength = 0.5  # You can later make this adjustable

        # Apply tint
        return self._color_image(emoji_img, r, g, b, strength)

    def _place_image(self, canvas, image, x, y):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        canvas.paste(image, (x, y), image)

    def _place_centered(self, canvas, image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        canvas_width, canvas_height = canvas.size
        image_width, image_height = image.size

        x = (canvas_width - image_width) // 2
        y = (canvas_height - image_height) // 2

        canvas.paste(image, (x, y), image)

    def _set_background(self, canvas, image, scale_factor=0.8):
        new_size = (int(canvas.size[0] * scale_factor), int(canvas.size[1] * scale_factor))
        bg = image.resize(new_size).convert("RGBA")
        self._place_centered(canvas, bg)

    def _extract_facial_features(self, image, emoji):
        # Get emoji entry
        entry = next((e for e in self.emoji_data if e["emoji"] == emoji), None)
        if not entry or "face" not in entry.get("keywords", []):
            return image
        
        # Colors to remove
        colors_to_remove = [
            (255, 204, 77),     # face yellow
            (254, 196, 87),
            (255, 203, 76),     # face yellow
            (80, 165, 230),     # face blue cold
            (73, 155, 221),     # face blue cold
            (73, 156, 221),     # face blue cold
            (218, 47, 71),      # face red angry
            (226, 64, 91),      # face red hot
        ]
        
        img = image.convert("RGBA")
        pixels = img.load()
        width, height = img.size

        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                if (r, g, b) in colors_to_remove:
                    pixels[x, y] = (0, 0, 0, 0)

        return img

    def _estimate_face_region_from_eyes(self, image, line_length=10):
        """
        Detects eye-like regions (both black and white) in an emoji.
        Draws a cross on detected eye centers for debugging.
        Shows grayscale and inverted grayscale debug views.
        Returns the modified image with crosses drawn.
        """
        img_rgba = image.convert("RGBA")
        np_img = np.array(img_rgba)
        alpha = np_img[:, :, 3]

        # Mask out transparent areas
        visible_mask = alpha > 0
        rgb = np_img[:, :, :3]
        rgb[~visible_mask] = [255, 255, 255]  # fill transparent with white

        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        inverted = cv2.bitwise_not(gray)

        # Show grayscale and inverted images for debugging
        Image.fromarray(gray).show(title="Grayscale")
        Image.fromarray(inverted).show(title="Inverted Grayscale")

        # Helper: detect circles
        def detect_from_grayscale(g):
            circles = cv2.HoughCircles(
                g,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=8,
                param1=50,
                param2=10,
                minRadius=2,
                maxRadius=10
            )
            return [(int(x), int(y)) for x, y, r in circles[0]] if circles is not None else []

        dark_eye_centers = detect_from_grayscale(gray)
        light_eye_centers = detect_from_grayscale(inverted)

        all_eyes = dark_eye_centers + light_eye_centers

        # Draw detection
        draw = ImageDraw.Draw(img_rgba)
        for x, y in all_eyes:
            draw.line([(x - line_length, y), (x + line_length, y)], fill="blue", width=1)
            draw.line([(x, y - line_length), (x, y + line_length)], fill="blue", width=1)

        if not all_eyes:
            print("⚠️ No eyes detected.")

        return img_rgba

    def _detect_eyes_with_binary_conversion(self, pil_image, line_length=10, threshold=58):
        """
        Detects eye-like circles by converting the emoji image to grayscale and then binary black-and-white.
        Shows both intermediate images for debugging, and returns the image with eye markers.
        """
        # Ensure image is in RGBA
        pil_image = pil_image.convert("RGBA")
        np_img = np.array(pil_image)
        alpha = np_img[:, :, 3]

        # Mask out transparent background
        visible_mask = alpha > 0
        rgb = np_img[:, :, :3]
        rgb[~visible_mask] = [255, 255, 255]  # fill transparent with white

        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Convert to pure black and white using threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Show grayscale and binary images
        # Image.fromarray(gray).show(title="Grayscale")
        # Image.fromarray(binary).show(title="Binary")

        # Detect circles using Hough transform on binary image
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=5.1,
            minRadius=1,
            maxRadius=3
        )

        # draw = ImageDraw.Draw(pil_image)

        # if circles is not None:
        #     circles = np.uint16(np.around(circles[0]))
        #     for x, y, r in circles:
        #         draw.line([(x - line_length, y), (x + line_length, y)], fill="blue", width=1)
        #         draw.line([(x, y - line_length), (x, y + line_length)], fill="blue", width=1)
        # else:
        #     print("⚠️ No circles detected.")

        if circles is None:
            print("⚠️ No eyes detected.")
            return []

        # Extract and round coordinates
        circles = np.round(circles[0]).astype(int)
        eye_coords = [(x, y) for x, y, r in circles]
        # If more than 2, select the closest pair
        if len(eye_coords) > 2:
            min_dist = float('inf')
            best_pair = (eye_coords[0], eye_coords[1])

            if(eye_coords[0] == eye_coords[1]):
                return [eye_coords[0]]

            for i in range(len(eye_coords)):
                for j in range(i+1, len(eye_coords)):
                    x1, y1 = eye_coords[i]
                    x2, y2 = eye_coords[j]
                    dist = np.hypot(x2 - x1, y2 - y1)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (eye_coords[i], eye_coords[j])
            return list(best_pair)
        return eye_coords

    def _remove_eyes(self, base_img, eye_coords, radius=5, sampling_radius=10):
        """
        Removes eyes dynamically by filling with the average surrounding color.
        
        - base_img: PIL image (RGBA)
        - eye_coords: List of (x, y) centers
        - radius: Radius of eye region to overwrite
        - sampling_radius: How large area around the eye to sample average color
        """
        base_img = base_img.copy()
        draw = ImageDraw.Draw(base_img)
        np_img = np.array(base_img)

        for x, y in eye_coords:
            # Define the sampling box
            x0 = max(0, x - sampling_radius)
            y0 = max(0, y - sampling_radius)
            x1 = min(np_img.shape[1], x + sampling_radius)
            y1 = min(np_img.shape[0], y + sampling_radius)

            # Crop the area and compute mean color
            sample_area = np_img[y0:y1, x0:x1, :3]  # (ignore alpha channel)
            avg_color = tuple(np.mean(sample_area.reshape(-1, 3), axis=0).astype(int)) + (255,)

            # Draw a filled circle over the eye using the average color
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=avg_color)

        return base_img

    def _overlay_facial_features(self, base_img, feature_img, eye_coords):
        """
        Overlays a facial feature image (like angry eyes or mask) onto the base emoji image
        using the given eye coordinates.
        
        - If one eye is detected: place the right half of the feature image centered on the eye.
        - If two eyes are detected: resize the feature image to fit between the eyes and center it.
        """
        base_img = self._remove_eyes(base_img, eye_coords)

        base_img = base_img.copy()
        
        if not eye_coords:
            print("❌ No eye coordinates provided.")
            return base_img

        feature_img = feature_img.convert("RGBA")

        

        if len(eye_coords) == 1:
            eye_x, eye_y = eye_coords[0]
            w, h = feature_img.size
            feature_half = feature_img.crop((w // 2, 0, w, h))  # right half
            new_w = 12  # default size
            new_h = int(h * (new_w / (w // 2)))
            resized_feature = feature_half.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x = eye_x - new_w // 2 +2
            y = eye_y - new_h // 2 +2
            base_img.paste(resized_feature, (x, y), resized_feature)

        elif len(eye_coords) >= 2:
            (x1, y1), (x2, y2) = eye_coords[:2]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            distance = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5)
            aspect_ratio = feature_img.height / feature_img.width
            new_w = distance + 40
            new_h = int(new_w * aspect_ratio) 
            resized_feature = feature_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x = cx - new_w // 2
            y = cy - new_h // 2
            base_img.paste(resized_feature, (x, y), resized_feature)

        return base_img

    def _decide_position(self, base_size, relation_type):
        """
        Given the base image size and the type of spatial relationship (wear, hold, ride),
        returns (x, y) coordinates where the object should be placed.
        """
        base_w, base_h = base_size

        if relation_type == "wear":
            # Top center (like hats, glasses, etc.)
            x = base_w // 2 - 16  # assuming object resized to ~32x32
            y = int(base_h * 0.2)  # slightly below top edge
        elif relation_type == "hold" or relation_type == "carry":
            # Lower side, simulating a hand holding
            x = base_w // 2 - 16
            y = int(base_h * 0.7)
        elif relation_type == "ride" or relation_type == "sit" or relation_type == "stand":
            # Bottom center, riding something
            x = base_w // 2 - 32  # make it a bit bigger maybe
            y = base_h - 48  # near bottom
        else:
            # Default center
            x = base_w // 2 - 16
            y = base_h // 2 - 16

        return x, y
    
    def _apply_spatial_relationship(self, base_img, object_img, relation_type):
        base_w, base_h = base_img.size

        x, y = decide_position((base_w, base_h), relation_type)
        
        object_resized = object_img.resize((32, 32))  # or smarter resizing later

        base_img.paste(object_resized, (x, y), object_resized)

        return base_img

    def decide_placement_order(spatial_map):
        """
        Decides which object (subject or object) should be placed first based on the action (verb).
        Returns a list of (first_to_place, second_to_place, action).
        """
        placement_instructions = []

        base_first_verbs = {"ride", "sit", "stand"}
        
        for subject, verb, obj in spatial_map:
            verb_lower = verb.lower()
            if verb_lower in base_first_verbs:
                # First place the object (e.g., horse), then the subject (e.g., penguin)
                placement_instructions.append((obj, subject, verb_lower))
            else:
                # Default: first place the subject, then the object (e.g., penguin holding ice)
                placement_instructions.append((subject, obj, verb_lower))

        return placement_instructions

    def decode_action(self, action):
        """
        Decodes the agent's continuous action array into structured placement commands.
        Action layout:
        [emoji_idx, x_norm, y_norm, scale_idx, layer, color_idx, crop_type]
        """

        # 0. Stop signal (new)
        stop_signal = float(action[7]) if len(action) > 7 else 0.0  # fallback for old models

        scaled = np.clip(action[0], 0.0, 1.0)
        emoji_idx = int(scaled * len(self.emoji_list))
        emoji_idx = min(emoji_idx, len(self.emoji_list) - 1)

        scale_idx = int(np.clip(round(action[3]), 0, 2))

        # 1. Normalize and interpret the action values
        scaled = np.clip(action[0], 0.0, 1.0)
        emoji_idx = int(scaled * len(self.emoji_list))
        emoji_idx = min(emoji_idx, len(self.emoji_list) - 1)

        # if emoji_idx >= len(self.emoji_list):
        #     print(f"⚠️ Agent selected invalid emoji index: {emoji_idx}")
        # else:
        #     print(f"✅ Agent selected emoji: {self.emoji_list[emoji_idx]}")

        x_norm = np.clip(action[1], 0.0, 1.0)
        y_norm = np.clip(action[2], 0.0, 1.0)
        scale_idx = int(np.clip(round(action[3]), 0, 2))
        layer = int(np.clip(round(action[4]), 0, 3))
        color_idx = int(np.clip(round(action[5]), 0, len(self.emoji_list)))
        crop_type = int(np.clip(round(action[6]), 0, 2))
        stop_signal = float(action[7]) if len(action) > 7 else 0.0

        # 2. Get size from scale index
        size_options = [32, 48, 256]
        size = size_options[scale_idx]

        # 3. Calculate max allowed positions to stay in canvas
        max_x = self.canvas_size[0] - size
        max_y = self.canvas_size[1] - size

        # 4. Clamp position
        x = int(np.clip(x_norm * self.canvas_size[0], 0, max_x))
        y = int(np.clip(y_norm * self.canvas_size[1], 0, max_y))


        

        # 4. Decode color action
        if color_idx < len(self.emoji_list):
            color_reference = self.emoji_list[color_idx]
        else:
            color_reference = None  # No coloring

        # 5. Decode crop type
        crop_actions = {
            0: None,        # No crop
            1: "accessory", # Crop accessory
            2: "face"       # Crop face
        }
        crop = crop_actions.get(crop_type, None)

        emoji = self.emoji_list[emoji_idx]


        # 6. Package everything nicely
        decoded = {
            "emoji": emoji,
            "x": x,
            "y": y,
            "size": size,
            "color_reference": color_reference,
            "crop": crop,
            "layer": layer,
            "stop": stop_signal > 0.5  # interpret as boolean
        }

        return decoded

    def step(self, action):
        """
        The agent takes an action.
        We decode it, apply it (place emoji, color it, crop it), and update the canvas.

        Action layout:
        [emoji_idx, x, y, scale_idx, color_reference, crop_type, layer]
        """
        # print(f"🚨 REAL STEP CALLED")
        decoded = self.decode_action(action)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        # ✅ 0. Early stop check
        if decoded["stop"]:
            done = True
            reward = self._calculate_reward(done=done)
            self.frames.append(np.array(self.canvas.convert("RGB")))
            # print(f"Step {self.step_count}: 🛑 Agent chose to stop.")
            return np.array(self.canvas.convert("RGB")), reward, True, False, {}
        
        # 1. Get the emoji image
        emoji_img = self._get_emoji_image(decoded["emoji"])

        # 2. Resize
        emoji_img = self._resize_image(emoji_img, decoded["size"], decoded["size"])

        # 3. Crop if necessary
        if decoded["crop"] == "face":
            emoji_img = self._extract_facial_features(emoji_img, decoded["emoji"])
        # elif decoded["crop"] == "accessory":
        #     emoji_img = self._extract_accessory(emoji_img)  # (You'll implement this soon!)

       # 4. Apply color if needed
        if decoded["color_reference"] is not None:
            # Only allow color on core emoji
            entry = next((e for e in self.emoji_data if e["emoji"] == decoded["emoji"]), None)
            if entry and any(k in self.role_map.get("core", []) for k in entry["fuzzy_scores"].keys()):
                emoji_img = self._apply_color_from_reference(emoji_img, decoded["color_reference"])
                self.colored_emojis.add(decoded["emoji"])

        # 5. Store Images for later rendering based on layering
        self.placed_emojis.append((
            decoded["emoji"],
            decoded["x"],
            decoded["y"],
            decoded["size"],
            decoded["layer"]
        ))

        # 6. Update environment
        

        # You can call your reward function here (very simple for now)
        reward = self._calculate_reward()

        self.frames.append(np.array(self.canvas.convert("RGB")))
        # print(f"Step {self.step_count}: placed emoji {decoded['emoji']} at ({decoded['x']}, {decoded['y']}) with size {decoded['size']} at layer {decoded['layer']}")

        return np.array(self.canvas.convert("RGB")), reward, done, False, {}

    def _render_all(self):
        self.canvas = Image.new("RGBA", self.canvas_size, (255, 255, 255, 0))  # Clear canvas

        # Sort emojis based on layer
        sorted_emojis = sorted(self.placed_emojis, key=lambda x: x[4])  # x[4] = layer

        for emoji, x, y, size, layer in sorted_emojis:
            emoji_img = self._get_emoji_image(emoji)
            emoji_img = self._resize_image(emoji_img, size, size)
            self._place_image(self.canvas, emoji_img, x, y)

        return np.array(self.canvas.convert("RGB"))

    def _composition_is_clean(self):
        """
        Returns True if the current emoji composition:
        - Has a core emoji
        - Has no duplicates
        - Has no color references
        """
        emoji_data_by_symbol = {e["emoji"]: e for e in self.emoji_data}
        placed_emoji_symbols = [e[0] for e in self.placed_emojis]
        core_emojis = set(self.role_map.get("core", []))
        color_emojis = set(self.role_map.get("color_reference", []))

        # Check for duplicates
        if len(set(placed_emoji_symbols)) < len(placed_emoji_symbols):
            return False

        core_found = False

        for emoji in placed_emoji_symbols:
            if emoji in color_emojis:
                return False
            entry = emoji_data_by_symbol.get(emoji)
            if not entry:
                continue
            if any(k in core_emojis for k in entry["fuzzy_scores"].keys()):
                core_found = True

        return core_found



    def decoded_has_color(self, emoji, placements):
        """
        Returns True if the given emoji has an associated color_reference in any placement.
        """
        return emoji in self.colored_emojis



  

    def _calculate_reward(self, done=False):
        rewards = 0.0
        penalties = 0.0

        placed = self.placed_emojis
        symbols = [e[0] for e in placed]
        counts = Counter(symbols)

        core_set = set(self.role_map.get("core", []))
        acc_set = set(self.role_map.get("accessory", []))
        color_set = set(self.role_map.get("color_reference", []))
        emoji_data_by_symbol = {e["emoji"]: e for e in self.emoji_data}

        canvas_w, canvas_h = self.canvas_size
        center_x, center_y = canvas_w // 2, canvas_h // 2

        core_found = False
        core_pos = None
        core_colored = False
        background_found = False

        has_color_requirement = any(
            k in self.role_map.get("color_reference", []) for k in self.target_vector.keys()
        )

        # 🚫 2. Light penalty if core is missing
       
        for emoji, x, y, size, layer in placed:
            entry = emoji_data_by_symbol.get(emoji)
            if not entry:
                continue

            keywords = entry["fuzzy_scores"].keys()

            # 🚫 1. Light penalty for placing a color emoji
            if emoji in color_set:
                penalties += 1.0

            # 🎯 CORE EMOJI
            if any(k in core_set for k in keywords):
                core_found = True
                core_pos = (x, y)
                core_colored = self.decoded_has_color(emoji, self.placed_emojis)

                rewards += 2.0  # ✅ base reward for placing a core emoji

                dist = np.hypot(x - center_x, y - center_y)
                if dist < 40:
                    rewards += 1.5  # ✅ near center

                if size == 48:
                    rewards += 1.0  # ✅ ideal size
                elif size == 256:
                    penalties += 2.0

                if layer == 1:
                    rewards += 1.0  # ✅ correct layer

                if has_color_requirement and core_colored:
                    rewards += 1.5  # ✅ correctly colored

            # 🎯 ACCESSORY EMOJI
            if any(k in acc_set for k in keywords):
                rewards += 1.0  # ✅ base reward for placing an accessory

                if size == 32:
                    rewards += 0.5  # ✅ small size
                elif size == 256:
                    penalties += 10.0  # ❌ too large for accessory
                    
                if layer == 2:
                    rewards += 0.5  # ✅ correct layer

                if core_pos:
                    dist = np.hypot(x - core_pos[0], y - core_pos[1])
                    if dist < 40:
                        rewards += 1  # ✅ near core

            # 🎯 BACKGROUND EMOJI
            if self.background and emoji == self.background:
                rewards += 4.0  # ✅ base reward for placing the background

                if size >= 200:
                    rewards += 4.0  # ✅ large size

                if layer == 0:
                    rewards += 1.5  # ✅ correct layer

                emoji_center_x = x + size // 2
                emoji_center_y = y + size // 2
                dist = np.hypot(emoji_center_x - center_x, emoji_center_y - center_y)
                if dist < 40:
                    rewards += 1.0  # ✅ near canvas center

            

        # Count each emoji appearance and punish duplicates
        counts = Counter([e[0] for e in self.placed_emojis])
        for emoji, count in counts.items():
            if count > 1:
                entry = emoji_data_by_symbol.get(emoji)
                if not entry:
                    continue
                
                keywords = entry["fuzzy_scores"].keys()

                if any(k in core_set for k in keywords):
                    penalties += (count - 1) * 6.0  # ❌ Strong penalty for duplicating core
                elif any(k in acc_set for k in keywords):
                    penalties += (count - 1) * 3.0  # ⚠️ Mild for accessory
                elif emoji in color_set:
                    penalties += (count - 1) * 1.0  # ⚠️ Rare, but low penalty
                else:
                    penalties += (count - 1) * 1.0  # General penalty for other emojis

        # Apply bonus for correct early stopping, else do nothing
        if done and self.step_count < self.max_steps:
            accessory_found = any(
                any(k in acc_set for k in emoji_data_by_symbol.get(e[0], {}).get("fuzzy_scores", {}))
                for e in self.placed_emojis
            )
            unique_emojis = {e[0] for e in self.placed_emojis}
            background_present = (
                self.background is None or self.background in unique_emojis
            )

            if core_found and accessory_found and background_present and len(unique_emojis) == len(self.placed_emojis):
                rewards += 4.0  # 🎉 Bonus for clean early stop

        if done and not core_found:
            penalties += 3.0  # ❌ discouraged early stop with nothing

        final_reward = round(rewards - penalties, 3)
        return max(final_reward, -5.0)








    




def generate_gif_from_episode(env, model, filename="emoji_result.gif"):
    frames = []

    obs = env.reset()
    raw_env = env.envs[0]  # Unwrap to access your EmojiCompositionEnv

    for _ in range(raw_env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Capture canvas
        frame = raw_env.canvas.convert("RGB")
        frames.append(np.array(frame))

        if done:
            break

    # Save the collected frames as a GIF
    imageio.mimsave(filename, frames, fps=1)
    print(f"🎞️ Saved GIF: {filename}")

def save_gif(frames, filename="episode.gif", fps=2):
    imageio.mimsave(filename, frames, fps=fps)

architectures = {
    "mlp_32x2": ([32, 32], 8000),
    "mlp_64x2": ([64, 64], 10000),
    "mlp_128x1": ([128], 25000),
    "mlp_128x2": ([128, 128], 30000),
    "mlp_256x2": ([256, 256], 50000),
    "mlp_128x3": ([128, 128, 128], 60000),
}



# === MAIN SCRIPT ===
def generate_emoji_image(chromosome: list[str], role_map: dict[str, list[str]], spatial_map: list[tuple[str, str, str]], background: Union[str, None], target_vector: dict, emoji_data: list[dict]):
    
    def unicode_to_filename(unicode_str):
        return "-".join(code.replace("U+", "").lower() for code in unicode_str.strip().split())

    emoji_to_path = {
        entry["emoji"]: os.path.join("emoji_images", unicode_to_filename(entry["unicode"]) + ".png")
        for entry in emoji_data
    }

    # ======================== Training phase ========================
    

    def make_env():
            return EmojiCompositionEnv(
                emoji_data=emoji_data,
                target_vector=target_vector,
                emoji_to_path=emoji_to_path,
                render_mode="rgb_array",
                role_map=role_map,
                spatial_map=spatial_map,
                background=background,
            )
    
    for arch_name, (net_arch, steps) in architectures.items():
        start_time = time.time()
        print(f"\n🔧 Running architecture: {arch_name} for {steps} timesteps")

        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))
        
        env = DummyVecEnv([make_env])
        env = VecTransposeImage(env)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cpu"  # force CPU if needed

        model = PPO("MlpPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs, ent_coef=0.1)

        output_dir = f"models/{arch_name}"
        os.makedirs(output_dir, exist_ok=True)

        model.learn(total_timesteps=steps, progress_bar=True)
        model.save(os.path.join(output_dir, "emoji_agent_model"))

        end_time = time.time()
        print(f"✅ Model saved to {output_dir}/emoji_agent_model.zip with Training time: {end_time-start_time}")

        env.close()
    


    # ======================== Testing phase ===========================
    # Create a fresh env
    # test_env = DummyVecEnv([
    #     lambda: EmojiCompositionEnv(
    #         emoji_data=emoji_data,
    #         target_vector=target_vector,
    #         emoji_to_path=emoji_to_path,
    #         role_map=role_map,
    #         spatial_map=spatial_map,
    #         background=background,
    #         render_mode="rgb_array"
    #     )
    # ])


    # Load model
    # model = PPO.load(MODEL_NAME, env=test_env, device=device)
    # print("✅ Model loaded for testing.")

    # Run test episodes
    # for episode_idx in range(20):
    #     obs = test_env.reset()
    #     raw_env = test_env.envs[0]
    #     total_reward = 0.0
    #     final_valid_image = None

    #     for step_idx in range(raw_env.max_steps):
    #         action, _states = model.predict(obs, deterministic=False)
    #         obs, reward, done, info = test_env.step(action)

    #         reward_value = float(reward[0])
    #         total_reward += reward_value

    #         # Only update final image if the episode is not ending now
    #         if not done:
    #             raw_env._render_canvas_from_placed_emojis()
    #             final_valid_image = raw_env.canvas.convert("RGB")

    #         if done:
    #             break

    #     # If final_valid_image was never set (episode stopped immediately), force render once
    #     if final_valid_image is None:
    #         raw_env._render_canvas_from_placed_emojis()
    #         final_valid_image = raw_env.canvas.convert("RGB")

    #     # Annotate reward
    #     draw = ImageDraw.Draw(final_valid_image)
    #     try:
    #         font = ImageFont.truetype("arial.ttf", 16)
    #     except:
    #         font = ImageFont.load_default()

    #     text = f"Reward: {round(total_reward, 2)}"
    #     bbox = draw.textbbox((0, 0), text, font=font)
    #     text_w = bbox[2] - bbox[0]
    #     text_h = bbox[3] - bbox[1]
    #     x = final_valid_image.width - text_w - 10
    #     y = final_valid_image.height - text_h - 10
    #     draw.rectangle([x - 4, y - 2, x + text_w + 4, y + text_h + 2], fill=(0, 0, 0, 180))
    #     draw.text((x, y), text, fill="white", font=font)

    #     final_valid_image.save(f"episodes/episode_{episode_idx:04d}.png")
    #     print(f"✅ Saved episodes/episode_{episode_idx:04d}.png")

    # test_env.close()







    return
