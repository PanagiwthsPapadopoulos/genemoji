import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from stable_baselines3 import PPO
from image_generation import EmojiCompositionEnv
import json
import numpy as np

# === Prompt setup ===
prompt = "an angry penguin holds ice and wears a monocle"
chromosome = ['🐧', '🧊', '⛰']
background = "mountain"
spatial_map = [('penguin', 'hold', 'ice')]
role_map = {'core': ['penguin'], 'modifier': ['black'], 'accessory': ['ice']}
target_vector = {'penguin': 1.0, 'black': 0.9, 'ice': 0.4, 'mountain': 0.5}

# === Load emoji dataset ===
with open("emoji_fuzzy_scores.json", "r", encoding="utf-8") as f:
    emoji_data = json.load(f)

emoji_data_filtered = [e for e in emoji_data if e["emoji"] in chromosome]
print(f"✅ emoji_data_filtered = {[e['emoji'] for e in emoji_data_filtered]}")
assert len(emoji_data_filtered) == len(chromosome), "❌ Missing emoji data!"

def unicode_to_filename(unicode_str):
    return "-".join(code.replace("U+", "").lower() for code in unicode_str.strip().split())

emoji_to_path = {}
for entry in emoji_data_filtered:
    emoji = entry["emoji"]
    unicode_filename = unicode_to_filename(entry["unicode"])
    path = os.path.join("emoji_images", f"{unicode_filename}.png")
    if os.path.exists(path):
        emoji_to_path[emoji] = path
    else:
        print(f"⚠️ Missing image for {emoji} → {path}")

# === Agents to evaluate ===
agents = [
    "mlp_32x2",
    "mlp_64x2",
    "mlp_128x1",
    "mlp_128x2",
    "mlp_256x2",
    "mlp_128x3",
]

NUMBER_OF_EPISODES = 10

for agent_name in agents:
    print(f"\n🧪 Evaluating {agent_name}")
    model_path = f"models/{agent_name}/emoji_agent_model.zip"
    output_dir = f"models/{agent_name}/eval"
    os.makedirs(output_dir, exist_ok=True)

    # Define raw environment
    def make_env():
        return EmojiCompositionEnv(
            emoji_data=emoji_data_filtered,
            target_vector=target_vector,
            emoji_to_path=emoji_to_path,
            render_mode="rgb_array",
            role_map=role_map,
            spatial_map=spatial_map,
            background=background,
        )

    env = make_env()
    model = PPO.load(model_path, device="cpu")

    rewards = []

    for i in range(NUMBER_OF_EPISODES):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break

        # Render final image
        env._render_canvas_from_placed_emojis()
        img = env.canvas.convert("RGB")
        img_path = os.path.join(output_dir, f"episode_{i:02d}.png")
        img.save(img_path)
        # print(f"🖼 Saved: {img_path}")
        rewards.append(total_reward)

    # Save reward data
    df = pd.DataFrame({"episode": list(range(NUMBER_OF_EPISODES)), "reward": rewards})
    df.to_csv(os.path.join(output_dir, "rewards.csv"), index=False)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards, ddof=1)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    # Plot reward graph
    plt.figure()
    plt.plot(df["episode"], df["reward"], marker='o')
    plt.title(f"Reward per Episode – {agent_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_plot.png"))
    plt.close()

    print(f"✅ Completed {agent_name}")
    print(f"   ➤ Mean reward: {mean_reward:.2f}")
    print(f"   ➤ Std deviation: {std_reward:.2f}")
    print(f"   ➤ Min reward: {min_reward:.2f}")
    print(f"   ➤ Max reward: {max_reward:.2f}")