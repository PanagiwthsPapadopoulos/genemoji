# 🧠 GenEmoji – Natural Language to Emoji Image Generation

**GenEmoji** is an AI system that transforms natural language descriptions into coherent emoji-based images. It integrates **Fuzzy Logic**, **Genetic Algorithms**, and **Deep Reinforcement Learning (PPO)** to interpret user input, generate symbolic emoji combinations, and render them visually.

This project was developed for the course **Computational Intelligence – Deep Reinforcement Learning**.

---

## 📁 Project Structure
```
├── main.py # Launch point for training agents
├── evaluate_and_plot.py # Evaluates PPO models and plots rewards
├── tokenization.py # Prompt parsing and fuzzy vector generation
├── emoji_generation.py # Genetic algorithm for emoji selection
├── image_generation.py # DRL environment and PPO agent training
├── emoji_fuzzy_scores.json # Dataset: emoji fuzzy scores by keyword
├── emoji_images/ # Folder of emoji PNGs named by Unicode
├── models/ # Trained PPO agents by architecture
└── Papadopoulos_Panagiotis_10697_DRL.pdf # Final project report
```
---

## 🧠 Problem Overview

The user enters a natural language prompt such as:

> "an angry penguin holds ice and wears a monocle"

The system performs:

1. **Fuzzy Linguistic Parsing** – Converts the prompt into a fuzzy feature vector using SpaCy and a custom fuzzy modifier dictionary.
2. **Emoji Selection via Genetic Algorithm** – Evolves a chromosome (emoji list) that matches the fuzzy vector.
3. **Emoji Composition via PPO Agent** – Trains a DRL agent to place emojis on a canvas to reflect the semantics of the prompt.

---

## 📦 Dataset

The dataset is custom-built and includes:
- Emoji character
- Unicode codepoint
- Keyword list
- Fuzzy scores per keyword (range [0, 1])

Each emoji is represented as a PNG image. Fuzzy scores were curated manually and with the aid of language models.

---

## 🛠 Tools & Libraries

| Library            | Role                                      |
|--------------------|-------------------------------------------|
| `SpaCy`            | POS tagging, dependency parsing           |
| `Gymnasium`        | Custom DRL environment (`EmojiCompositionEnv`) |
| `stable-baselines3`| PPO training and policy modeling          |
| `PyTorch`          | Backend for PPO (via SB3)                 |
| `Pillow`           | Emoji image loading, composition, rendering |
| `OpenCV`           | Eye and facial feature detection (Hough transform) |
| `Matplotlib`       | Reward visualization                      |
| `NumPy`            | Array operations                          |
| `tqdm`             | Progress tracking                         |

---

## 🧬 Genetic Algorithm

Each **chromosome** is a list of 4 emojis. Fitness evaluation considers:

- Fuzzy similarity to target vector
- Keyword coverage
- Emoji diversity
- Strong matches (≥ 80% match)
- Efficiency (coverage/length)
- Penalties: redundancy and noise
- Relevance bonuses for keyword similarity

Mutation rate decays over generations. Tournament selection and elitism are used for evolution.

---

## 🤖 PPO Agent & DRL Environment

The PPO agent is trained to place emojis using an 8-dimensional continuous action space:

1. Emoji index  
2. (x, y) canvas position  
3. Scale (small, medium, large)  
4. Layer (background to foreground)  
5. Color reference emoji  
6. Crop type (face, accessory, none)  
7. Stop signal (optional)

The agent is rewarded for:
- Correct emoji layering and positioning
- Placing a core emoji near the center
- Keeping accessories near the core
- Avoiding duplicates or noise
- Early stopping with a clean composition

---

## 🧪 Evaluation

Evaluation is performed on 6 neural network architectures:

- `mlp_32x2`, `mlp_64x2`, `mlp_128x1`, `mlp_128x2`, `mlp_128x3`, `mlp_256x2`

Each is trained for a different number of timesteps and compared by mean reward, variance, and visual output.

Run evaluation with:

```bash
python evaluate_and_plot.py
```
Output includes:

- Episode .png images
- rewards.csv
- reward_plot.png

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train PPO agents

Edit the `prompt` inside `main.py`, then run:

```bash
python main.py
```
This will train PPO agents using six different MLP architectures and save the trained models under the models/ directory.

### 3. Evaluate trained agents

To evaluate the trained agents, run the following command in your terminal:
```
python evaluate_and_plot.py
```
Each trained agent is evaluated across 10 episodes. For every architecture, the following files are generated:

- .png images of emoji compositions  
- rewards.csv with episode reward scores  
- reward_plot.png showing reward progression

Results are saved in:
```
results/<architecture_name>/
```
---

## 📝 Example Prompt

an angry penguin holds ice and wears a monocle

Resulting fuzzy target vector:
```
{  
  "penguin": 1.0,  
  "angry": 0.9,  
  "ice": 0.4,  
  "monocle": 0.4  
}
```
---

## 📄 License

This project was developed as part of academic coursework.  
No license is currently attached.

---

















