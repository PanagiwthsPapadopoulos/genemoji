import json
import random
import numpy as np

# PARAMETERS
POP_SIZE = 450
CHROMOSOME_LENGTH = 4
GENERATIONS = 20
# MUTATION_RATE = 0.7

# STEP 1 – Load emoji keyword data
# with open("emoji_fuzzy_scores.json", "r", encoding="utf-8") as f:
#     emoji_data = json.load(f)

# STEP 2 – Load target fuzzy vector (from prompt)
# with open("fuzzy_prompt_vector.json", "r", encoding="utf-8") as f:
#     target_vector = json.load(f)

# STEP 3 – Build full keyword list
# all_keywords = set()
# for emoji in emoji_data:
#     all_keywords.update(emoji["fuzzy_scores"].keys())
# all_keywords = sorted(list(all_keywords))

# STEP 4 – Helper: get fuzzy vector for a chromosome
def get_combined_vector(chromosome, all_keywords, emoji_data):
    combined = {key: 0.0 for key in all_keywords}
    for emoji_symbol in chromosome:
        for entry in emoji_data:
            if entry["emoji"] == emoji_symbol:
                for k, v in entry["fuzzy_scores"].items():
                    combined[k] = max(combined[k], v)
    return combined

# STEP 5 – Fitness function 
def fitness(chromosome, all_keywords, emoji_data, target_vector):
    combined = get_combined_vector(chromosome, all_keywords, emoji_data)
    prompt_keywords = target_vector.keys()

    # 1. Fuzzy Dot Product Similarity (fix overshooting problem)
    weighted_match = sum(
        target_vector[k] * combined.get(k, 0.0)
        for k in prompt_keywords
    )
    fuzzy_similarity = weighted_match / sum(target_vector.values())

    # 2. Keyword Coverage
    coverage = sum(1 for k in prompt_keywords if combined.get(k, 0.0) > 0.0)
    coverage_ratio = coverage / len(prompt_keywords)

    # 3. Emoji Diversity
    diversity = len(set(chromosome)) / len(chromosome)

    # 4. Strong Match Bonus
    strong_matches = sum(
        1 for k in prompt_keywords if combined.get(k, 0.0) >= 0.8 * target_vector[k]
    )
    strong_bonus = strong_matches / len(prompt_keywords)

    # 5. Efficiency Bonus
    efficiency = coverage / len(chromosome) if chromosome else 0.0

    # 6. Redundancy Penalty
    from collections import defaultdict
    keyword_hits = defaultdict(int)
    for emoji in chromosome:
        entry = next(e for e in emoji_data if e["emoji"] == emoji)
        for k in entry["fuzzy_scores"].keys():
            if k in prompt_keywords:
                keyword_hits[k] += 1
    redundant_keywords = sum(1 for k in keyword_hits if keyword_hits[k] > 1)
    redundancy_penalty = redundant_keywords / len(prompt_keywords)

    # Keyword Relevance Bonus
    relevance_bonus = 0.0
    for k1 in prompt_keywords:
        for k2 in combined:
            if k1 != k2 and target_vector.get(k1, 0.0) > 0 and combined.get(k2, 0.0) > 0:
                if k1 in k2 or k2 in k1:  # simple substring relevance
                    relevance_bonus += 0.01
    relevance_bonus = min(relevance_bonus, 0.1)

    # 7. Noise Penalty
    noise_emojis = 0
    for emoji in chromosome:
        entry = next(e for e in emoji_data if e["emoji"] == emoji)
        if not any(k in prompt_keywords for k in entry["fuzzy_scores"].keys()):
            noise_emojis += 1
    noise_ratio = noise_emojis / len(chromosome)

    # Final fitness score
    score = (
        0.5 * fuzzy_similarity +
        0.15 * coverage_ratio +
        0.1 * diversity +
        0.15 * strong_bonus +
        0.1 * efficiency +
        relevance_bonus
    )
    score -= 0.1 * redundancy_penalty
    score -= 0.1 * noise_ratio
    return score

def run_genetic_algorithm(target_vector: dict, emoji_data: dict, all_keywords: set) -> tuple[list[str], list[dict]]:
    # STEP 6 – Prepare emoji pool
    prompt_keywords = set(target_vector.keys())
    relevant = [
        e["emoji"] for e in emoji_data
        if any(k in prompt_keywords for k in e["fuzzy_scores"].keys())
    ]
    random_others = random.sample(
        [e["emoji"] for e in emoji_data if e["emoji"] not in relevant],
        k=max(1, len(relevant) // 5)
    )
    emoji_pool = [e["emoji"] for e in emoji_data]

    # STEP 7 – Initialize population
    population = [
        [random.choice(emoji_pool) for _ in range(CHROMOSOME_LENGTH)]
        for _ in range(POP_SIZE)
    ]

    # STEP 8 – Evolution loop
    for generation in range(GENERATIONS):
        population = sorted(population, key=lambda chromo: fitness(chromo, all_keywords, emoji_data, target_vector), reverse=True)
        print(f"Gen {generation+1}: Best fitness = {fitness(population[0], all_keywords, emoji_data, target_vector):.4f}  Chromosome = {population[0]}")

        new_population = population[:2]  # Elitism: keep top 2

        # Decaying mutation rate
        MUTATION_RATE = max(0.3, 0.9 - generation * 0.05)

        while len(new_population) < POP_SIZE:
            parent1, parent2 = random.choices(population[:5], k=2)  # Tournament selection
            crossover_point = random.randint(1, CHROMOSOME_LENGTH - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            
            # Mutation
            if random.random() < MUTATION_RATE:
                mutation_index = random.randint(0, CHROMOSOME_LENGTH - 1)
                child[mutation_index] = random.choice(emoji_pool)

            new_population.append(child)

        population = new_population

    # STEP 9 – Final output
    best = population[0]
    best_vector = get_combined_vector(best, all_keywords, emoji_data)

    # print("🔬 With fuzzy vector:", get_combined_vector(best))

    return best


# print("📊 Keyword Match Analysis:")
# print(f"{'Keyword':<12} {'Prompt':<8} {'Generated':<10} {'Δ':<6} Status")
# print("-" * 50)

# all_keys =  {k for k, v in target_vector.items() if v > 0.0}
# for key in sorted(all_keys):
#     target_score = target_vector.get(key, 0.0)
#     actual_score = best_vector.get(key, 0.0)
#     delta = abs(target_score - actual_score)
#     status = "✅ match" if delta < 0.2 else ("⚠️ partial" if delta < 0.5 else "❌ miss")
#     print(f"{key:<12} {target_score:<8.2f} {actual_score:<10.2f} {delta:<6.2f} {status}")
