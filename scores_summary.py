import pandas as pd

# === FILES ===
input_files = [
    "openai-gpt4o-creativity-scored.csv",
    "google-gemini2.5pro-creativity-scored.csv",
    "google-gemini2.0flash-creativity-scored.csv",
    "claude-sonnet4.0-creativity-scored.csv",
    "human-creativity-scored.csv",
]

# === MAPPING TO GROUP NAMES ===
group_names = {
    "openai-gpt4o": "GPT-4o",
    "google-gemini2.5pro": "Gemini 2.5 Pro",
    "google-gemini2.0flash": "Gemini 2.0 Flash",
    "claude-sonnet4.0": "Claude 4.0 Sonnet",
    "human": "Human",
}

# === AGGREGATE METRICS ===
results = []

for file in input_files:
    df = pd.read_csv(file)
    base_name = file.replace("-creativity-scored.csv", "")
    group = group_names.get(base_name, base_name)

    novelty_avg = df['novelty_score'].mean()
    coherence_avg = df['coherence_score'].mean()
    emergence_avg = df['emergence_score'].mean()

    results.append({
        "Group": group,
        "Novelty Score (Avg.)": round(novelty_avg, 3),
        "Coherence Score (Avg.)": round(coherence_avg, 3),
        "Emergence Score (Avg.)": round(emergence_avg, 3)
    })

summary_df = pd.DataFrame(results)

# === DISPLAY ===
print(summary_df.to_markdown(index=False))