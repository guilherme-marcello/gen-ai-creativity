import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# === CONFIG ===
input_files = [
    "openai-gpt4o-creativity-scored.csv",
    "google-gemini2.5pro-creativity-scored.csv",
    "google-gemini2.0flash-creativity-scored.csv",
    "claude-sonnet4.0-creativity-scored.csv",
    "human-creativity-scored.csv",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === COLLECTED DATA ===
all_rows = []

for file in input_files:
    print(f"🔍 Processing {file}...")
    df = pd.read_csv(file)
    is_human = 1 if "human" in file else 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Computing features"):
        c1 = str(row['concept_1'])
        c2 = str(row['concept_2'])

        emb1 = embedder.encode(c1, convert_to_tensor=True)
        emb2 = embedder.encode(c2, convert_to_tensor=True)

        concepts_distance = 1 - util.pytorch_cos_sim(emb1, emb2).item()

        new_row = {
            "concepts_distance": concepts_distance,
            "novelty_score": row['novelty_score'],
            "coherence_score": row['coherence_score'],
            "emergence_score": row['emergence_score'],
            "human_generated": is_human
        }

        all_rows.append(new_row)

# === FINAL DATASET ===
final_df = pd.DataFrame(all_rows)
final_df.to_csv("creativity_classification_dataset.csv", index=False)
print("✅ Saved: creativity_classification_dataset.csv")
