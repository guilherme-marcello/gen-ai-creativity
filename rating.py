import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from tqdm import tqdm

# === CONFIG ===
GEMINI_API_KEY = None # Set the Gemini API key here
model_name = 'all-MiniLM-L6-v2'  # Sentence-BERT model for novelty
input_files = [
    "openai-gpt4o-creativity.csv",
    "google-gemini2.5pro-creativity.csv",
    "google-gemini2.0flash-creativity.csv",
    "claude-sonnet4.0-creativity.csv",

    "human-creativity.csv",
]

# === INIT MODELS ===
embedder = SentenceTransformer(model_name)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel('gemini-pro')

# === FUNCTIONS ===

def compute_novelty(concept1, concept2, blended):
    emb1 = embedder.encode(concept1, convert_to_tensor=True)
    emb2 = embedder.encode(concept2, convert_to_tensor=True)
    emb_blend = embedder.encode(blended, convert_to_tensor=True)

    dist1 = 1 - util.pytorch_cos_sim(emb_blend, emb1).item()
    dist2 = 1 - util.pytorch_cos_sim(emb_blend, emb2).item()
    return min(dist1, dist2)

def get_coherence_score_gemini(concept1, concept2, blended_concept):
    prompt = f"""
You are evaluating the coherence of a blended concept created from two source concepts.

Source Concept 1: "{concept1}"
Source Concept 2: "{concept2}"
Blended Concept: "{blended_concept}"

On a scale from 1 to 10, rate how logically coherent and meaningful the blended concept is, considering how well it integrates the source concepts. Just return the number, nothing else.

Score:"""
    try:
        response = gemini.generate_content(prompt)
        content = response.text.strip()
        score = float(next(s for s in content.split() if s.replace('.', '', 1).isdigit()))
        return score
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

# === MAIN LOOP ===

DATASET_DIR = "blended_concepts_dataset"
for file in input_files:
    print(f"\n🔍 Processing {file}...")
    file = f"{DATASET_DIR}/{file}"
    df = pd.read_csv(file)

    novelty_scores = []
    coherence_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        c1 = str(row['concept_1'])
        c2 = str(row['concept_2'])
        blend = str(row['blended_concept'])

        novelty = compute_novelty(c1, c2, blend)
        coherence = get_coherence_score_gemini(c1, c2, blend)

        novelty_scores.append(novelty)
        coherence_scores.append(coherence)

    df['novelty_score'] = novelty_scores
    df['coherence_score'] = coherence_scores

    # Save the scored file
    output_file = file.replace('.csv', '-scored.csv')
    df.to_csv(output_file, index=False)
    print(f"✅ Saved scored results to: {output_file}")
