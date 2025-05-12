import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm  # Import tqdm for progress bars

# Initialize the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_entity(url):
    return url.split('/')[-1] if url.startswith("http://dbpedia.org/resource/") else url

def normalize_fellowships(fellowships):
    return [set(map(extract_entity, group)) for group in fellowships]

def sentence_transformer_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    # Convert sets to lists and encode them
    embeddings1 = model.encode(list(set1), convert_to_tensor=True)
    embeddings2 = model.encode(list(set2), convert_to_tensor=True)
    # Compute cosine similarity
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    # Return the maximum similarity score
    return cosine_scores.max().item()

# Load files
with open("./brexit-test/NEW_SAG/polarization/fellowships.json", encoding="utf-8") as f1, open("../brexit-normalized-test/NEW_SAG_TEST/polarization/fellowships.json", encoding="utf-8") as f2:
    data1 = json.load(f1)["fellowships"]
    data2 = json.load(f2)["fellowships"]

fellowships1 = normalize_fellowships(data1)
fellowships2 = normalize_fellowships(data2)

SIMILARITY_THRESHOLD = 0.1
all_similarities = []
best_matches = []
unmatched_pairs = []

# Compare all pairs (cross product)
for i, group1 in enumerate(tqdm(fellowships1, desc="Processing fellowships1")):  # Add tqdm for outer loop
    total_sim = 0
    best_score = -1
    best_group2 = None
    best_j = -1
    for j, group2 in enumerate(tqdm(fellowships2, desc=f"Comparing group {i}", leave=False)):  # Add tqdm for inner loop
        sim = sentence_transformer_similarity(group1, group2)
        total_sim += sim
        all_similarities.append({"file1_group": i, "file2_group": j, "similarity": sim})
        if sim > best_score:
            best_score = sim
            best_group2 = group2
            best_j = j
        if sim < SIMILARITY_THRESHOLD:
            unmatched_pairs.append({
                "file1_group": i,
                "file2_group": j,
                "similarity": sim,
                "file1_entities": sorted(group1),
                "file2_entities": sorted(group2)
            })
    # Normalize the total similarity by the number of comparisons
    normalized_sim = total_sim / len(fellowships2) if fellowships2 else 0
    best_matches.append({
        "file1_group": i,
        "file2_group": best_j,
        "similarity": normalized_sim,
        "file1_entities": sorted(group1),
        "file2_entities": sorted(best_group2) if best_group2 else None
    })

# Output to JSON file
output_file = "fellowship_comparison_results.json"
output_data = {
    "best_matches": best_matches,
    "unmatched_pairs": unmatched_pairs,
    "all_similarities": all_similarities
}

with open(output_file, "w", encoding="utf-8") as out:
    json.dump(output_data, out, indent=4)

print(f"Results written to {output_file}")
