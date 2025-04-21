import json
from pathlib import Path
from itertools import product

def extract_entity(url):
    return url.split('/')[-1] if url.startswith("http://dbpedia.org/resource/") else url

def normalize_fellowships(fellowships):
    return [set(map(extract_entity, group)) for group in fellowships]

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)

# Load files
with open("./brexit-test/polarization/fellowships.json", encoding="utf-8") as f1, open("../test-brexit/gpt_polarization/fellowships.json", encoding="utf-8") as f2:
    data1 = json.load(f1)["fellowships"]
    data2 = json.load(f2)["fellowships"]

fellowships1 = normalize_fellowships(data1)
fellowships2 = normalize_fellowships(data2)

SIMILARITY_THRESHOLD = 0.1
all_similarities = []
best_matches = []
unmatched_pairs = []

# Compare all pairs (cross product)
for i, group1 in enumerate(fellowships1):
    best_score = -1
    best_group2 = None
    best_j = -1
    for j, group2 in enumerate(fellowships2):
        sim = jaccard_similarity(group1, group2)
        all_similarities.append(((i, j), sim))
        if sim > best_score:
            best_score = sim
            best_group2 = group2
            best_j = j
        if sim < SIMILARITY_THRESHOLD:
            unmatched_pairs.append(((i, j), sim, group1, group2))
    best_matches.append((i, best_j, best_score, group1, best_group2))

# Output to file
output_file = "fellowship_comparison_results.txt"
with open(output_file, "w", encoding="utf-8") as out:
    out.write("=== BEST MATCHES ===\n")
    for i, j, sim, g1, g2 in best_matches:
        out.write(f"Group {i} (File 1) best matches Group {j} (File 2) with Jaccard similarity = {sim:.2f}\n")
        out.write(f"Group1: {g1}\n")
        out.write(f"Group2: {g2}\n\n")

    out.write("\n=== UNMATCHED PAIRS (Similarity < {:.2f}) ===\n".format(SIMILARITY_THRESHOLD))
    for (i, j), sim, g1, g2 in unmatched_pairs:
        out.write(f"Group {i} (File 1) vs Group {j} (File 2): Jaccard similarity = {sim:.2f}\n")
        out.write(f"  File 1: {sorted(g1)}\n")
        out.write(f"  File 2: {sorted(g2)}\n\n")

    # Optional: full similarity matrix
    out.write("\n=== FULL SIMILARITIES ===\n")
    for (i, j), sim in all_similarities:
        out.write(f"File 1 Group {i} vs File 2 Group {j}: {sim:.2f}\n")

print(f"Results written to {output_file}")
