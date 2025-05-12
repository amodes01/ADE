import os
import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# === Config ===
input_dir = "./brexit-normalized-test/gpt_results"
output_dir = "./brexit-normalized-test/gpt_normalized"
model = SentenceTransformer("all-MiniLM-L6-v2")
similarity_threshold = 0.6  # Lowered for broader normalization

# === Manual alias map ===
alias_map = {
    "Trump": "Donald Trump",
    "President Trump": "Donald Trump",
    "The President": "Donald Trump",
    "Obama": "Barack Obama",
    "President Obama": "Barack Obama",
    "President Barack Obama": "Barack Obama",
    "EU": "European Union",
    "The EU": "European Union",
    "The European Union": "European Union",
    "U.K.": "United Kingdom",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
    "Great Britain": "United Kingdom",
    "US": "United States",
    "The United States": "United States",
    "America": "United States",
    "Republicans": "Republican Party",
    "GOP": "Republican Party",
    "Democrats": "Democratic Party",
    "UKIP": "UK Independence Party",
    "Brexiters": "Brexit supporters",
    "Brexiteers": "Brexit supporters",
    "Leave movement": "Brexit supporters",
    "Remainers": "Remain campaign",
    "Remain voters": "Remain campaign",
    "Leavers": "Brexit supporters",
    "European Commission": "European Union",
    "Angela Merkel": "Chancellor of Germany",
    "German Chancellor": "Chancellor of Germany",
    "Boris Johnson": "Prime Minister Boris Johnson",
    "David Cameron": "Prime Minister David Cameron",
    "France": "French Republic",
    "UK Independence Party (UKIP)": "UK Independence Party",
    "Conservative Party": "Conservative Party (UK)",
    "Labour Party": "Labour Party (UK)",
    "Greens": "Green Party",
    "Russia": "Russian Federation"
}

# === Step 1: Collect references ===
def collect_all_references(base_dir):
    references = set()
    for root, _, files in os.walk(base_dir):
        for file in tqdm(files, desc="Collecting references", unit="file"):
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for entry in data.get("topical_attitudes", []):
                            references.update(entry.get("source", {}).get("references", []))
                            references.update(entry.get("target", {}).get("references", []))
                        for entry in data.get("entity_attitudes", []):
                            references.update(entry.get("entity1", {}).get("references", []))
                            references.update(entry.get("entity2", {}).get("references", []))
                except json.JSONDecodeError as e:
                    print(f"[JSON ERROR] {path} → {e}")
    return list(references)

# === Step 2: Build normalization map ===
def build_reference_map(refs, threshold=0.6):
    # Step 2.1: Apply manual aliasing
    aliased_refs = [alias_map.get(r, r) for r in refs]
    unique_refs = list(set(aliased_refs))  # Remove duplicates introduced by aliasing

    # Step 2.2: Semantic clustering
    embeddings = model.encode(unique_refs, convert_to_tensor=True)
    ref_map = {}
    used = set()

    for i, ref in enumerate(tqdm(unique_refs, desc="Building reference map", unit="ref")):
        if ref in used:
            continue
        ref_map[ref] = ref  # Canonical form
        for j in range(i + 1, len(unique_refs)):
            if unique_refs[j] in used:
                continue
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > threshold:
                ref_map[unique_refs[j]] = ref
                used.add(unique_refs[j])

    # Step 2.3: Final map that includes original refs → final canonical form
    final_map = {}
    for original in refs:
        alias = alias_map.get(original, original)
        final_map[original] = ref_map.get(alias, alias)

    return final_map

# === Step 3: Normalize JSON references ===
def normalize_file(data, ref_map):
    def norm(refs):
        return [ref_map.get(r, r) for r in refs]

    for t in data.get("topical_attitudes", []):
        if "source" in t:
            t["source"]["references"] = norm(t["source"].get("references", []))
        if "target" in t:
            t["target"]["references"] = norm(t["target"].get("references", []))

    for e in data.get("entity_attitudes", []):
        if "entity1" in e:
            e["entity1"]["references"] = norm(e["entity1"].get("references", []))
        if "entity2" in e:
            e["entity2"]["references"] = norm(e["entity2"].get("references", []))

    return data

# === Step 4: Process the dataset ===
start_time = time.time()

all_refs = collect_all_references(input_dir)
ref_map = build_reference_map(all_refs, threshold=similarity_threshold)

for root, _, files in os.walk(input_dir):
    rel_path = os.path.relpath(root, input_dir)
    out_dir = os.path.join(output_dir, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for file in tqdm(files, desc=f"Normalizing files in {rel_path}", unit="file"):
        if file.endswith(".json"):
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_dir, file)
            try:
                with open(in_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                normalized = normalize_file(data, ref_map)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(normalized, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError as e:
                print(f"[JSON ERROR] {in_path} → {e}")
            except Exception as e:
                print(f"[GENERAL ERROR] {in_path} → {e}")

end_time = time.time()
print(f"\nNormalization complete. Output saved to '{output_dir}'")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Normalization complete. Output saved to './brexit-normalized-test/gpt_normalized'
# Time taken: 251.65 seconds