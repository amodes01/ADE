import json
import re
import inflect
import spacy
from collections import defaultdict
from tqdm import tqdm

# Setup
inflector = inflect.engine()
nlp = spacy.load("en_core_web_sm")

# Junk and low-value words to skip
BLACKLIST_TERMS = {
    "update", "event", "email", "offer", "thursday", "headline", 
    "the independent", "sign", "vote", "reuter"
}

# Normalize entity/topic string
def normalize(term):
    term = term.lower()
    term = re.sub(r"[^\w\s]", "", term)
    term = " ".join([
        inflector.singular_noun(word) if inflector.singular_noun(word) else word
        for word in term.split()
    ])
    doc = nlp(term)
    term = " ".join([token.lemma_ for token in doc])

    aliases = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "british": "united kingdom",
        "britain": "united kingdom",
        "eu": "european union",
        "e.u.": "european union",
        "us": "united states",
        "u.s.": "united states",
        "america": "united states",
        "president obama": "barack obama",
        "barack h. obama": "barack obama",
        "obama": "barack obama",
        "president trump": "donald trump",
        "donald j. trump": "donald trump",
        "bori johnson": "boris johnson",
        "reuter": "reuters",
        "britain vote": "brexit",
        "the uk sign": "united kingdom",
        "the vote": "vote",
    }
    return aliases.get(term.strip(), term.strip())

def normalize_pair(pair):
    return tuple(sorted(normalize(p) for p in pair))

def is_valid_pair(pair):
    if pair[0] == pair[1]:
        return False
    if any(term in BLACKLIST_TERMS for term in pair):
        return False
    return True

def group_pairs(entries):
    grouped = defaultdict(list)
    for entry in tqdm(entries, desc="Grouping pairs", leave=False):
        norm_pair = normalize_pair(entry['pair'])
        if is_valid_pair(norm_pair):
            grouped[norm_pair].append(entry)
    return grouped

def combine_grouped_entries(grouped):
    combined = {}
    for norm_pair, entries in tqdm(grouped.items(), desc="Combining grouped entries", leave=False):
        total_count = sum(e['count'] for e in entries)
        attitudes = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for e in entries:
            for att in attitudes:
                attitudes[att] += e['attitudes'].get(att, 0)
        combined[norm_pair] = {'count': total_count, 'attitudes': attitudes}
    return combined

# Load data
with open('brexit_detailed_evaluation_results.json', 'r') as f:
    data = json.load(f)

results = {}

for section in tqdm(['entity', 'topical'], desc="Processing sections"):
    print(f"\n=== PROCESSING: {section.upper()} ===")
    results[section] = {}

    for dataset in tqdm(['true_data', 'pred_data'], desc=f"Processing datasets for {section}", leave=False):
        print(f"  -> Normalizing and grouping: {dataset}")
        entries = data.get(section, {}).get('pair_frequencies', {}).get(dataset, [])
        grouped = group_pairs(entries)
        combined = combine_grouped_entries(grouped)
        print(f"     Original: {len(entries)} → Grouped (valid): {len(combined)}")
        # Convert tuple keys to strings for JSON compatibility
        results[section][dataset] = {str(k): v for k, v in combined.items()}

# Save output
with open('grouped_pair_frequencies_cleaned.json', 'w') as f_out:
    json.dump(results, f_out, indent=2)

# Display top 20
for section in ['entity', 'topical']:
    for dataset in ['true_data', 'pred_data']:
        print(f"\nTop 20: {section.upper()} - {dataset.upper()}")
        sorted_items = sorted(results[section][dataset].items(), key=lambda x: x[1]['count'], reverse=True)
        for pair, stats in sorted_items[:20]:
            print(f"Pair: {pair}, Count: {stats['count']}, Attitudes: {stats['attitudes']}")


# Top 20: ENTITY - TRUE_DATA
# Pair: ('european union', 'united kingdom'), Count: 71, Attitudes: {'Positive': 7, 'Neutral': 18, 'Negative': 46}
# Pair: ('united kingdom', 'united state'), Count: 25, Attitudes: {'Positive': 15, 'Neutral': 5, 'Negative': 5}
# Pair: ('donald trump', 'hillary clinton'), Count: 20, Attitudes: {'Positive': 2, 'Neutral': 0, 'Negative': 18}
# Pair: ('nato', 'russia'), Count: 15, Attitudes: {'Positive': 2, 'Neutral': 2, 'Negative': 11}
# Pair: ('barack obama', 'united kingdom'), Count: 14, Attitudes: {'Positive': 7, 'Neutral': 2, 'Negative': 5}
# Pair: ('brexit', 'donald trump'), Count: 11, Attitudes: {'Positive': 6, 'Neutral': 1, 'Negative': 4}
# Pair: ('donald trump', 'nigel farage'), Count: 11, Attitudes: {'Positive': 3, 'Neutral': 8, 'Negative': 0}
# Pair: ('barack obama', 'david cameron'), Count: 10, Attitudes: {'Positive': 5, 'Neutral': 4, 'Negative': 1}
# Pair: ('barack obama', 'donald trump'), Count: 10, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 10}
# Pair: ('I', 'united kingdom'), Count: 10, Attitudes: {'Positive': 4, 'Neutral': 3, 'Negative': 3}
# Pair: ('boris johnson', 'donald trump'), Count: 9, Attitudes: {'Positive': 1, 'Neutral': 7, 'Negative': 1}
# Pair: ('barack obama', 'european union'), Count: 9, Attitudes: {'Positive': 5, 'Neutral': 3, 'Negative': 1}
# Pair: ('bernie sander', 'donald trump'), Count: 8, Attitudes: {'Positive': 2, 'Neutral': 4, 'Negative': 2}
# Pair: ('david cameron', 'nigel farage'), Count: 8, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 8}
# Pair: ('brexit', 'david cameron'), Count: 8, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 8}
# Pair: ('germany', 'united kingdom'), Count: 8, Attitudes: {'Positive': 1, 'Neutral': 5, 'Negative': 2}
# Pair: ('boris johnson', 'david cameron'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 1, 'Negative': 5}
# Pair: ('european union', 'united state'), Count: 7, Attitudes: {'Positive': 3, 'Neutral': 1, 'Negative': 3}
# Pair: ('india', 'united kingdom'), Count: 7, Attitudes: {'Positive': 7, 'Neutral': 0, 'Negative': 0}
# Pair: ('france', 'united kingdom'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 1, 'Negative': 5}

# Top 20: ENTITY - PRED_DATA
# Pair: ('union', 'united kingdom'), Count: 66, Attitudes: {'Positive': 9, 'Neutral': 47, 'Negative': 10}
# Pair: ('brexit', 'united kingdom'), Count: 40, Attitudes: {'Positive': 4, 'Neutral': 27, 'Negative': 9}
# Pair: ('european union', 'united kingdom'), Count: 32, Attitudes: {'Positive': 3, 'Neutral': 25, 'Negative': 4}
# Pair: ('brexit', 'union'), Count: 30, Attitudes: {'Positive': 5, 'Neutral': 17, 'Negative': 8}
# Pair: ('brexit', 'european union'), Count: 14, Attitudes: {'Positive': 5, 'Neutral': 7, 'Negative': 2}
# Pair: ('london', 'union'), Count: 10, Attitudes: {'Positive': 1, 'Neutral': 7, 'Negative': 2}
# Pair: ('reuters', 'union'), Count: 10, Attitudes: {'Positive': 2, 'Neutral': 8, 'Negative': 0}
# Pair: ('barack obama', 'union'), Count: 10, Attitudes: {'Positive': 3, 'Neutral': 4, 'Negative': 3}
# Pair: ('european union', 'union'), Count: 10, Attitudes: {'Positive': 1, 'Neutral': 8, 'Negative': 1}
# Pair: ('barack obama', 'brexit'), Count: 10, Attitudes: {'Positive': 3, 'Neutral': 4, 'Negative': 3}
# Pair: ('barack obama', 'united kingdom'), Count: 7, Attitudes: {'Positive': 3, 'Neutral': 3, 'Negative': 1}
# Pair: ('briton', 'union'), Count: 6, Attitudes: {'Positive': 2, 'Neutral': 4, 'Negative': 0}
# Pair: ('brexit', 'donald trump'), Count: 5, Attitudes: {'Positive': 1, 'Neutral': 4, 'Negative': 0}
# Pair: ('united kingdom', 'united state'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('london', 'reuters'), Count: 4, Attitudes: {'Positive': 2, 'Neutral': 1, 'Negative': 1}
# Pair: ('brexit', 'brussel'), Count: 4, Attitudes: {'Positive': 3, 'Neutral': 1, 'Negative': 0}
# Pair: ('brexit', 'united state'), Count: 4, Attitudes: {'Positive': 1, 'Neutral': 3, 'Negative': 0}
# Pair: ('donald trump', 'republican'), Count: 4, Attitudes: {'Positive': 0, 'Neutral': 4, 'Negative': 0}
# Pair: ('david cameron', 'union'), Count: 4, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 1}
# Pair: ('great britain', 'union'), Count: 4, Attitudes: {'Positive': 1, 'Neutral': 3, 'Negative': 0}

# Top 20: TOPICAL - TRUE_DATA
# Pair: ('brexit', 'european union'), Count: 69, Attitudes: {'Positive': 2, 'Neutral': 9, 'Negative': 58}
# Pair: ('brexit', 'united kingdom'), Count: 32, Attitudes: {'Positive': 2, 'Neutral': 9, 'Negative': 21}
# Pair: ('barack obama', 'brexit'), Count: 24, Attitudes: {'Positive': 2, 'Neutral': 8, 'Negative': 14}
# Pair: ('brexit', 'donald trump'), Count: 12, Attitudes: {'Positive': 10, 'Neutral': 1, 'Negative': 1}
# Pair: ('brexit', 'nigel farage'), Count: 11, Attitudes: {'Positive': 9, 'Neutral': 0, 'Negative': 2}
# Pair: ('brexit', 'united state'), Count: 11, Attitudes: {'Positive': 2, 'Neutral': 4, 'Negative': 5}
# Pair: ('european union', 'united kingdom'), Count: 11, Attitudes: {'Positive': 0, 'Neutral': 4, 'Negative': 7}
# Pair: ('boris johnson', 'brexit'), Count: 8, Attitudes: {'Positive': 7, 'Neutral': 0, 'Negative': 1}
# Pair: ('brexit', 'david cameron'), Count: 8, Attitudes: {'Positive': 0, 'Neutral': 1, 'Negative': 7}
# Pair: ('brexit', 'immigration'), Count: 8, Attitudes: {'Positive': 1, 'Neutral': 0, 'Negative': 7}
# Pair: ('brexit', 'president barack obama'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 4}
# Pair: ('donald trump', 'immigration'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 0, 'Negative': 6}
# Pair: ('brexit', 'uk economy'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 7}
# Pair: ('brexit', 'hillary clinton'), Count: 6, Attitudes: {'Positive': 1, 'Neutral': 1, 'Negative': 4}
# Pair: ('brexit', 'nicola sturgeon'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 4}
# Pair: ('capitalism', 'european'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 4}
# Pair: ('brexit', 'erik bidenkap'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 2}
# Pair: ('barack obama', 'globalization'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 2}
# Pair: ('eu membership', 'scotland'), Count: 5, Attitudes: {'Positive': 4, 'Neutral': 1, 'Negative': 0}
# Pair: ('nato', 'russia'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 3}

# Top 20: TOPICAL - PRED_DATA
# Pair: ('union', 'united kingdom'), Count: 33, Attitudes: {'Positive': 7, 'Neutral': 23, 'Negative': 3}
# Pair: ('brexit', 'united kingdom'), Count: 21, Attitudes: {'Positive': 3, 'Neutral': 16, 'Negative': 2}
# Pair: ('brexit', 'union'), Count: 12, Attitudes: {'Positive': 0, 'Neutral': 10, 'Negative': 2}
# Pair: ('american', 'brexit'), Count: 8, Attitudes: {'Positive': 4, 'Neutral': 3, 'Negative': 1}
# Pair: ('the world', 'union'), Count: 8, Attitudes: {'Positive': 3, 'Neutral': 4, 'Negative': 1}
# Pair: ('brexit', 'the late headline'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 7, 'Negative': 0}
# Pair: ('europe', 'union'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 5, 'Negative': 1}
# Pair: ('friday', 'union'), Count: 7, Attitudes: {'Positive': 4, 'Neutral': 3, 'Negative': 0}
# Pair: ('brexit', 'europe'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 4}
# Pair: ('britain decision', 'union'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 7, 'Negative': 0}
# Pair: ('europe', 'united kingdom'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 6, 'Negative': 1}
# Pair: ('european union', 'united kingdom'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 4, 'Negative': 2}
# Pair: ('brexit', 'the world'), Count: 6, Attitudes: {'Positive': 2, 'Neutral': 2, 'Negative': 2}
# Pair: ('the world', 'united kingdom'), Count: 6, Attitudes: {'Positive': 2, 'Neutral': 3, 'Negative': 1}
# Pair: ('brexit', 'the late insight'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('a valid email addre', 'brexit'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('friday', 'reuters'), Count: 5, Attitudes: {'Positive': 3, 'Neutral': 2, 'Negative': 0}
# Pair: ('london', 'reuters'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('london', 'united kingdom'), Count: 5, Attitudes: {'Positive': 3, 'Neutral': 2, 'Negative': 0}
# Pair: ('a referendum', 'united kingdom'), Count: 5, Attitudes: {'Positive': 2, 'Neutral': 3, 'Negative': 0}