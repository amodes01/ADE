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
        "trump": "donald trump",
        "hillary clinton": "hillary clinton",
        "hillary": "hillary clinton",
        "bori johnson": "boris johnson",
        "reuter": "reuters",
        "britain vote": "brexit",
        "the uk sign": "united kingdom",
        "the vote": "vote",
        "europe": "european union",
        "union": "european union",  # when it appears alone
        "eu": "european union",
        "great britain": "united kingdom",
        "briton": "united kingdom",
        "american": "united states",    
        "united state": "united states",   
        "united state of america": "united states", 
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

# Identify overlapping pairs
overlap_pairs = {}

for section in tqdm(['entity', 'topical'], desc="Identifying overlaps"):
    true_pairs = set(results[section]['true_data'].keys())
    pred_pairs = set(results[section]['pred_data'].keys())
    overlap = true_pairs & pred_pairs  # exact match of normalized pairs

    overlap_pairs[section] = []

    for pair in tqdm(sorted(overlap, key=lambda x: x[0] + x[1]), desc=f"Processing overlaps for {section}", leave=False):
        overlap_pairs[section].append({
            "pair": pair,
            "true_data": results[section]['true_data'][pair],
            "pred_data": results[section]['pred_data'][pair]
        })

# Save both cleaned results and overlaps
output = {
    "grouped_pair_frequencies": results,
    "overlap_pairs": overlap_pairs
}

with open('grouped_pair_frequencies_with_overlap.json', 'w') as f_out:
    json.dump(output, f_out, indent=2)

# Optional: preview some overlaps in descending order of counts
for section in ['entity', 'topical']:
    print(f"\n== OVERLAPPING {section.upper()} PAIRS ({len(overlap_pairs[section])}) ==")
    sorted_overlaps = sorted(
        overlap_pairs[section],
        key=lambda x: x['true_data']['count'] + x['pred_data']['count'],
        reverse=True
    )
    for entry in sorted_overlaps[:10]:  # show top 10
        true_attitudes = entry['true_data']['attitudes']
        pred_attitudes = entry['pred_data']['attitudes']
        print(
            f"{entry['pair']} → TRUE: {entry['true_data']['count']} "
            f"(Neg: {true_attitudes['Negative']}, Neu: {true_attitudes['Neutral']}, Pos: {true_attitudes['Positive']}) | "
            f"PRED: {entry['pred_data']['count']} "
            f"(Neg: {pred_attitudes['Negative']}, Neu: {pred_attitudes['Neutral']}, Pos: {pred_attitudes['Positive']})"
        )


# Top 20: ENTITY - TRUE_DATA
# Pair: ('european union', 'united kingdom'), Count: 72, Attitudes: {'Positive': 7, 'Neutral': 18, 'Negative': 47}
# Pair: ('united kingdom', 'united states'), Count: 29, Attitudes: {'Positive': 18, 'Neutral': 6, 'Negative': 5}
# Pair: ('donald trump', 'hillary clinton'), Count: 21, Attitudes: {'Positive': 2, 'Neutral': 0, 'Negative': 19}
# Pair: ('barack obama', 'united kingdom'), Count: 16, Attitudes: {'Positive': 7, 'Neutral': 4, 'Negative': 5}
# Pair: ('nato', 'russia'), Count: 15, Attitudes: {'Positive': 2, 'Neutral': 2, 'Negative': 11}
# Pair: ('brexit', 'donald trump'), Count: 13, Attitudes: {'Positive': 6, 'Neutral': 2, 'Negative': 5}
# Pair: ('donald trump', 'nigel farage'), Count: 11, Attitudes: {'Positive': 3, 'Neutral': 8, 'Negative': 0}
# Pair: ('european union', 'united states'), Count: 11, Attitudes: {'Positive': 3, 'Neutral': 3, 'Negative': 5}
# Pair: ('barack obama', 'european union'), Count: 11, Attitudes: {'Positive': 6, 'Neutral': 4, 'Negative': 1}
# Pair: ('barack obama', 'david cameron'), Count: 10, Attitudes: {'Positive': 5, 'Neutral': 4, 'Negative': 1}
# Pair: ('barack obama', 'donald trump'), Count: 10, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 10}
# Pair: ('I', 'united kingdom'), Count: 10, Attitudes: {'Positive': 4, 'Neutral': 3, 'Negative': 3}
# Pair: ('boris johnson', 'donald trump'), Count: 9, Attitudes: {'Positive': 1, 'Neutral': 7, 'Negative': 1}
# Pair: ('bernie sander', 'donald trump'), Count: 8, Attitudes: {'Positive': 2, 'Neutral': 4, 'Negative': 2}
# Pair: ('david cameron', 'nigel farage'), Count: 8, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 8}
# Pair: ('brexit', 'david cameron'), Count: 8, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 8}
# Pair: ('germany', 'united kingdom'), Count: 8, Attitudes: {'Positive': 1, 'Neutral': 5, 'Negative': 2}
# Pair: ('boris johnson', 'david cameron'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 1, 'Negative': 5}
# Pair: ('india', 'united kingdom'), Count: 7, Attitudes: {'Positive': 7, 'Neutral': 0, 'Negative': 0}
# Pair: ('france', 'united kingdom'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 1, 'Negative': 5}

# Top 20: ENTITY - PRED_DATA
# Pair: ('european union', 'united kingdom'), Count: 109, Attitudes: {'Positive': 15, 'Neutral': 80, 'Negative': 14}
# Pair: ('brexit', 'european union'), Count: 44, Attitudes: {'Positive': 10, 'Neutral': 24, 'Negative': 10}
# Pair: ('brexit', 'united kingdom'), Count: 43, Attitudes: {'Positive': 6, 'Neutral': 28, 'Negative': 9}
# Pair: ('barack obama', 'european union'), Count: 13, Attitudes: {'Positive': 3, 'Neutral': 6, 'Negative': 4}
# Pair: ('european union', 'reuters'), Count: 11, Attitudes: {'Positive': 2, 'Neutral': 9, 'Negative': 0}
# Pair: ('european union', 'london'), Count: 10, Attitudes: {'Positive': 1, 'Neutral': 7, 'Negative': 2}
# Pair: ('barack obama', 'brexit'), Count: 10, Attitudes: {'Positive': 3, 'Neutral': 4, 'Negative': 3}
# Pair: ('barack obama', 'united kingdom'), Count: 8, Attitudes: {'Positive': 3, 'Neutral': 4, 'Negative': 1}
# Pair: ('brexit', 'donald trump'), Count: 7, Attitudes: {'Positive': 1, 'Neutral': 6, 'Negative': 0}
# Pair: ('david cameron', 'european union'), Count: 6, Attitudes: {'Positive': 1, 'Neutral': 3, 'Negative': 2}
# Pair: ('united kingdom', 'united states'), Count: 6, Attitudes: {'Positive': 1, 'Neutral': 5, 'Negative': 0}
# Pair: ('reuters', 'united kingdom'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 6, 'Negative': 0}
# Pair: ('european union', 'russia'), Count: 5, Attitudes: {'Positive': 3, 'Neutral': 1, 'Negative': 1}
# Pair: ('brussel', 'european union'), Count: 5, Attitudes: {'Positive': 3, 'Neutral': 2, 'Negative': 0}
# Pair: ('I president', 'european union'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('london', 'reuters'), Count: 4, Attitudes: {'Positive': 2, 'Neutral': 1, 'Negative': 1}
# Pair: ('brexit', 'brussel'), Count: 4, Attitudes: {'Positive': 3, 'Neutral': 1, 'Negative': 0}
# Pair: ('brexit', 'united states'), Count: 4, Attitudes: {'Positive': 1, 'Neutral': 3, 'Negative': 0}
# Pair: ('donald trump', 'republican'), Count: 4, Attitudes: {'Positive': 0, 'Neutral': 4, 'Negative': 0}
# Pair: ('I president', 'barack obama'), Count: 4, Attitudes: {'Positive': 1, 'Neutral': 0, 'Negative': 3}

# Top 20: TOPICAL - TRUE_DATA
# Pair: ('brexit', 'european union'), Count: 70, Attitudes: {'Positive': 2, 'Neutral': 10, 'Negative': 58}
# Pair: ('brexit', 'united kingdom'), Count: 34, Attitudes: {'Positive': 3, 'Neutral': 9, 'Negative': 22}
# Pair: ('barack obama', 'brexit'), Count: 24, Attitudes: {'Positive': 2, 'Neutral': 8, 'Negative': 14}
# Pair: ('brexit', 'united states'), Count: 15, Attitudes: {'Positive': 3, 'Neutral': 4, 'Negative': 8}
# Pair: ('brexit', 'donald trump'), Count: 14, Attitudes: {'Positive': 11, 'Neutral': 1, 'Negative': 2}
# Pair: ('european union', 'united kingdom'), Count: 12, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 7}
# Pair: ('brexit', 'nigel farage'), Count: 11, Attitudes: {'Positive': 9, 'Neutral': 0, 'Negative': 2}
# Pair: ('boris johnson', 'brexit'), Count: 8, Attitudes: {'Positive': 7, 'Neutral': 0, 'Negative': 1}
# Pair: ('brexit', 'david cameron'), Count: 8, Attitudes: {'Positive': 0, 'Neutral': 1, 'Negative': 7}
# Pair: ('brexit', 'immigration'), Count: 8, Attitudes: {'Positive': 1, 'Neutral': 0, 'Negative': 7}
# Pair: ('donald trump', 'immigration'), Count: 8, Attitudes: {'Positive': 1, 'Neutral': 0, 'Negative': 7}
# Pair: ('brexit', 'president barack obama'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 4}
# Pair: ('brexit', 'uk economy'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 0, 'Negative': 7}
# Pair: ('brexit', 'hillary clinton'), Count: 6, Attitudes: {'Positive': 1, 'Neutral': 1, 'Negative': 4}
# Pair: ('brexit', 'nicola sturgeon'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 4}
# Pair: ('capitalism', 'european'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 4}
# Pair: ('brexit', 'erik bidenkap'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 2}
# Pair: ('barack obama', 'globalization'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 3, 'Negative': 2}
# Pair: ('eu membership', 'scotland'), Count: 5, Attitudes: {'Positive': 4, 'Neutral': 1, 'Negative': 0}
# Pair: ('nato', 'russia'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 3}

# Top 20: TOPICAL - PRED_DATA
# Pair: ('european union', 'united kingdom'), Count: 47, Attitudes: {'Positive': 7, 'Neutral': 34, 'Negative': 6}
# Pair: ('brexit', 'united kingdom'), Count: 21, Attitudes: {'Positive': 3, 'Neutral': 16, 'Negative': 2}
# Pair: ('brexit', 'european union'), Count: 20, Attitudes: {'Positive': 0, 'Neutral': 14, 'Negative': 6}
# Pair: ('european union', 'the world'), Count: 11, Attitudes: {'Positive': 4, 'Neutral': 6, 'Negative': 1}
# Pair: ('britain decision', 'european union'), Count: 10, Attitudes: {'Positive': 0, 'Neutral': 10, 'Negative': 0}
# Pair: ('brexit', 'united states'), Count: 9, Attitudes: {'Positive': 4, 'Neutral': 4, 'Negative': 1}
# Pair: ('the world', 'united kingdom'), Count: 8, Attitudes: {'Positive': 4, 'Neutral': 3, 'Negative': 1}
# Pair: ('brexit', 'the late headline'), Count: 7, Attitudes: {'Positive': 0, 'Neutral': 7, 'Negative': 0}
# Pair: ('european union', 'friday'), Count: 7, Attitudes: {'Positive': 4, 'Neutral': 3, 'Negative': 0}
# Pair: ('brexit', 'the world'), Count: 6, Attitudes: {'Positive': 2, 'Neutral': 2, 'Negative': 2}
# Pair: ('a historic referendum', 'european union'), Count: 6, Attitudes: {'Positive': 0, 'Neutral': 6, 'Negative': 0}
# Pair: ('london', 'united kingdom'), Count: 6, Attitudes: {'Positive': 3, 'Neutral': 3, 'Negative': 0}
# Pair: ('brexit', 'the late insight'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('a valid email addre', 'brexit'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('friday', 'reuters'), Count: 5, Attitudes: {'Positive': 3, 'Neutral': 2, 'Negative': 0}
# Pair: ('london', 'reuters'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 5, 'Negative': 0}
# Pair: ('a referendum', 'united kingdom'), Count: 5, Attitudes: {'Positive': 2, 'Neutral': 3, 'Negative': 0}
# Pair: ('barack obama', 'united kingdom'), Count: 5, Attitudes: {'Positive': 1, 'Neutral': 0, 'Negative': 4}
# Pair: ('european union', 'the country'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 2, 'Negative': 3}
# Pair: ('european union', 'june'), Count: 5, Attitudes: {'Positive': 0, 'Neutral': 4, 'Negative': 1}

# == OVERLAPPING ENTITY PAIRS (113) ==
# ('european union', 'united kingdom') → TRUE: 72 (Neg: 47, Neu: 18, Pos: 7) | PRED: 109 (Neg: 14, Neu: 80, Pos: 15)
# ('brexit', 'european union') → TRUE: 7 (Neg: 7, Neu: 0, Pos: 0) | PRED: 44 (Neg: 10, Neu: 24, Pos: 10)
# ('brexit', 'united kingdom') → TRUE: 5 (Neg: 1, Neu: 2, Pos: 2) | PRED: 43 (Neg: 9, Neu: 28, Pos: 6)
# ('united kingdom', 'united states') → TRUE: 29 (Neg: 5, Neu: 6, Pos: 18) | PRED: 6 (Neg: 0, Neu: 5, Pos: 1)
# ('barack obama', 'european union') → TRUE: 11 (Neg: 1, Neu: 4, Pos: 6) | PRED: 13 (Neg: 4, Neu: 6, Pos: 3)
# ('barack obama', 'united kingdom') → TRUE: 16 (Neg: 5, Neu: 4, Pos: 7) | PRED: 8 (Neg: 1, Neu: 4, Pos: 3)
# ('donald trump', 'hillary clinton') → TRUE: 21 (Neg: 19, Neu: 0, Pos: 2) | PRED: 1 (Neg: 0, Neu: 1, Pos: 0)
# ('brexit', 'donald trump') → TRUE: 13 (Neg: 5, Neu: 2, Pos: 6) | PRED: 7 (Neg: 0, Neu: 6, Pos: 1)
# ('nato', 'russia') → TRUE: 15 (Neg: 11, Neu: 2, Pos: 2) | PRED: 3 (Neg: 0, Neu: 2, Pos: 1)
# ('barack obama', 'brexit') → TRUE: 6 (Neg: 3, Neu: 2, Pos: 1) | PRED: 10 (Neg: 3, Neu: 4, Pos: 3)

# == OVERLAPPING TOPICAL PAIRS (45) ==
# ('brexit', 'european union') → TRUE: 70 (Neg: 58, Neu: 10, Pos: 2) | PRED: 20 (Neg: 6, Neu: 14, Pos: 0)
# ('european union', 'united kingdom') → TRUE: 12 (Neg: 7, Neu: 5, Pos: 0) | PRED: 47 (Neg: 6, Neu: 34, Pos: 7)
# ('brexit', 'united kingdom') → TRUE: 34 (Neg: 22, Neu: 9, Pos: 3) | PRED: 21 (Neg: 2, Neu: 16, Pos: 3)
# ('barack obama', 'brexit') → TRUE: 24 (Neg: 14, Neu: 8, Pos: 2) | PRED: 1 (Neg: 0, Neu: 1, Pos: 0)
# ('brexit', 'united states') → TRUE: 15 (Neg: 8, Neu: 4, Pos: 3) | PRED: 9 (Neg: 1, Neu: 4, Pos: 4)
# ('european union', 'united states') → TRUE: 4 (Neg: 1, Neu: 0, Pos: 3) | PRED: 5 (Neg: 1, Neu: 4, Pos: 0)
# ('boris johnson', 'brexit') → TRUE: 8 (Neg: 1, Neu: 0, Pos: 7) | PRED: 1 (Neg: 0, Neu: 1, Pos: 0)
# ('brexit', 'david cameron') → TRUE: 8 (Neg: 7, Neu: 1, Pos: 0) | PRED: 1 (Neg: 0, Neu: 1, Pos: 0)
# ('barack obama', 'european union') → TRUE: 5 (Neg: 1, Neu: 2, Pos: 2) | PRED: 3 (Neg: 1, Neu: 1, Pos: 1)
# ('united kingdom', 'united states') → TRUE: 2 (Neg: 1, Neu: 1, Pos: 0) | PRED: 5 (Neg: 1, Neu: 3, Pos: 1)