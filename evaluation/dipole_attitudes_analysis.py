import pickle
import numpy as np
from collections import Counter
import math
import json
import os

def load_data(prefix):
    """Load attitudes and node mapping for a dataset"""
    with open(f"{prefix}/attitudes.pckl", "rb") as f:
        attitudes = pickle.load(f)
    with open(f"{prefix}/int_to_node.pckl", "rb") as f:
        int_to_node = pickle.load(f)
    return attitudes, int_to_node

def analyze_dataset(attitudes, dataset_name):
    """Calculate all metrics for a single dataset"""
    # Basic counts
    num_attitudes = len(attitudes)
    topics = [entry['topic']['nps'][0] if entry['topic']['nps'] else "Unknown" 
              for entry in attitudes]
    unique_topics = set(topics)
    
    # Polarization metrics
    polarization_scores = [entry.get('pi_res', entry.get('pi', 0)) for entry in attitudes]
    avg_polarization = np.mean(polarization_scores)
    clarity_score = np.mean([abs(pi - 0.5) * 2 for pi in polarization_scores])  # 0-1 scale
    
    # Topic analysis
    topic_counts = Counter(topics)
    top_10_topics = topic_counts.most_common(10)
    
    # Sociopolitical detection (simple keyword-based)
    socio_keywords = {
    "government",
    "referendum",
    "citizen",
    "chief executive",
    "vote",
    "campaign",
    "sanction",
    "independence",
    "sovereignty",
    "influence",
    "control",
    "former mayor",
    "brexit",
    "security",
    "country",
    "right-wing populism",
    "minimum wage",
    "populism",
    "globalization",
    "elite political class",
    "migrant crisis",
    "inequality",
    "economic inequality",
    "immigration",
    "political uncertainty"
}

    socio_topics = [t for t in topics if any(kw in t.lower() for kw in socio_keywords)]
    pct_sociopolitical = len(socio_topics) / num_attitudes * 100
    
    # Topic entropy
    topic_probs = [count/num_attitudes for count in topic_counts.values()]
    entropy = -sum(p * math.log(p) for p in topic_probs if p > 0)
    return {
        "Dataset": dataset_name,
        "Total Attitudes": num_attitudes,
        "Unique Topics": len(unique_topics),
        "Avg Polarization": round(avg_polarization, 3),
        "Polarization Clarity": round(clarity_score, 3),
        "% Sociopolitical": round(pct_sociopolitical, 1),
        "Topic Entropy": round(entropy, 3),
        "Top 10 Topics": top_10_topics,
        "Node Pairs": len({frozenset(entry['dipole']) for entry in attitudes}),
        "Max Polarization": round(max(polarization_scores), 3),
        "Min Polarization": round(min(polarization_scores), 3)
    }


# Load both datasets
gpt_data, gpt_nodes = load_data("../brexit-normalized-test/polarization")
polar_data, polar_nodes = load_data("./brexit-test/polarization")

# Generate reports
gpt_report = analyze_dataset(gpt_data, "GPT")
polar_report = analyze_dataset(polar_data, "POLAR")

# print("GPT Unique Topics:")
# for topic in set(entry['topic']['nps'][0] if entry['topic']['nps'] else "Unknown" for entry in gpt_data):
#     print(topic)

# print("\nPOLAR Unique Topics:")
# for topic in set(entry['topic']['nps'][0] if entry['topic']['nps'] else "Unknown" for entry in polar_data):
#     print(topic)

# Save to JSON files
with open('gpt_attitude_dipoles_analysis.json', 'w') as f:
    json.dump(gpt_report, f, indent=2)

with open('polar_attitude_dipoles_analysis.json', 'w') as f:
    json.dump(polar_report, f, indent=2)

# Also create a combined comparison file
comparison = {
    "GPT": gpt_report,
    "POLAR": polar_report
}
with open('attitude_dipoles_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("Analysis complete.")