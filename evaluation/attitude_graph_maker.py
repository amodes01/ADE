import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# Load data
with open("brexit_detailed_evaluation_results.json", "r") as f:
    data = json.load(f)

# Prepare per-sample attitude storage
per_sample_counts = {
    "entity": {"true_data": [], "pred_data": []},
    "topical": {"true_data": [], "pred_data": []},
}

# Extract per-sample attitude summaries
samples = data.get("samples", {})
for sample_id, sample in tqdm(samples.items(), desc="Collecting per-sample data"):
    for data_type in ["true_data", "pred_data"]:
        summary = sample.get(data_type, {}).get("attitude_summary", {})
        for section in ["entity", "topical"]:
            counts = summary.get(section, {})
            per_sample_counts[section][data_type].append({
                "Negative": counts.get("Negative", 0),
                "Neutral": counts.get("Neutral", 0),
                "Positive": counts.get("Positive", 0)
            })

# Prepare the data for Seaborn plots
def prepare_plot_data(section, data_type, sample_list):
    # Convert data into a format suitable for Seaborn (long format DataFrame)
    plot_data = []
    for i, sample in enumerate(sample_list):
        for attitude, count in sample.items():
            plot_data.append({
                "Sample Index": i,
                "Attitude": attitude,
                "Count": count,
                "Section": section,
                "Data Type": data_type
            })
    return pd.DataFrame(plot_data)

# Plot Seaborn combo plot (Box + Strip)
def plot_combo_attitude_chart(section, data_type, sample_list):
    df = prepare_plot_data(section, data_type, sample_list)

    plt.figure(figsize=(16, 6))
    
    # Create boxplot
    sns.boxplot(x="Attitude", y="Count", data=df, palette="Set2", width=0.5)
    
    # Overlay stripplot (with jitter)
    sns.stripplot(x="Attitude", y="Count", data=df, jitter=True, color="black", alpha=0.5)
    
    plt.title(f"{data_type.replace('_', ' ').title()} - {section.title()} (Combo Plot of Attitudes)")
    plt.xlabel("Attitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"polar_brexit_{section}_{data_type}_combo_plot.png")
    plt.close()

# Generate all 4 charts
for section in ["entity", "topical"]:
    for data_type in ["true_data", "pred_data"]:
        plot_combo_attitude_chart(section, data_type, per_sample_counts[section][data_type])
