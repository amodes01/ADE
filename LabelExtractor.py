import os
import json
import numpy as np
from collections import Counter

def analyze_attitudes(directory):
    """Scans a directory and accompanying subdirectories to analyze attitudes."""

    # Counter initialization
    overall_counts = Counter()
    entity_counts = Counter()
    topical_counts = Counter()

    # list initialization
    file_attitudes = []
    entity_attitudes = []
    topical_attitudes = []

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):  # Get the json responses
                file_path = os.path.join(subdir, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)  # Load json to search for "attitude" field
                        attitudes = []
                        e_attitudes = []
                        t_attitudes = []

                        # Extract attitudes from topical_attitudes
                        if "topical_attitudes" in data:
                            for entry in data["topical_attitudes"]:
                                if "attitude" in entry:
                                    attitudes.append(entry["attitude"])
                                    t_attitudes.append(entry["attitude"])

                        # Extract attitudes from entity_attitudes
                        if "entity_attitudes" in data:
                            for entry in data["entity_attitudes"]:
                                if "attitude" in entry:
                                    attitudes.append(entry["attitude"])
                                    e_attitudes.append(entry["attitude"])

                        # Collect data per file
                        file_attitudes.append(attitudes)
                        entity_attitudes.append(e_attitudes)
                        topical_attitudes.append(t_attitudes)

                        # Update counters
                        overall_counts.update(attitudes)
                        entity_counts.update(e_attitudes)
                        topical_counts.update(t_attitudes)

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # Skip files with errors (missing , etc.)

    # Compute statistics
    total_files = len(file_attitudes)
    attitude_types = ["Positive", "Neutral", "Negative"]

    def compute_avg_std(total_counts, total_files, attitude_types):
        """Computes average and standard deviation for each attitude category."""
        if total_files == 0:
            return {att: 0 for att in attitude_types}, {att: 0 for att in attitude_types}

        # Calculate averages by dividing total counts by the number of files
        avg = {att: total_counts[att] / total_files for att in attitude_types}

        # Calculate standard deviations based on frequency of attitude per file
        std_dev = {}
        for att in attitude_types:
            counts_per_file = []
            for attitudes in file_attitudes:  # this is the list of attitudes per file
                counts_per_file.append(1 if att in attitudes else 0)
            std_dev[att] = round(np.std(counts_per_file, ddof=1), 2)  # rounding to 2 decimal places

        return avg, std_dev

    overall_avg, overall_std = compute_avg_std(overall_counts, total_files, attitude_types)
    entity_avg, entity_std = compute_avg_std(entity_counts, total_files, attitude_types)
    topical_avg, topical_std = compute_avg_std(topical_counts, total_files, attitude_types)

    # Print results with 2 decimal precision
    print(f"\nAnalysis for directory: {directory}")
    print(f"Total files analyzed: {total_files}")
    print(f"Overall attitude distribution: {dict(overall_counts)}")
    print(f"Average attitudes per file: {double_precision(overall_avg)}")
    print(f"Standard deviation: {double_precision(overall_std)}\n")

    print(f"Entity Attitudes Distribution: {dict(entity_counts)}")
    print(f"Average entity attitudes per file: {double_precision(entity_avg)}")
    print(f"Entity standard deviation: {double_precision(entity_std)}\n")

    print(f"Topical Attitudes Distribution: {dict(topical_counts)}")
    print(f"Average topical attitudes per file: {double_precision(topical_avg)}")
    print(f"Topical standard deviation: {double_precision(topical_std)}")
    print("-" * 50)

def double_precision(avg_std_dict):
    return {att: f"{value:.2f}" for att, value in avg_std_dict.items()}

if __name__ == "__main__":
    directories = [
        "./test-brexit/gpt_results",
        "./ready-articles-musk/results-v1",
        "./ready-articles-boxing/results-v1/authenticated",
        "./ready-articles-supper/results/authenticated",
        "./ready-articles-vaccine/results/authenticated"
    ]

    for directory in directories:
        analyze_attitudes(directory)
