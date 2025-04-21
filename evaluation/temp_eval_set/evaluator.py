import os
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import numpy as np


def load_json_files_from_folder(folder_path):
    """Load all JSON files from a given folder."""
    files = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                files[file_name] = json.load(f)
    return files

def calculate_metrics(true_data, pred_data):
    """Calculate evaluation metrics."""
    all_true_labels = []
    all_pred_labels = []
    
    for true_item, pred_item in zip(true_data, pred_data):
        # Check if 'topical_attitudes' is not empty before iterating
        if true_item.get("topical_attitudes") and pred_item.get("topical_attitudes"):
            for true_attitude, pred_attitude in zip(true_item["topical_attitudes"], pred_item["topical_attitudes"]):
                all_true_labels.append(true_attitude["attitude"])
                all_pred_labels.append(pred_attitude["attitude"])
        
        # Check if 'entity_attitudes' is not empty before iterating
        if true_item.get("entity_attitudes") and pred_item.get("entity_attitudes"):
            for true_entity_attitude, pred_entity_attitude in zip(true_item["entity_attitudes"], pred_item["entity_attitudes"]):
                all_true_labels.append(true_entity_attitude["attitude"])
                all_pred_labels.append(pred_entity_attitude["attitude"])

    # If there are no true or predicted labels, return nan (this avoids division by zero in metrics)
    if len(all_true_labels) == 0 or len(all_pred_labels) == 0:
        return np.nan, np.nan, np.nan, np.nan

    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    return precision, recall, f1, accuracy

def evaluate_predictions(true_folder, pred_folder):
    """Evaluate all JSON prediction files in the true_folder against the pred_folder."""
    # Load the true and predicted files
    true_files = load_json_files_from_folder(true_folder)
    pred_files = load_json_files_from_folder(pred_folder)

    # Ensure both folders contain the same files (for simplicity, assuming same file names)
    if set(true_files.keys()) != set(pred_files.keys()):
        raise ValueError("The true and prediction folders do not contain the same files.")
    
    results = defaultdict(list)

    # Evaluate each pair of true and predicted files
    for file_name in true_files:
        true_data = true_files[file_name]
        pred_data = pred_files[file_name]
        
        # Calculate evaluation metrics
        precision, recall, f1, accuracy = calculate_metrics([true_data], [pred_data])
        
        # Save the results for this file, checking if they are NaN
        if np.isnan(precision) or np.isnan(recall) or np.isnan(f1) or np.isnan(accuracy):
            print(f"Skipping {file_name} due to empty or missing labels.")
            continue
        
        results[file_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
        
        print(f"Evaluating {file_name}:")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1 Score: {f1:.4f}")
        print(f" Accuracy: {accuracy:.4f}\n")
    
    return results

# Example usage
true_folder = './predictions'
pred_folder = './responses'
evaluation_results = evaluate_predictions(true_folder, pred_folder)

# Optionally, you can save results to a file
with open("evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)
