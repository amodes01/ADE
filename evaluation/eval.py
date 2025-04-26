import json
import re
from Mistral_Parser import LLMJsonParser
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from collections import defaultdict
import pickle
from datasets import load_from_disk
import logging
from fuzzywuzzy import fuzz
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd       
import seaborn as sns     


def unpickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def calculate_rouge_score(true_justification, pred_justification):
    """
    Calculate the ROUGE score between two justifications.
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=[pred_justification], references=[true_justification])
    return results["rougeL"]  # Use ROUGE-L for sentence-level similarity


def convert_numpy_types(obj):
    """Convert numpy types and frozensets to native Python types for JSON serialization."""
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.item() if obj.size == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return [convert_numpy_types(x) for x in obj]
    return obj


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def entity_similarity(entity1: str, entity2: str) -> float:
    """
    Compute semantic similarity between two entities using Sentence Transformers.

    Args:
        entity1 (str): First entity.
        entity2 (str): Second entity.

    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    embeddings = model.encode([entity1, entity2], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score


def find_matching_entity(entity_refs, reference_set, threshold=0.7):
    """
    Check if any entity reference in a frozenset matches any reference in another frozenset using semantic similarity.
    """
    for ref1 in entity_refs:
        for ref2 in reference_set:
            if entity_similarity(ref1, ref2) >= threshold:
                if ref1 != ref2:
                    print(f"Matching Entity Pair: ({ref1}, {ref2})")
                return True
    return False

def find_matching_topic(topic_refs, reference_set, threshold=0.7):
    """
    Check if any topic reference in a frozenset matches any reference in another frozenset using semantic similarity.
    """
    for ref1 in topic_refs:
        for ref2 in reference_set:
            if entity_similarity(ref1, ref2) >= threshold:
                if ref1 != ref2:
                    print(f"Matching Topic Pair: ({ref1}, {ref2})")
                return True
    return False

def build_entity_mapping(attitudes):
    """
    Build a mapping for entity_attitudes where keys are sets of entity references.
    """
    mapping = {}
    for attitude in attitudes:
        if not isinstance(attitude, dict):
            continue

        if "entity1" not in attitude or "entity2" not in attitude or "attitude" not in attitude:
            continue

        if not isinstance(attitude["entity1"], dict) or not isinstance(attitude["entity2"], dict):
            continue

        entity1_refs = set()
        entity2_refs = set()

        if "references" in attitude["entity1"]:
            entity1_refs = set(attitude["entity1"]["references"])
        elif "entity" in attitude["entity1"]:
            entity1_refs = {attitude["entity1"]["entity"]}

        if "references" in attitude["entity2"]:
            entity2_refs = set(attitude["entity2"]["references"])
        elif "entity" in attitude["entity2"]:
            entity2_refs = {attitude["entity2"]["entity"]}

        if not entity1_refs or not entity2_refs:
            continue

        justifications = attitude.get("justifications", [])

        key = (frozenset(entity1_refs), frozenset(entity2_refs))

        if key not in mapping:
            mapping[key] = []

        mapping[key].append({
            "attitude": attitude["attitude"],
            "justifications": justifications
        })

    return mapping

def build_topic_mapping(attitudes):
    """
    Build a mapping for topical_attitudes where keys are sets of topic references.
    """
    mapping = {}
    for attitude in attitudes:
        if not isinstance(attitude, dict):
            continue

        if "source" not in attitude or "target" not in attitude or "attitude" not in attitude:
            continue

        if not isinstance(attitude["source"], dict) or not isinstance(attitude["target"], dict):
            continue

        source_refs = set()
        target_refs = set()

        if "references" in attitude["source"]:
            source_refs = set(attitude["source"]["references"])
        elif "entity" in attitude["source"]:
            source_refs = {attitude["source"]["entity"]}
        elif "topic" in attitude["source"]:
            source_refs = {attitude["source"]["topic"]}
        else:
            continue

        if "references" in attitude["target"]:
            target_refs = set(attitude["target"]["references"])
        elif "entity" in attitude["target"]:
            target_refs = {attitude["target"]["entity"]}
        elif "topic" in attitude["target"]:
            target_refs = {attitude["target"]["topic"]}
        else:
            continue

        justifications = attitude.get("justifications", [])

        key = (frozenset(source_refs), frozenset(target_refs))

        if key not in mapping:
            mapping[key] = []

        mapping[key].append({
            "attitude": attitude["attitude"],
            "justifications": justifications
        })

    return mapping

def extract_entity_attitudes(data):
    """Extract the 'entity_attitudes' list from the JSON response."""
    if isinstance(data, dict) and "entity_attitudes" in data:
        return data["entity_attitudes"]
    return []

def extract_topical_attitudes(data):
    """Extract the 'topical_attitudes' list from the JSON response."""
    if isinstance(data, dict) and "topical_attitudes" in data:
        return data["topical_attitudes"]
    return []

def extract_entity_attitudes_polar(data):
    """Extract the 'entity_attitudes' list from the JSON response."""
    if isinstance(data, list):
        # Search through list items for dictionaries containing entity_attitudes
        for item in data:
            if isinstance(item, dict) and "entity_attitudes" in item:
                return item["entity_attitudes"]
    return []


def extract_topical_attitudes_polar(data):
    """Extract the 'topical_attitudes' list from the JSON response (list version)."""
    if isinstance(data, list):
        # Search through list items for dictionaries containing topical_attitudes
        for item in data:
            if isinstance(item, dict) and "topical_attitudes" in item:
                return item["topical_attitudes"]
    return []

def build_entity_mapping_polar(attitudes):
    """
    Build a mapping for entity_attitudes where keys are sets of entity references.
    """
    mapping = {}
    for attitude in attitudes:
        if not isinstance(attitude, dict):
            continue

        if "entity1" not in attitude or "entity2" not in attitude or "attitude" not in attitude:
            continue

        if not isinstance(attitude["entity1"], dict) or not isinstance(attitude["entity2"], dict):
            continue

        entity1_refs = set()
        entity2_refs = set()

        
        if "entity" in attitude["entity1"]:
            entity1_refs = {attitude["entity1"]["entity"]}

        
        if "entity" in attitude["entity2"]:
            entity2_refs = {attitude["entity2"]["entity"]}

        if not entity1_refs or not entity2_refs:
            continue

        justifications = attitude.get("justifications", [])

        key = (frozenset(entity1_refs), frozenset(entity2_refs))

        if key not in mapping:
            mapping[key] = []

        mapping[key].append({
            "attitude": attitude["attitude"],
            "justifications": justifications
        })

    return mapping


def build_topic_mapping_polar(attitudes):
    """
    Build a mapping for topical_attitudes where keys are sets of topic references.
    """
    mapping = {}
    for attitude in attitudes:
        if not isinstance(attitude, dict):
            continue

        if "source" not in attitude or "target" not in attitude or "attitude" not in attitude:
            continue

        if not isinstance(attitude["source"], dict) or not isinstance(attitude["target"], dict):
            continue

        source_refs = set()
        target_refs = set()

        
        if "entity" in attitude["source"]:
            source_refs = {attitude["source"]["entity"]}
        elif "topic" in attitude["source"]:
            source_refs = {attitude["source"]["topic"]}
        else:
            continue

        
        if "entity" in attitude["target"]:
            target_refs = {attitude["target"]["entity"]}
        elif "topic" in attitude["target"]:
            target_refs = {attitude["target"]["topic"]}
        else:
            continue

        justifications = attitude.get("justifications", [])

        key = (frozenset(source_refs), frozenset(target_refs))

        if key not in mapping:
            mapping[key] = []

        mapping[key].append({
            "attitude": attitude["attitude"],
            "justifications": justifications
        })

    return mapping



def normalize_label(label):
    """Normalize a label to a standard format (lowercase, trimmed)."""
    return label.strip().lower()

def jaccard_similarity(set1, set2):
    """Calculate the Jaccard Similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_pair_similarity(true_refs1, true_refs2, pred_refs1, pred_refs2):
    """Calculate the similarity between two pairs of references using Jaccard similarity."""
    similarity1 = jaccard_similarity(true_refs1, pred_refs1)
    similarity2 = jaccard_similarity(true_refs2, pred_refs2)
    return (similarity1 + similarity2) / 2

def process_counts(true_mapping, pred_mapping, metrics_category, is_topic=False):
    """Process and count matches between true and predicted pairs."""
    if "matched_pairs" not in metrics_category:
        metrics_category["matched_pairs"] = []
    if "unmatched_true_pairs" not in metrics_category:
        metrics_category["unmatched_true_pairs"] = []
    if "unmatched_pred_pairs" not in metrics_category:
        metrics_category["unmatched_pred_pairs"] = []


    matched_true_pairs = set()
    matched_pred_pairs = set()

    # Compare all true pairs against predictions
    for (true_refs1, true_refs2), true_attitude_list in true_mapping.items():
        true_pair = (true_refs1, true_refs2)
        pair_matched = False

        for (pred_refs1, pred_refs2), pred_attitude_list in pred_mapping.items():
            pred_pair = (pred_refs1, pred_refs2)

            if is_topic:
                is_match = (find_matching_topic(true_refs1, pred_refs1) and 
                          find_matching_topic(true_refs2, pred_refs2))
            else:
                is_match = (find_matching_entity(true_refs1, pred_refs1) and 
                          find_matching_entity(true_refs2, pred_refs2))

            if is_match:
                pair_matched = True
                metrics_category["pair_counts"]["tp"] += 1
                matched_true_pairs.add(true_pair)
                matched_pred_pairs.add(pred_pair)

                metrics_category["matched_pairs"].append({
                    "true_pair": list(true_pair),
                    "pred_pair": list(pred_pair)
                })

                print(f"Matched Pair: {true_pair} with {pred_pair}")

                # Compare attitudes within matched pairs
                for true_info in true_attitude_list:
                    for pred_info in pred_attitude_list:
                        if normalize_label(true_info["attitude"]) == normalize_label(pred_info["attitude"]):
                            metrics_category["attitude_counts"]["tp"] += 1
                        else:
                            metrics_category["attitude_counts"]["fp"] += 1
                            metrics_category["attitude_counts"]["fn"] += 1

                        # Compare justifications
                        if true_info.get("justifications") and pred_info.get("justifications"):
                            metrics_category["justification_counts"]["total"] += len(true_info["justifications"])
                            for true_just in true_info["justifications"]:
                                for pred_just in pred_info["justifications"]:
                                    if calculate_rouge_score(true_just, pred_just) >= 0.5:
                                        metrics_category["justification_counts"]["matched"] += 1
                                        break

        if not pair_matched:
            metrics_category["pair_counts"]["fn"] += 1
            metrics_category["unmatched_true_pairs"].append({
                "true_pair": [list(true_refs1), list(true_refs2)]
            })

    # Count unmatched predicted pairs as false positives
    for (pred_refs1, pred_refs2), pred_attitude_list in pred_mapping.items():
        pred_pair = (pred_refs1, pred_refs2)
        if pred_pair not in matched_pred_pairs:
            metrics_category["pair_counts"]["fp"] += 1
            for pred_info in pred_attitude_list:
                metrics_category["attitude_counts"]["fp"] += 1

    for (pred_refs1, pred_refs2), pred_attitude_list in pred_mapping.items():
        pred_pair = (pred_refs1, pred_refs2)
        if pred_pair not in matched_pred_pairs:
            metrics_category["pair_counts"]["fp"] += 1
            metrics_category["unmatched_pred_pairs"].append({
                "pred_pair": [list(pred_refs1), list(pred_refs2)]
            })
            for pred_info in pred_attitude_list:
                metrics_category["attitude_counts"]["fp"] += 1


def calculate_metrics(true_data, pred_data):
    """Calculate evaluation metrics and return raw counts."""
    metrics = {
        "entity": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "topical": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        }
    }

    # Extract attitudes from data
    true_entity_attitudes = extract_entity_attitudes(true_data)
    pred_entity_attitudes = extract_entity_attitudes(pred_data)
    true_topic_attitudes = extract_topical_attitudes(true_data)
    pred_topic_attitudes = extract_topical_attitudes(pred_data)

    # Process entity attitudes if they exist
    if true_entity_attitudes or pred_entity_attitudes:
        true_entity_mapping = build_entity_mapping(true_entity_attitudes)
        pred_entity_mapping = build_entity_mapping(pred_entity_attitudes)
        
        metrics["entity"]["sample_counts"]["true_pairs"] = len(true_entity_mapping)
        metrics["entity"]["sample_counts"]["pred_pairs"] = len(pred_entity_mapping)
        metrics["entity"]["sample_counts"]["true_attitudes"] = sum(len(v) for v in true_entity_mapping.values())
        metrics["entity"]["sample_counts"]["pred_attitudes"] = sum(len(v) for v in pred_entity_mapping.values())
        
        if true_entity_mapping or pred_entity_mapping:
            process_counts(true_entity_mapping, pred_entity_mapping, metrics["entity"], is_topic=False)

    # Process topical attitudes if they exist
    if true_topic_attitudes or pred_topic_attitudes:
        true_topic_mapping = build_topic_mapping(true_topic_attitudes)
        pred_topic_mapping = build_topic_mapping(pred_topic_attitudes)
        
        metrics["topical"]["sample_counts"]["true_pairs"] = len(true_topic_mapping)
        metrics["topical"]["sample_counts"]["pred_pairs"] = len(pred_topic_mapping)
        metrics["topical"]["sample_counts"]["true_attitudes"] = sum(len(v) for v in true_topic_mapping.values())
        metrics["topical"]["sample_counts"]["pred_attitudes"] = sum(len(v) for v in pred_topic_mapping.values())
        
        if true_topic_mapping or pred_topic_mapping:
            process_counts(true_topic_mapping, pred_topic_mapping, metrics["topical"], is_topic=True)

    return metrics




def visualize_metrics(all_metrics, polar=False):
    """Visualizes Precision, Recall, F1, and Justification Overlap for entity and topical relationships using boxplots."""
    
    categories = ['entity', 'topical']
    metrics = ['precision', 'recall', 'f1']
    
    # Convert the dataset into long format for plotting
    data = {
        'relationship': [],
        'metric': [],
        'score': []
    }
    
    for cat in categories:
        for s in all_metrics['samples'].values():
            # Extract Pair Matching (Precision, Recall, F1)
            if 'pair_matching' in s['metrics'][cat]:  
                for metric in metrics:
                    data['relationship'].append(cat.title())  # 'Entity' or 'Topical'
                    data['metric'].append(metric.capitalize())  # 'Precision', 'Recall', 'F1'
                    data['score'].append(s['metrics'][cat]['pair_matching'][metric])
                    #print(counter,cat, metric, s['metrics'][cat]['pair_matching'][metric])
            
            # Extract Justification Overlap
            if 'justification' in s['metrics'][cat]:  
                data['relationship'].append(cat.title())
                data['metric'].append('Overlap')  # Custom label for Justification Overlap
                data['score'].append(s['metrics'][cat]['justification']['overlap'])
            

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Debugging: Print summary statistics
    print(df.describe())

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='score', hue='relationship', data=df, showfliers=False)  # Hide extreme outliers
    sns.stripplot(x='metric', y='score', hue='relationship', data=df, dodge=True, jitter=True, alpha=0.4)  # Individual points
    
    plt.title('Metric Distributions: Precision, Recall, F1 & Justification Overlap')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Relationship Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if polar:
        plt.savefig('polar_metric_distributions.png', dpi=300)
    else:
        plt.savefig('metric_distributions.png', dpi=300)
    plt.show()  # Display in real-time instead of just saving


def calculate_final_metrics(metrics_data):
    """Calculate final metrics from counts data."""
    final_metrics = {}
    
    for category in ["entity", "topical"]:
        if category not in metrics_data:
            continue
            
        final_metrics[category] = {}
        
        # Pair matching metrics
        tp = metrics_data[category]["pair_counts"]["tp"]
        fp = metrics_data[category]["pair_counts"]["fp"]
        fn = metrics_data[category]["pair_counts"]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        final_metrics[category]["pair_matching"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        # Attitude prediction metrics
        tp_att = metrics_data[category]["attitude_counts"]["tp"]
        fp_att = metrics_data[category]["attitude_counts"]["fp"]
        fn_att = metrics_data[category]["attitude_counts"]["fn"]
        
        att_precision = tp_att / (tp_att + fp_att) if (tp_att + fp_att) > 0 else 0
        att_recall = tp_att / (tp_att + fn_att) if (tp_att + fn_att) > 0 else 0
        att_f1 = 2 * (att_precision * att_recall) / (att_precision + att_recall) if (att_precision + att_recall) > 0 else 0
        att_accuracy = tp_att / (tp_att + fp_att + fn_att) if (tp_att + fp_att + fn_att) > 0 else 0
        
        final_metrics[category]["attitude_prediction"] = {
            "precision": att_precision,
            "recall": att_recall,
            "f1": att_f1,
            "accuracy": att_accuracy,
            "true_positives": tp_att,
            "false_positives": fp_att,
            "false_negatives": fn_att
        }
        
        # Justification metrics
        matched = metrics_data[category]["justification_counts"]["matched"]
        total = metrics_data[category]["justification_counts"]["total"]
        overlap = matched / total if total > 0 else 0
        
        final_metrics[category]["justification"] = {
            "overlap": overlap,
            "matched_justifications": matched,
            "total_justifications": total
        }
        
        # Counts - use sample_counts if available, otherwise use total_counts
        counts_source = metrics_data[category].get("sample_counts", metrics_data[category].get("total_counts", {}))
        final_metrics[category]["counts"] = counts_source
    
    return final_metrics



def calculate_metrics_polar(true_data, pred_data):
    """Calculate evaluation metrics and return raw counts."""
    metrics = {
        "entity": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "topical": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        }
    }

    # Extract attitudes from data
    true_entity_attitudes = extract_entity_attitudes(true_data)
    pred_entity_attitudes = extract_entity_attitudes_polar(pred_data)
    true_topic_attitudes = extract_topical_attitudes(true_data)
    pred_topic_attitudes = extract_topical_attitudes_polar(pred_data)


    # Process entity attitudes if they exist
    if true_entity_attitudes:
        true_entity_mapping = build_entity_mapping(true_entity_attitudes)
        pred_entity_mapping = build_entity_mapping_polar(pred_entity_attitudes)

        
        metrics["entity"]["sample_counts"]["true_pairs"] = len(true_entity_mapping)
        metrics["entity"]["sample_counts"]["pred_pairs"] = len(pred_entity_mapping)
        metrics["entity"]["sample_counts"]["true_attitudes"] = sum(len(v) for v in true_entity_mapping.values())
        metrics["entity"]["sample_counts"]["pred_attitudes"] = sum(len(v) for v in pred_entity_mapping.values())
        
        if true_entity_mapping:
            process_counts(true_entity_mapping, pred_entity_mapping, metrics["entity"], is_topic=False)

    # Process topical attitudes if they exist
    if true_topic_attitudes:
        true_topic_mapping = build_topic_mapping(true_topic_attitudes)
        pred_topic_mapping = build_topic_mapping_polar(pred_topic_attitudes)

        
        metrics["topical"]["sample_counts"]["true_pairs"] = len(true_topic_mapping)
        metrics["topical"]["sample_counts"]["pred_pairs"] = len(pred_topic_mapping)
        metrics["topical"]["sample_counts"]["true_attitudes"] = sum(len(v) for v in true_topic_mapping.values())
        metrics["topical"]["sample_counts"]["pred_attitudes"] = sum(len(v) for v in pred_topic_mapping.values())
        
        if true_topic_mapping:
            process_counts(true_topic_mapping, pred_topic_mapping, metrics["topical"], is_topic=True)

    return metrics


def evaluate_predictions(true_dataset_path, pred_files_path):
    """Evaluate predictions against ground truth."""
    true_dataset = load_from_disk(true_dataset_path)
    pred_files = unpickle(pred_files_path)
    num_samples = min(len(true_dataset), len(pred_files))

    # Initialize results storage
    all_metrics = {
        "entity": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0},
            "total_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "topical": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0},
            "total_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "samples": {}
    }


    for i in tqdm(range(num_samples), desc="Evaluating samples", unit="sample"):
        

        # Extract JSON for ground truth
        best_json, _ = LLMJsonParser.parse_json(true_dataset[i]["output"])
        true_data = best_json if best_json is not None else {}

        # Extract JSON for predictions
        best_json, _ = LLMJsonParser.parse_json(pred_files[i]["response"])
        pred_data = best_json if best_json is not None else {}

        if not true_data or not pred_data:
            continue

        # Calculate metrics
        metrics = calculate_metrics(true_data, pred_data)

        # Store sample results with calculated metrics
        sample_metrics = calculate_final_metrics({
            "entity": metrics["entity"],
            "topical": metrics["topical"]
        })
        all_metrics["samples"][f"sample_{i}"] = {
            "metrics": sample_metrics,
            "true_data": true_data,
            "pred_data": pred_data
        }

        # Aggregate counts
        for category in ["entity", "topical"]:
            # Update performance counts
            for count_type in ["pair_counts", "attitude_counts", "justification_counts"]:
                for metric, value in metrics[category][count_type].items():
                    all_metrics[category][count_type][metric] += value
            
            # Update sample counts (for this sample)
            for count in metrics[category]["sample_counts"]:
                all_metrics[category]["sample_counts"][count] = metrics[category]["sample_counts"][count]
            
            # Update running totals
            for count in metrics[category]["sample_counts"]:
                all_metrics[category]["total_counts"][count] += metrics[category]["sample_counts"][count]

    # Calculate final metrics from aggregated counts
    final_metrics = calculate_final_metrics({
        "entity": {
            "pair_counts": all_metrics["entity"]["pair_counts"],
            "attitude_counts": all_metrics["entity"]["attitude_counts"],
            "justification_counts": all_metrics["entity"]["justification_counts"],
            "sample_counts": all_metrics["entity"]["total_counts"]  # Use accumulated totals
        },
        "topical": {
            "pair_counts": all_metrics["topical"]["pair_counts"],
            "attitude_counts": all_metrics["topical"]["attitude_counts"],
            "justification_counts": all_metrics["topical"]["justification_counts"],
            "sample_counts": all_metrics["topical"]["total_counts"]  # Use accumulated totals
        }
    })
    
    # Merge final metrics into all_metrics
    for category in ["entity", "topical"]:
        all_metrics[category].update(final_metrics.get(category, {}))

    # Print summary
    print("\nEvaluation Summary:")
    print("="*50)
    
    # Entity metrics
    if "entity" in all_metrics:
        print("\nEntity Pair Matching Metrics:")
        print(f"- Precision: {all_metrics['entity']['pair_matching']['precision']:.4f}")
        print(f"- Recall:    {all_metrics['entity']['pair_matching']['recall']:.4f}")
        print(f"- F1:        {all_metrics['entity']['pair_matching']['f1']:.4f}")
        print(f"- True Positives: {all_metrics['entity']['pair_matching']['true_positives']}")
        print(f"- False Positives: {all_metrics['entity']['pair_matching']['false_positives']}")
        print(f"- False Negatives: {all_metrics['entity']['pair_matching']['false_negatives']}")
        
        print("\nEntity Attitude Prediction Metrics (for matched pairs):")
        print(f"- Precision: {all_metrics['entity']['attitude_prediction']['precision']:.4f}")
        print(f"- Recall:    {all_metrics['entity']['attitude_prediction']['recall']:.4f}")
        print(f"- F1:        {all_metrics['entity']['attitude_prediction']['f1']:.4f}")
        print(f"- Accuracy:  {all_metrics['entity']['attitude_prediction']['accuracy']:.4f}")
        print(f"- True Positives: {all_metrics['entity']['attitude_prediction']['true_positives']}")
        print(f"- False Positives: {all_metrics['entity']['attitude_prediction']['false_positives']}")
        print(f"- False Negatives: {all_metrics['entity']['attitude_prediction']['false_negatives']}")
        
        print("\nEntity Justification Metrics:")
        print(f"- Average Overlap: {all_metrics['entity']['justification']['overlap']:.4f}")
        print(f"- Matched Justifications: {all_metrics['entity']['justification']['matched_justifications']} of {all_metrics['entity']['justification']['total_justifications']}")
        
        print("\nEntity Counts:")
        print(f"- Total True Pairs: {all_metrics['entity']['total_counts']['true_pairs']}")
        print(f"- Total Pred Pairs: {all_metrics['entity']['total_counts']['pred_pairs']}")
        print(f"- Total True Attitudes: {all_metrics['entity']['total_counts']['true_attitudes']}")
        print(f"- Total Pred Attitudes: {all_metrics['entity']['total_counts']['pred_attitudes']}")
    
    # Topical metrics
    if "topical" in all_metrics:
        print("\nTopical Pair Matching Metrics:")
        print(f"- Precision: {all_metrics['topical']['pair_matching']['precision']:.4f}")
        print(f"- Recall:    {all_metrics['topical']['pair_matching']['recall']:.4f}")
        print(f"- F1:        {all_metrics['topical']['pair_matching']['f1']:.4f}")
        print(f"- True Positives: {all_metrics['topical']['pair_matching']['true_positives']}")
        print(f"- False Positives: {all_metrics['topical']['pair_matching']['false_positives']}")
        print(f"- False Negatives: {all_metrics['topical']['pair_matching']['false_negatives']}")
        
        print("\nTopical Attitude Prediction Metrics (for matched pairs):")
        print(f"- Precision: {all_metrics['topical']['attitude_prediction']['precision']:.4f}")
        print(f"- Recall:    {all_metrics['topical']['attitude_prediction']['recall']:.4f}")
        print(f"- F1:        {all_metrics['topical']['attitude_prediction']['f1']:.4f}")
        print(f"- Accuracy:  {all_metrics['topical']['attitude_prediction']['accuracy']:.4f}")
        print(f"- True Positives: {all_metrics['topical']['attitude_prediction']['true_positives']}")
        print(f"- False Positives: {all_metrics['topical']['attitude_prediction']['false_positives']}")
        print(f"- False Negatives: {all_metrics['topical']['attitude_prediction']['false_negatives']}")
        
        print("\nTopical Justification Metrics:")
        print(f"- Average Overlap: {all_metrics['topical']['justification']['overlap']:.4f}")
        print(f"- Matched Justifications: {all_metrics['topical']['justification']['matched_justifications']} of {all_metrics['topical']['justification']['total_justifications']}")
        
        print("\nTopical Counts:")
        print(f"- Total True Pairs: {all_metrics['topical']['total_counts']['true_pairs']}")
        print(f"- Total Pred Pairs: {all_metrics['topical']['total_counts']['pred_pairs']}")
        print(f"- Total True Attitudes: {all_metrics['topical']['total_counts']['true_attitudes']}")
        print(f"- Total Pred Attitudes: {all_metrics['topical']['total_counts']['pred_attitudes']}")
    
    # Combined counts
    print("\nCombined Counts:")
    total_true_pairs = all_metrics['entity']['total_counts']['true_pairs'] + all_metrics['topical']['total_counts']['true_pairs']
    total_pred_pairs = all_metrics['entity']['total_counts']['pred_pairs'] + all_metrics['topical']['total_counts']['pred_pairs']
    total_true_attitudes = all_metrics['entity']['total_counts']['true_attitudes'] + all_metrics['topical']['total_counts']['true_attitudes']
    total_pred_attitudes = all_metrics['entity']['total_counts']['pred_attitudes'] + all_metrics['topical']['total_counts']['pred_attitudes']
    
    print(f"- Total True Pairs: {total_true_pairs}")
    print(f"- Total Pred Pairs: {total_pred_pairs}")
    print(f"- Total True Attitudes: {total_true_attitudes}")
    print(f"- Total Pred Attitudes: {total_pred_attitudes}")
    print("="*50)

    # Visualize the metrics
    visualize_metrics(all_metrics)
    
    return all_metrics


def count_attitudes(data, is_polar=False):
    attitude_counts = {
        "entity": {"Positive": 0, "Neutral": 0, "Negative": 0},
        "topical": {"Positive": 0, "Neutral": 0, "Negative": 0}
    }

    # Extract attitudes
    if not is_polar:
        entity_attitudes = extract_entity_attitudes(data)
        topical_attitudes = extract_topical_attitudes(data)
    else:
        entity_attitudes = extract_entity_attitudes_polar(data)
        topical_attitudes = extract_topical_attitudes_polar(data)

    for item in entity_attitudes:
        att = item.get("attitude", None)
        if att in attitude_counts["entity"]:
            attitude_counts["entity"][att] += 1

    for item in topical_attitudes:
        att = item.get("attitude", None)
        if att in attitude_counts["topical"]:
            attitude_counts["topical"][att] += 1

    return attitude_counts


import os

def evaluate_predictions_polar(true_dataset_path, polar_dir_path):
    """Evaluate POLAR predictions against ground truth."""
    true_dataset = load_from_disk(true_dataset_path)
    
    # Load POLAR JSON files in order (article0.json to article404.json)
    polar_files = []
    for i in range(405):  # 0-404 inclusive
        file_path = os.path.join(polar_dir_path, f"article{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    polar_data = json.load(f)
                    polar_files.append({"response": json.dumps(polar_data)})  # Wrap to match expected format
                except json.JSONDecodeError:
                    polar_files.append({"response": "{}"})  # Empty if invalid
        else:
            polar_files.append({"response": "{}"})  # Empty if missing

    num_samples = min(len(true_dataset), len(polar_files))

    # Initialize results storage
    all_metrics = {
        "entity": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0},
            "total_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "topical": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0},
            "total_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "samples": {}
    }

    overall_attitude_summary = {
        "true_data": {
            "entity": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "topical": {"Positive": 0, "Neutral": 0, "Negative": 0}
        },
        "pred_data": {
            "entity": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "topical": {"Positive": 0, "Neutral": 0, "Negative": 0}
        }
    }

    for i in tqdm(range(num_samples), desc="Evaluating samples", unit="sample"):
        # Extract JSON for ground truth
        best_json, _ = LLMJsonParser.parse_json(true_dataset[i]["output"])
        true_data = best_json if best_json is not None else {}

        # Get corresponding POLAR prediction
        polar_json, _ = LLMJsonParser.parse_json(polar_files[i]["response"])
        pred_data = polar_json if polar_json is not None else {}

        if not true_data or not pred_data:
            continue

        # Extract all unique entities and topics from true and pred data
        def extract_unique_items(data,Polar=False):
            entities = set()
            topics = set()
            
            # Extract from entity_attitudes
            if not Polar:
                entity_attitudes = extract_entity_attitudes(data)
            else:
                entity_attitudes = extract_entity_attitudes_polar(data)
            for attitude in entity_attitudes:
                if isinstance(attitude, dict):
                    for entity_key in ["entity1", "entity2"]:
                        if entity_key in attitude and isinstance(attitude[entity_key], dict):
                            if "entity" in attitude[entity_key]:
                                entities.add(attitude[entity_key]["entity"])
                            if "references" in attitude[entity_key]:
                                entities.update(attitude[entity_key]["references"])
            
            # Extract from topical_attitudes
            if not Polar:
                topical_attitudes = extract_topical_attitudes(data)
            else:
                topical_attitudes = extract_topical_attitudes_polar(data)
            for attitude in topical_attitudes:
                if isinstance(attitude, dict):
                    for topic_key in ["source", "target"]:
                        if topic_key in attitude and isinstance(attitude[topic_key], dict):
                            if "topic" in attitude[topic_key]:
                                topics.add(attitude[topic_key]["topic"])
                            if "references" in attitude[topic_key]:
                                topics.update(attitude[topic_key]["references"])
            
            return {
                "entities": list(entities),
                "topics": list(topics)
            }

        true_items = extract_unique_items(true_data)
        pred_items = extract_unique_items(pred_data, True)



        # Calculate metrics
        metrics = calculate_metrics_polar(true_data, pred_data)

        # Calculate matching metrics for entities and topics
        def calculate_matching_metrics(true_items, pred_items, threshold=0.7):
            # Entity matching
            true_entities = set(true_items["entities"])
            pred_entities = set(pred_items["entities"])
            
            entity_matches = []
            entity_tp = 0
            entity_fp = 0
            entity_fn = 0
            
            matched_pred_entities = set()
            
            for true_ent in true_entities:
                matched = False
                for pred_ent in pred_entities:
                    if entity_similarity(true_ent, pred_ent) >= threshold:
                        entity_matches.append({
                            "true_entity": true_ent,
                            "pred_entity": pred_ent,
                            "similarity": entity_similarity(true_ent, pred_ent)
                        })
                        matched_pred_entities.add(pred_ent)
                        matched = True
                        entity_tp += 1
                        break
                if not matched:
                    entity_fn += 1
            
            entity_fp = len(pred_entities - matched_pred_entities)
            
            # Topic matching
            true_topics = set(true_items["topics"])
            pred_topics = set(pred_items["topics"])
            
            topic_matches = []
            topic_tp = 0
            topic_fp = 0
            topic_fn = 0
            
            matched_pred_topics = set()
            
            for true_topic in true_topics:
                matched = False
                for pred_topic in pred_topics:
                    if entity_similarity(true_topic, pred_topic) >= threshold:
                        topic_matches.append({
                            "true_topic": true_topic,
                            "pred_topic": pred_topic,
                            "similarity": entity_similarity(true_topic, pred_topic)
                        })
                        matched_pred_topics.add(pred_topic)
                        matched = True
                        topic_tp += 1
                        break
                if not matched:
                    topic_fn += 1
            
            topic_fp = len(pred_topics - matched_pred_topics)
            
            # Calculate precision, recall, f1
            def calculate_prf1(tp, fp, fn):
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                return {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                }
            
            return {
                "entity_matching": {
                    "matches": entity_matches,
                    "metrics": calculate_prf1(entity_tp, entity_fp, entity_fn),
                    "true_count": len(true_entities),
                    "pred_count": len(pred_entities)
                },
                "topic_matching": {
                    "matches": topic_matches,
                    "metrics": calculate_prf1(topic_tp, topic_fp, topic_fn),
                    "true_count": len(true_topics),
                    "pred_count": len(pred_topics)
                }
            }

        matching_metrics = calculate_matching_metrics(true_items, pred_items)

        # Store sample results with calculated metrics
        sample_metrics = calculate_final_metrics({
            "entity": metrics["entity"],
            "topical": metrics["topical"]
        })

        true_attitude_summary = count_attitudes(true_data, is_polar=False)
        pred_attitude_summary = count_attitudes(pred_data, is_polar=True)

        # Update global attitude counts
        for key in ["entity", "topical"]:
            for polarity in ["Positive", "Neutral", "Negative"]:
                overall_attitude_summary["true_data"][key][polarity] += true_attitude_summary[key][polarity]
                overall_attitude_summary["pred_data"][key][polarity] += pred_attitude_summary[key][polarity]

        
        all_metrics["samples"][f"sample_{i}"] = {
            "metrics": sample_metrics,
            "true_data": {
                "entities": true_items["entities"],
                "topics": true_items["topics"],
                "attitude_summary": true_attitude_summary
            },
            "pred_data": {
                "entities": pred_items["entities"],
                "topics": pred_items["topics"],
                "attitude_summary": pred_attitude_summary
            },
            "matching_metrics": matching_metrics,
            "matched_pairs": {
                "entity": metrics["entity"].get("matched_pairs", []),
                "topical": metrics["topical"].get("matched_pairs", [])
            },
            "unmatched_pairs": {
                "entity": {
                    "true_unmatched": metrics["entity"].get("unmatched_true_pairs", []),
                    "pred_unmatched": metrics["entity"].get("unmatched_pred_pairs", [])
                },
                "topical": {
                    "true_unmatched": metrics["topical"].get("unmatched_true_pairs", []),
                    "pred_unmatched": metrics["topical"].get("unmatched_pred_pairs", [])
                }
            }
        }

        # Aggregate counts
        for category in ["entity", "topical"]:
            for count_type in ["pair_counts", "attitude_counts", "justification_counts"]:
                for metric, value in metrics[category][count_type].items():
                    all_metrics[category][count_type][metric] += value
            
            for count in metrics[category]["sample_counts"]:
                all_metrics[category]["sample_counts"][count] = metrics[category]["sample_counts"][count]
                all_metrics[category]["total_counts"][count] += metrics[category]["sample_counts"][count]

    # Calculate final metrics

    all_metrics["attitude_summary"] = overall_attitude_summary


    final_metrics = calculate_final_metrics({
        "entity": {
            "pair_counts": all_metrics["entity"]["pair_counts"],
            "attitude_counts": all_metrics["entity"]["attitude_counts"],
            "justification_counts": all_metrics["entity"]["justification_counts"],
            "sample_counts": all_metrics["entity"]["total_counts"]
        },
        "topical": {
            "pair_counts": all_metrics["topical"]["pair_counts"],
            "attitude_counts": all_metrics["topical"]["attitude_counts"],
            "justification_counts": all_metrics["topical"]["justification_counts"],
            "sample_counts": all_metrics["topical"]["total_counts"]
        }
    })
    
    for category in ["entity", "topical"]:
        all_metrics[category].update(final_metrics.get(category, {}))

    # Print summary and visualize
    print("\nEvaluation Summary:")
    print("="*50)
    
    # ... (rest of your existing summary printing code)

    visualize_metrics(all_metrics, True)
    
    return all_metrics



def evaluate_predictions_polar_recursive(true_dataset_root, polar_dir_root):
    """Evaluate POLAR predictions when both paths contain subdirectories of JSON files."""
    
    def collect_pickle_paths(base_path):
        pkl_paths = {}
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".PCKL"):
                    rel_path = os.path.relpath(os.path.join(root, file), base_path)
                    pkl_paths[rel_path] = os.path.join(root, file)
        return pkl_paths

    true_paths = collect_pickle_paths(true_dataset_root)
    polar_paths = collect_pickle_paths(polar_dir_root)

    # Find common files only
    common_files = set(true_paths.keys()) & set(polar_paths.keys())

    if not common_files:
        print("No matching files found between the true dataset and the POLAR predictions.")
        return

    # Initialize results storage
    all_metrics = {
        "entity": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0},
            "total_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "topical": {
            "pair_counts": {"tp": 0, "fp": 0, "fn": 0},
            "attitude_counts": {"tp": 0, "fp": 0, "fn": 0},
            "justification_counts": {"matched": 0, "total": 0},
            "sample_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0},
            "total_counts": {"true_pairs": 0, "pred_pairs": 0, "true_attitudes": 0, "pred_attitudes": 0}
        },
        "samples": {}
    }

    overall_attitude_summary = {
        "true_data": {"entity": {"Positive": 0, "Neutral": 0, "Negative": 0},
                      "topical": {"Positive": 0, "Neutral": 0, "Negative": 0}},
        "pred_data": {"entity": {"Positive": 0, "Neutral": 0, "Negative": 0},
                      "topical": {"Positive": 0, "Neutral": 0, "Negative": 0}}
    }

    for i, rel_path in enumerate(tqdm(sorted(common_files), desc="Evaluating samples", unit="file")):
        with open(true_paths[rel_path], 'r', encoding='utf-8') as f:
            try:
                true_json = json.load(f)
                true_data, _ = LLMJsonParser.parse_json(true_json["output"])
            except:
                continue

        with open(polar_paths[rel_path], 'r', encoding='utf-8') as f:
            try:
                polar_json = json.load(f)
                pred_data, _ = LLMJsonParser.parse_json(json.dumps(polar_json))
            except:
                continue

        if not true_data or not pred_data:
            continue

# Extract all unique entities and topics from true and pred data
        def extract_unique_items(data,Polar=False):
            entities = set()
            topics = set()
            
            # Extract from entity_attitudes
            if not Polar:
                entity_attitudes = extract_entity_attitudes(data)
            else:
                entity_attitudes = extract_entity_attitudes_polar(data)
            for attitude in entity_attitudes:
                if isinstance(attitude, dict):
                    for entity_key in ["entity1", "entity2"]:
                        if entity_key in attitude and isinstance(attitude[entity_key], dict):
                            if "entity" in attitude[entity_key]:
                                entities.add(attitude[entity_key]["entity"])
                            if "references" in attitude[entity_key]:
                                entities.update(attitude[entity_key]["references"])
            
            # Extract from topical_attitudes
            if not Polar:
                topical_attitudes = extract_topical_attitudes(data)
            else:
                topical_attitudes = extract_topical_attitudes_polar(data)
            for attitude in topical_attitudes:
                if isinstance(attitude, dict):
                    for topic_key in ["source", "target"]:
                        if topic_key in attitude and isinstance(attitude[topic_key], dict):
                            if "topic" in attitude[topic_key]:
                                topics.add(attitude[topic_key]["topic"])
                            if "references" in attitude[topic_key]:
                                topics.update(attitude[topic_key]["references"])
            
            return {
                "entities": list(entities),
                "topics": list(topics)
            }

        true_items = extract_unique_items(true_data)
        pred_items = extract_unique_items(pred_data, True)



        # Calculate metrics
        metrics = calculate_metrics_polar(true_data, pred_data)

        # Calculate matching metrics for entities and topics
        def calculate_matching_metrics(true_items, pred_items, threshold=0.7):
            # Entity matching
            true_entities = set(true_items["entities"])
            pred_entities = set(pred_items["entities"])
            
            entity_matches = []
            entity_tp = 0
            entity_fp = 0
            entity_fn = 0
            
            matched_pred_entities = set()
            
            for true_ent in true_entities:
                matched = False
                for pred_ent in pred_entities:
                    if entity_similarity(true_ent, pred_ent) >= threshold:
                        entity_matches.append({
                            "true_entity": true_ent,
                            "pred_entity": pred_ent,
                            "similarity": entity_similarity(true_ent, pred_ent)
                        })
                        matched_pred_entities.add(pred_ent)
                        matched = True
                        entity_tp += 1
                        break
                if not matched:
                    entity_fn += 1
            
            entity_fp = len(pred_entities - matched_pred_entities)
            
            # Topic matching
            true_topics = set(true_items["topics"])
            pred_topics = set(pred_items["topics"])
            
            topic_matches = []
            topic_tp = 0
            topic_fp = 0
            topic_fn = 0
            
            matched_pred_topics = set()
            
            for true_topic in true_topics:
                matched = False
                for pred_topic in pred_topics:
                    if entity_similarity(true_topic, pred_topic) >= threshold:
                        topic_matches.append({
                            "true_topic": true_topic,
                            "pred_topic": pred_topic,
                            "similarity": entity_similarity(true_topic, pred_topic)
                        })
                        matched_pred_topics.add(pred_topic)
                        matched = True
                        topic_tp += 1
                        break
                if not matched:
                    topic_fn += 1
            
            topic_fp = len(pred_topics - matched_pred_topics)
            
            # Calculate precision, recall, f1
            def calculate_prf1(tp, fp, fn):
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                return {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                }
            
            return {
                "entity_matching": {
                    "matches": entity_matches,
                    "metrics": calculate_prf1(entity_tp, entity_fp, entity_fn),
                    "true_count": len(true_entities),
                    "pred_count": len(pred_entities)
                },
                "topic_matching": {
                    "matches": topic_matches,
                    "metrics": calculate_prf1(topic_tp, topic_fp, topic_fn),
                    "true_count": len(true_topics),
                    "pred_count": len(pred_topics)
                }
            }

        matching_metrics = calculate_matching_metrics(true_items, pred_items)

        # Store sample results with calculated metrics
        sample_metrics = calculate_final_metrics({
            "entity": metrics["entity"],
            "topical": metrics["topical"]
        })

        true_attitude_summary = count_attitudes(true_data, is_polar=False)
        pred_attitude_summary = count_attitudes(pred_data, is_polar=True)

        # Update global attitude counts
        for key in ["entity", "topical"]:
            for polarity in ["Positive", "Neutral", "Negative"]:
                overall_attitude_summary["true_data"][key][polarity] += true_attitude_summary[key][polarity]
                overall_attitude_summary["pred_data"][key][polarity] += pred_attitude_summary[key][polarity]

        
        all_metrics["samples"][f"sample_{i}"] = {
            "metrics": sample_metrics,
            "true_data": {
                "entities": true_items["entities"],
                "topics": true_items["topics"],
                "attitude_summary": true_attitude_summary
            },
            "pred_data": {
                "entities": pred_items["entities"],
                "topics": pred_items["topics"],
                "attitude_summary": pred_attitude_summary
            },
            "matching_metrics": matching_metrics,
            "matched_pairs": {
                "entity": metrics["entity"].get("matched_pairs", []),
                "topical": metrics["topical"].get("matched_pairs", [])
            },
            "unmatched_pairs": {
                "entity": {
                    "true_unmatched": metrics["entity"].get("unmatched_true_pairs", []),
                    "pred_unmatched": metrics["entity"].get("unmatched_pred_pairs", [])
                },
                "topical": {
                    "true_unmatched": metrics["topical"].get("unmatched_true_pairs", []),
                    "pred_unmatched": metrics["topical"].get("unmatched_pred_pairs", [])
                }
            }
        }

        # Final aggregation
        for category in ["entity", "topical"]:
            for count_type in ["pair_counts", "attitude_counts", "justification_counts"]:
                for metric, value in metrics[category][count_type].items():
                    all_metrics[category][count_type][metric] += value
            
            for count in metrics[category]["sample_counts"]:
                all_metrics[category]["sample_counts"][count] = metrics[category]["sample_counts"][count]
                all_metrics[category]["total_counts"][count] += metrics[category]["sample_counts"][count]

    # Calculate final metrics

    all_metrics["attitude_summary"] = overall_attitude_summary


    final_metrics = calculate_final_metrics({
        "entity": {
            "pair_counts": all_metrics["entity"]["pair_counts"],
            "attitude_counts": all_metrics["entity"]["attitude_counts"],
            "justification_counts": all_metrics["entity"]["justification_counts"],
            "sample_counts": all_metrics["entity"]["total_counts"]
        },
        "topical": {
            "pair_counts": all_metrics["topical"]["pair_counts"],
            "attitude_counts": all_metrics["topical"]["attitude_counts"],
            "justification_counts": all_metrics["topical"]["justification_counts"],
            "sample_counts": all_metrics["topical"]["total_counts"]
        }
    })

    for category in ["entity", "topical"]:
        all_metrics[category].update(final_metrics.get(category, {}))

    all_metrics["attitude_summary"] = overall_attitude_summary

    print("\nEvaluation Summary:")
    print("=" * 50)
    visualize_metrics(all_metrics, True)

    return all_metrics



if __name__ == "__main__":
    true_dataset_path = "./test_dataset/test.dataset"
    pred_files_path = "./test.dataset.pckl"
    polar_files_path = "./Polar_New/attitudes"

    brexit_dataset_path = "../test-brexit/gpt_attitudes"
    brexit_polar_path = "./brexit-test/attitudes"

    # evaluation_results = evaluate_predictions(true_dataset_path, pred_files_path)

    brexit_results = evaluate_predictions_polar_recursive(brexit_dataset_path, brexit_polar_path)

    with open("brexit_detailed_evaluation_results.json", "w") as f:
        json.dump(convert_numpy_types(brexit_results), f, indent=4)

    polar_results = evaluate_predictions_polar(true_dataset_path, polar_files_path)
    # Save results to a JSON file with type conversion
    # with open("detailed_evaluation_results.json", "w") as f:
    #     json.dump(convert_numpy_types(evaluation_results), f, indent=4)

    with open("polar_detailed_evaluation_results.json", "w") as f:
        json.dump(convert_numpy_types(polar_results), f, indent=4)
    