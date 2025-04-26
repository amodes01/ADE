import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import os
import time


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def jaccard_similarity(set1, set2):
    mlb = MultiLabelBinarizer()
    combined = list(set1 | set2)
    x = [set1, set2]
    binarized = mlb.fit_transform(x)
    return jaccard_score(binarized[0], binarized[1])

def semantic_similarity(list1, list2, model):
    sentence1 = " ".join(list1)
    sentence2 = " ".join(list2)
    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return score.item()

def calculate_metrics(matched, unmatched_file1, unmatched_file2, data1, data2):
    # True Positives (correct matches)
    TP = len(matched)
    
    # False Positives (incorrect matches - we can't determine this without ground truth)
    # In this implementation, we'll consider all matches as correct (FP = 0)
    FP = 0
    
    # False Negatives (missed matches - topics that should have matched but didn't)
    # Without ground truth, we'll consider all unmatched topics as potential FNs
    FN = len(unmatched_file1) + len(unmatched_file2)
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1 Score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': TP,
        'false_positives': FP,
        'false_negatives': FN,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'total_topics_file1': len(data1),
        'total_topics_file2': len(data2),
        'matched_topics': TP,
        'unmatched_topics_file1': len(unmatched_file1),
        'unmatched_topics_file2': len(unmatched_file2)
    }

def compare_jsons(data1, data2, field='noun_phrases', N=5, method='semantic', threshold=0.6, output_path='match_output.json'):
    model = SentenceTransformer('all-MiniLM-L6-v2') if method == 'semantic' else None
    matched = {}
    used_2 = set()

    print(f"\nComparing {len(data1)} topics from file 1 to {len(data2)} topics from file 2...")

    for t1 in tqdm(data1, desc="Processing topics"):
        v1 = data1[t1]
        list1 = v1.get(field, [])[:N]
        if not list1:
            continue

        for t2 in data2:
            if t2 in used_2:
                continue
            v2 = data2[t2]
            list2 = v2.get(field, [])[:N]
            if not list2:
                continue

            if method == 'semantic':
                score = semantic_similarity(list1, list2, model)
            else:
                score = jaccard_similarity(set(list1), set(list2))

            if score >= threshold:
                matched[t1] = {
                    'matched_to': t2,
                    'score': round(score, 4),
                    'file1_items': list1,
                    'file2_items': list2
                }
                used_2.add(t2)
                break

    all_1 = set(data1.keys())
    all_2 = set(data2.keys())
    unmatched_1 = list(all_1 - matched.keys())
    unmatched_2 = list(all_2 - used_2)

    # Calculate evaluation metrics
    metrics = calculate_metrics(matched, unmatched_1, unmatched_2, data1, data2)

    output = {
        'matched': matched,
        'unmatched_file1': unmatched_1,
        'unmatched_file2': unmatched_2,
        'metrics': metrics
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_path}")
    print("\nEvaluation Metrics:")
    print(f"- Precision: {metrics['precision']}")
    print(f"- Recall: {metrics['recall']}")
    print(f"- F1 Score: {metrics['f1_score']}")
    print(f"- True Positives (matched topics): {metrics['true_positives']}")
    print(f"- False Negatives (unmatched topics): {metrics['false_negatives']}")

# === USAGE ===
file1 = './brexit-test/topics_uncompressed.json'
file2 = '../test-brexit/gpt_topics_uncompressed.json'

json1 = load_json(file1)
json2 = load_json(file2)
output_file = 'topic_matches.json'
start_time = time.time()



compare_jsons(
    json1,
    json2,
    field='noun_phrases',  # 'noun_phrases' or 'pre_processed'
    N=5,
    method='jaccard',     # 'semantic' or 'jaccard'
    threshold=0.6,
    output_path=output_file
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"topic_evaluator Elapsed time: {elapsed_time:.2f} seconds")