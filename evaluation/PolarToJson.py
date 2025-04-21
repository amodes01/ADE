import pickle
import json
import os

attitudes_dir = './Polar_Dataset_Articles/attitudes/11037/'
output_dir = './Polar_New/attitudes/'

os.makedirs(output_dir, exist_ok=True)

def map_attitude(value):
    if isinstance(value, list) and value:
        value = sum(value) / len(value)  
    if value > 0:
        return "Positive"
    elif value < 0:
        return "Negative"
    return "Neutral"

def get_entity_text(entity_id, entities_list):
    """Find entity text by matching either wikid or dbpedia"""
    for entity in entities_list:
        if entity.get('wikid') == entity_id or entity.get('dbpedia') == entity_id:
            return entity.get('text', entity_id)
    return entity_id

for filename in os.listdir(attitudes_dir):
    if filename.endswith('.pckl'):
        file_path = os.path.join(attitudes_dir, filename)
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            output_data = []
            has_any_attitudes = False

            for item in dataset.get('attitudes', []):
                all_entities = item.get('entities', [])
                transformed = {}

                sentence = item.get('sentence', '')

                # Process entity_attitudes
                entity_attitudes = item.get('entity_attitudes', {})
                if entity_attitudes:
                    transformed["entity_attitudes"] = []
                    for entities, score in entity_attitudes.items():
                        entity1, entity2 = entities[0], entities[1]
                        attitude = map_attitude(score)
                        transformed["entity_attitudes"].append({
                            "entity1": {"entity": get_entity_text(entity1, all_entities), "references": [entity1]},
                            "entity2": {"entity": get_entity_text(entity2, all_entities), "references": [entity2]},
                            "justifications": [sentence],
                            "attitude": attitude
                        })
                    has_any_attitudes = True

                # Process topical_attitudes
                np_attitudes = item.get('noun_phrase_attitudes', {})
                if np_attitudes:
                    transformed["topical_attitudes"] = []
                    for (source_entity, topic), scores in np_attitudes.items():
                        attitude = map_attitude(scores)
                        transformed["topical_attitudes"].append({
                            "source": {"entity": get_entity_text(source_entity, all_entities), "references": [source_entity]},
                            "target": {"topic": topic, "references": [topic]},
                            "attitude": attitude
                        })
                    has_any_attitudes = True

                # Only add if either has data or we haven't added the empty case yet
                if transformed or (not has_any_attitudes and not output_data):
                    output_data.append({
                        "entity_attitudes": transformed.get("entity_attitudes", []),
                        "topical_attitudes": transformed.get("topical_attitudes", [])
                    })

        # Save JSON
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=2, ensure_ascii=False)
        print(f"Saved JSON to {json_path}")