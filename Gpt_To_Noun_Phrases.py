import os
import json

def convert_to_polar_format(json_filename, output_file):
    """Converts a single JSON file into the Polar format and saves as JSON."""
    try:
        with open(json_filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Skipping malformed JSON file: {json_filename} ({e})")
        return  # Skip this file and continue with the next

    # Use a dictionary keyed by sentence to consolidate relationships
    sentence_map = {}

    def update_entry(existing_entry, new_entry):
        """Merge new data into an existing entry."""
        existing_entity_titles = {e['title'] for e in existing_entry['entities']}
        for entity in new_entry['entities']:
            if entity['title'] not in existing_entity_titles:
                existing_entry['entities'].append(entity)

        existing_phrases = {p['ngram'] for p in existing_entry['noun_phrases']}
        for phrase in new_entry['noun_phrases']:
            if phrase['ngram'] not in existing_phrases:
                existing_entry['noun_phrases'].append(phrase)

    for topical in data.get('topical_attitudes', []):
        sentence = topical["justifications"][0]
        entry = {
            "sentence": sentence,
            "from": -1,
            "to": -1,
            "entities": [],
            "noun_phrases": []
        }
        try:
            source_entity = topical["source"]["entity"]
        except KeyError:
            source_entity = topical["source"]["topic"]
        entry["entities"].append({
            'begin': -1,
            'end': -1,
            'title': source_entity,
            'score': 1.0,
            'rank': 1.0,
            'text': source_entity,
            'types': [],
            'wikid': source_entity,
            'dbpedia': source_entity
        })

        if "topic" in topical["target"]:
            target_topic = topical["target"]["topic"]
            entry["noun_phrases"].append({
                'ngram': target_topic,
                'from': -1,
                'to': -1,
                #'topics': [target_topic]
            })

        if sentence in sentence_map:
            update_entry(sentence_map[sentence], entry)
        else:
            sentence_map[sentence] = entry

    for entity_relation in data.get('entity_attitudes', []):
        try:
            sentence = entity_relation["justifications"][0]
        except (KeyError, IndexError) as e:
            print(f"Skipping file due to missing or malformed 'justifications' in: {json_filename} ({e})")
            return  # Skip this file and continue with the next

        entry = {
            "sentence": sentence,
            "from": -1,
            "to": -1,
            "entities": [],
            "noun_phrases": []
        }

        entity1 = entity_relation["entity1"]["entity"]
        entity2 = entity_relation["entity2"]["entity"]
        for entity in [entity1, entity2]:
            entry["entities"].append({
                'begin': -1,
                'end': -1,
                'title': entity,
                'score': 1.0,
                'rank': 1.0,
                'text': entity,
                'types': [],
                'wikid': entity,
                'dbpedia': entity
            })

        if sentence in sentence_map:
            update_entry(sentence_map[sentence], entry)
        else:
            sentence_map[sentence] = entry

    polar_data ={
        'uid': os.path.basename(json_filename),
        'noun_phrases': list(sentence_map.values())
    }

    polar_data_string = json.dumps(polar_data)

    # Save the output as a JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(polar_data_string, file)

def process_directory(input_dir, output_dir):
    """Walks through a directory and processes each JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                input_file = os.path.join(root, file)
                date_string = os.path.basename(os.path.dirname(input_file))
                specific_output_dir = os.path.join(output_dir, date_string)
                os.makedirs(specific_output_dir, exist_ok=True)
                output_file = os.path.join(specific_output_dir, f"polar_{file}")
                #print(f"Processing: {file} -> {output_file}")
                convert_to_polar_format(input_file, output_file)

if __name__ == "__main__":
    input_directory = "./test-brexit/gpt_results"
    output_dir = "./test-brexit/gpt_noun_phrases"
    process_directory(input_directory, output_dir)
