import os
import time
import re
import json
import pickle

class JSONCleaner:
    """Cleans and validates JSON files, preserving directory structure."""
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def clean_json_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for root, _, files in os.walk(self.input_dir):
            if root.startswith(self.output_dir):
                continue
            relative_path = os.path.relpath(root, self.input_dir)
            target_dir = os.path.join(self.output_dir, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                            cleaned_content = content[7:-3].strip() if content.startswith('```json') and content.endswith('```') else content
                            json.loads(cleaned_content)
                            new_file_path = os.path.join(target_dir, filename)
                            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                                new_file.write(cleaned_content)
                    except json.JSONDecodeError:
                        print(f"{filename}: Invalid JSON format, not copied.")
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

class PolarJSONConverter:
    """Converts JSON files into the Polar format and saves them as JSON."""
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def convert_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json'):
                    input_file = os.path.join(root, file)
                    date_string = os.path.basename(os.path.dirname(input_file))
                    specific_output_dir = os.path.join(self.output_dir, date_string)
                    os.makedirs(specific_output_dir, exist_ok=True)
                    output_file = os.path.join(specific_output_dir, f"polar_{file}")
                    self.convert_to_polar_format(input_file, output_file)

    def convert_to_polar_format(self, json_filename, output_file):
        try:
            with open(json_filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Skipping malformed JSON file: {json_filename}")
            return
        
        sentence_map = {}
        for topical in data.get('topical_attitudes', []):
            sentence = topical['justifications'][0]
            source_entity = topical.get('source', {}).get('entity', topical.get('source', {}).get('topic', ''))
            target_topic = topical.get('target', {}).get('topic', '')
            entry = {"sentence": sentence, "entities": [{'title': source_entity}], "noun_phrases": [{'ngram': target_topic}]} if target_topic else {"sentence": sentence, "entities": [{'title': source_entity}], "noun_phrases": []}
            sentence_map[sentence] = entry
        
        polar_data = {"uid": os.path.basename(json_filename), "noun_phrases": list(sentence_map.values())}
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(polar_data, file)

class PolarPickleConverter:
    """Converts JSON files into the Polar format and saves them as pickle files."""
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def convert_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json'):
                    input_file = os.path.join(root, file)
                    date_string = os.path.basename(os.path.dirname(input_file))
                    specific_output_dir = os.path.join(self.output_dir, date_string)
                    os.makedirs(specific_output_dir, exist_ok=True)
                    output_file = os.path.join(specific_output_dir, f"polar_{file.replace('.json', '.PCKL')}")
                    self.convert_to_polar_format(input_file, output_file)

    def convert_to_polar_format(self, json_filename, output_file):
        try:
            with open(json_filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Skipping malformed JSON file: {json_filename}")
            return
        
        sentence_map = {}
        for topical in data.get('topical_attitudes', []):
            sentence = topical['justifications'][0]
            source_entity = topical.get('source', {}).get('entity', topical.get('source', {}).get('topic', ''))
            target_topic = topical.get('target', {}).get('topic', '')
            entry = {"sentence": sentence, "entities": [{'title': source_entity}], "noun_phrases": [{'ngram': target_topic}], "entity_attitudes": {}, "noun_phrase_attitudes": {}} if target_topic else {"sentence": sentence, "entities": [{'title': source_entity}], "noun_phrases": [], "entity_attitudes": {}, "noun_phrase_attitudes": {}}
            sentence_map[sentence] = entry
        
        polar_data = {"uid": os.path.basename(json_filename), "attitudes": list(sentence_map.values())}
        with open(output_file, 'wb') as file:
            pickle.dump(polar_data, file)

if __name__ == "__main__":
    start_time = time.time()
    base_dir = "./test-data"
    cleaned_dir = os.path.join(base_dir, "cleaned")
    json_output_dir = os.path.join(base_dir, "polar_json")
    pickle_output_dir = os.path.join(base_dir, "polar_pickle")
    
    cleaner = JSONCleaner(base_dir, cleaned_dir)
    cleaner.clean_json_files()
    
    json_converter = PolarJSONConverter(cleaned_dir, json_output_dir)
    json_converter.convert_files()
    
    pickle_converter = PolarPickleConverter(cleaned_dir, pickle_output_dir)
    pickle_converter.convert_files()
    
    print(f"Total Elapsed time: {time.time() - start_time:.2f} seconds")
