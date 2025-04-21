import json
import re

input_file = "./attitude-fine-tune-mixed.jsonl"
output_file = "./attitude-fine-tune-mixed_split.jsonl"

topical_start = "\"topical_attitudes\""
entity_start = "\"entity_attitudes\""

def extract_section(text, start, end):
    start_idx = text.find(start)
    end_idx = text.find(end, start_idx) if end else len(text)
    if start_idx != -1:
        return text[start_idx:end_idx].strip(', ')  # Ensure no trailing comma
    return None

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        input_text = data.get("input", "")
        output_text = data.get("output", "")
        
        topical_attitudes = extract_section(output_text, topical_start, entity_start)
        entity_attitudes = extract_section(output_text, entity_start, None)
        
        if topical_attitudes or entity_attitudes:
            if topical_attitudes:
                new_entry = {
                    "instruction": "Extract attitudes for topics from the article in JSON format and categorize them [Positive, Neutral, Negative] and justify them quoting the article", 
                    "input": input_text, 
                    "output": "{" + topical_attitudes + "}"
                }
                outfile.write(json.dumps(new_entry) + "\n")
            if entity_attitudes:
                new_entry = {
                    "instruction": "Extract attitudes between entities from the article in JSON format and categorize them [Positive, Neutral, Negative] and justify them quoting the article", 
                    "input": input_text, 
                    "output": "{" + entity_attitudes + "}"
                }
                outfile.write(json.dumps(new_entry) + "\n")
        else:
            outfile.write(line)  # Copy the original line if no attitudes are found

print(f"Processed file saved to {output_file}")
