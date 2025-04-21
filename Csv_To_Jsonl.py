import pandas as pd
import json

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def format_output(output):
    try:
        # Attempt to parse the output as json
        parsed_output = json.loads(output)
        return json.dumps(parsed_output, ensure_ascii=False)  # Return the valid json string
    except:
        # If it’s not valid json, skip
        return None

prompt = read_file("slimmerPrompt.txt")

# Load CSV file
file_path = "attitude_fine-tune.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Remove spaces in column names

# Convert to jsonl format

# Control Parameters
limit = False   # True to follow the Num of lines limit
N = 400  # num of lines
cnt = 0

output_file = "attitude_fine-tune-full.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        if limit and cnt >= N:
            break
        
        # Format the output and check if it's valid json
        formatted_output = format_output(row["output"])
        
        if formatted_output is None:
            continue  # Skip if the output is not valid json
        
        entry = {
            "instruction": "Extract attitudes between entities and topics from the article in json format and categorize them [Positive, Neutral, Negative] and justify them quoting the article",
            "input": row["text"],
            "output": formatted_output  # Only write valid json output
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")  # Write as jsonl
        cnt += 1

print(f"Converted data saved to {output_file}")
