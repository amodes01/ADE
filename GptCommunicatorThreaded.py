import json
import requests
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to read a file
def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

# Function to extract 'text' from improperly formatted JSON-like strings
def extract_text_from_string(data_str):
    match = re.search(r'"text":\s*"(.*?)"', data_str)
    if match:
        return match.group(1)
    return None

# Function to process a single JSON file
def process_json_file(prompt, base_folder, file):
    json_file_path = os.path.join(base_folder, file)
    
    # Read the article content from the JSON file
    with open(json_file_path, 'r') as f:
        article = f.read()

    textfield = article.split("\\\", \\\"text\\\": \\\"")
    article = textfield[1].strip()
    endfield = article.split("\\\"}\"")
    article = endfield[0]

    # Construct the JSON payload
    json_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\n{article}"
            }
        ],
        "temperature": 0.2,
        "stream": False
    }

    try:
        response = requests.post(
            'http://127.0.0.1:9846/v1/chat/completions',
            headers={
                'Authorization': 'Bearer abc',
                'Content-Type': 'application/json'
            },
            data=json.dumps(json_payload)
        )
        response.raise_for_status()
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Request error for {file}: {e}")
        content = "Request failed or invalid response format."
    except (KeyError, IndexError) as e:
        print(f"Parsing error for {file}: {e}")
        content = "Unexpected response format."

    # Save the response to a results.json file
    result_path = os.path.join(f'{base_folder}/results', f'Results-{file}')
    os.makedirs(f'{base_folder}/results', exist_ok=True)
    with open(result_path, 'w') as f:
        f.write(content)

    return content

# Main function to process all JSON files in parallel
def process_folders(base_folder, prompt_file):
    prompt = read_file(prompt_file)
    json_files = [file for file in os.listdir(base_folder) if file.endswith('.json')]

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(process_json_file, prompt, base_folder, file): file for file in json_files}

        # Wrap futures in tqdm for progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file = futures[future]
            try:
                future.result()  # This will raise any exceptions caught during processing
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    # Base folder and prompt file
    base_folder = "./ready-articles-brexit-v2"
    prompt_file = "prompt.txt"

    start_time = time.time()
    process_folders(base_folder, prompt_file)
    end_time = time.time()
    elapsed_time_end = end_time - start_time
    print(f"Total Elapsed time: {elapsed_time_end:.2f} seconds")
