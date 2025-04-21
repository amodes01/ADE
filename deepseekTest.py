import json
import requests
import os
import re
import time
import argparse
import ollama
from tqdm import tqdm  # Import tqdm for progress bars

# Function to read a file
def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

# Function to extract 'text' from improperly formatted JSON-like strings
def extract_text_from_string(data_str):
    match = re.search(r'"text":\s*"(.*?)"', data_str)
    if match:
        return match.group(1)  # Return the captured text content
    return None

# Function to call GPT-3.5-Turbo via API request
def call_gpt35(prompt, article):
    json_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\n{article}"
            }
        ],
        "temperature": 0.2,  # Adjust the temperature as needed
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
        return response_json['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return "Request failed or invalid response format."

    except (KeyError, IndexError) as e:
        print(f"Parsing error: {e}")
        return "Unexpected response format."

# Function to call DeepSeek-R1 via Ollama
def call_deepseek(prompt, article):
    try:
        start_time = time.time()
        response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": f"{prompt}\n\n{article}"}])
        end_time = time.time()

        print(f"Done in {end_time - start_time:.2f} seconds")
        return response["message"]["content"]
    except Exception as e:
        print(f"Ollama error: {e}")
        return "Ollama request failed."

# Main function to process folders
def process_folders(base_folder, prompt_file, use_deepseek=False):
    prompt = read_file(prompt_file)

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(base_folder, "gpt_split_articles"))
        for file in files if file.endswith(".json")
    ]

    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        for file in all_files:
            date_string = file.split("/")[-2]
            json_file_path = file

            with open(json_file_path, 'r') as f:
                article = f.read()

            textfield = article.split("\\\", \\\"text\\\": \\\"")
            article = textfield[1].strip()
            endfield = article.split("\\\"}\"")
            article = endfield[0]

            # Call the appropriate model
            content = call_deepseek(prompt, article) if use_deepseek else call_gpt35(prompt, article)

            # Save results
            dir_name = ""
            if use_deepseek:
                dir_name = "deepseek_responses"
            else:
                dir_name = "gpt_responses"
            result_path = os.path.join(base_folder, dir_name, date_string)
            os.makedirs(result_path, exist_ok=True)

            with open(os.path.join(result_path, file.split("/")[-1]), 'w') as f:
                f.write(content)

            time.sleep(3)  # Avoid overload
            pbar.update(1)

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process articles with GPT-3.5-Turbo or DeepSeek-R1.")
    parser.add_argument("--model", choices=["gpt-3.5", "deepseek"], default="gpt-3.5", help="Choose the model to use.")
    args = parser.parse_args()

    base_folder = "./test-musk"
    prompt_file = "prompt.txt"

    start_time = time.time()
    process_folders(base_folder, prompt_file, use_deepseek=(args.model == "deepseek"))
    end_time = time.time()

    print(f"Processing completed in {end_time - start_time:.2f} seconds")
#2700-3500
#957-1000-600