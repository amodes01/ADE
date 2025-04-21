import json
import requests
import os
import re
import time
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

# Iterate through all subfolders and process each JSON file
def process_folders(base_folder, prompt_file):
    # Read the prompt once, since it's the same for all files
    prompt = read_file(prompt_file)

    # List all JSON files in the base folder
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(base_folder, "gpt_split_articles"))
        for file in files if file.endswith(".json")
    ]

    # Progress bar for processing files
    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        for file in all_files:
            date_string = file.split("/")[-2]
            # Construct the full path for the current JSON file
            json_file_path = file

            # Read the article content from the JSON file
            with open(json_file_path, 'r') as f:
                article = f.read()

            textfield = article.split("\\\", \\\"text\\\": \\\"")  # Get everything after the text field
            article = textfield[1].strip()
            endfield = article.split("\\\"}\"")  # Get everything before the end of the JSON
            article = endfield[0]

            # Construct the JSON payload with a temperature setting
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

            # Send the POST request and handle any errors in the response
            try:
                response = requests.post(
                    'http://127.0.0.1:9846/v1/chat/completions',
                    headers={
                        'Authorization': 'Bearer abc',
                        'Content-Type': 'application/json'
                    },
                    data=json.dumps(json_payload)
                )

                # Check if the request was successful
                response.raise_for_status()

                # Attempt to extract 'content' from the response JSON
                response_json = response.json()
                content = response_json['choices'][0]['message']['content']

            except requests.exceptions.RequestException as e:
                print(f"Request error for {file}: {e}")
                content = "Request failed or invalid response format."

            except (KeyError, IndexError) as e:
                print(f"Parsing error for {file}: {e}")
                content = "Unexpected response format."

            os.makedirs(os.path.join(base_folder, "gpt_responses"), exist_ok=True)

            # Save the response to a results.json file in the current subfolder
            result_path = os.path.join(base_folder, "gpt_responses" , date_string)

            
            os.makedirs(result_path, exist_ok=True)

            with open(os.path.join(result_path, file.split("/")[-1]), 'w') as f:
                f.write(content)

            # Introduce a delay between requests to avoid server overload
            time.sleep(3)  # Adjust the sleep duration as needed

            # Update the progress bar after processing each file
            pbar.update(1)

# Base folder and prompt file
if __name__ == "__main__":
    base_folder = "./test2-unrestricted-olympics"
    prompt_file = "prompt.txt"

    start_time = time.time()
    # Call the function to process all folders and files
    process_folders(base_folder, prompt_file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"GPT_Communicator Elapsed time: {elapsed_time:.2f} seconds")
