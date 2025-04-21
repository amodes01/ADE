import os
import time
import re
import json

def check_json_files(directory, results_dir):
    # Ensure the results directory structure matches the input directory
    os.makedirs(results_dir, exist_ok=True)
    
    for root, _, files in os.walk(directory):
        # Skip the authenticated folder to avoid redundant processing
        if root.startswith(results_dir):
            continue
        
        relative_path = os.path.relpath(root, directory)
        target_dir = os.path.join(results_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                        # Remove <think>...</think> and everything in between
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                        # Clean the content if it has extra backticks
                        if content.startswith('```json') and content.endswith('```'):
                            cleaned_content = content[7:-3].strip()
                        else:
                            cleaned_content = content

                        # Check if the cleaned content is valid JSON format
                        try:
                            json.loads(cleaned_content)  # Validate JSON
                            new_file_path = os.path.join(target_dir, filename)
                            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                                new_file.write(cleaned_content)
                        except json.JSONDecodeError:
                            print(f"{filename}: Invalid JSON format, not copied.")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    base_directory = './test-musk/deepseek_responses'
    authenticated_directory = os.path.join(base_directory, 'authenticated')
    check_json_files(base_directory, authenticated_directory)
    end_time = time.time()
    elapsed_time_end = end_time - start_time
    print(f"Total Elapsed time: {elapsed_time_end:.2f} seconds")
