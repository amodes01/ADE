import os
import json
from tqdm import tqdm

def fix_json(content):
    """
    Fix improperly formatted JSON content.
    Ensures the content starts with '{' and ends with '}'.
    """
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError("Invalid JSON structure: Cannot locate proper JSON delimiters.")
    
    # Trim to the proper JSON structure
    return content[start_idx:end_idx + 1]

def main():
    base_folder = "./test-brexit-v2"
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(base_folder, "groq_responses"))
        for file in files if file.endswith(".json")
    ]

    os.makedirs(os.path.join(base_folder, "groq_authenticated"), exist_ok=True)

    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        for file in all_files:
            try:
                # Extract the date_string from the folder name
                date_string = file.split("/")[-2]

                # Ensure result directory exists
                result_path = os.path.join(base_folder, "groq_authenticated", date_string)
                os.makedirs(result_path, exist_ok=True)

                with open(file, 'r') as f:
                    content = f.read()
                
                # Fix and validate JSON content
                try:
                    fixed_content = fix_json(content)
                    json.loads(fixed_content)  # Validate the fixed JSON structure
                except ValueError as e:
                    print(f"Skipping {file}: {e}")
                    pbar.update(1)
                    continue

                # Write the fixed content to the new location
                output_file = os.path.join(result_path, os.path.basename(file))
                with open(output_file, 'w') as f:
                    f.write(fixed_content)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
            
            pbar.update(1)

if __name__ == "__main__":
    main()
