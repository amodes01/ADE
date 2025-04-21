import os
import json
import shutil
import re
from pathlib import Path


def organize_json_files(input_dir, output_dir, map_file):
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: The directory {input_dir} does not exist.")
        return

    # Ensure the map file exists
    if not os.path.exists(map_file):
        print(f"Error: The map file {map_file} does not exist.")
        return

    output_dir=os.path.join(output_dir, "gpt_results")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the mapping from gpt_splitter_map.json
    with open(map_file, 'r') as f:
        gpt_splitter_map = json.load(f)

    # Iterate through all JSON files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            # Normalize the file name for mapping lookup
            normalized_name = re.sub(r".-Part",r"-Part",file_name[8:])


            
            # Check if the normalized name matches any key in the mapping
            for mapped_path, folder_name in gpt_splitter_map.items():
                if normalized_name in mapped_path:
                    # Create the target folder in the output directory
                    target_folder = os.path.join(output_dir, folder_name)
                    os.makedirs(target_folder, exist_ok=True)

                    # Copy the JSON file (with its original name) to the target folder
                    shutil.copy(file_path, target_folder)
                    #print(f"Placed {normalized_name} in {target_folder}")
                    break
            else:
                # If no match is found in the mapping
                print(f"Warning: {file_name} (normalized as {normalized_name}) does not have a mapping.")

if __name__ == "__main__":
    # Example usage
    input_directory = "./ready-articles-brexit/results/authenticated"  # Replace with your input directory
    output_directory = "./test-brexit"  # Replace with your output directory
    map_filename = "./test-brexit/gpt_splitter_map.json"  # Replace with the path to your map file

    organize_json_files(input_directory, output_directory, map_filename)
