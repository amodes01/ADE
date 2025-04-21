import os
import re
import json
import shutil

def find_matching_files(split_articles_dir, articles_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile regex pattern to match -Part[number] before .json
    part_pattern = re.compile(r'(-Part\d+)(?=\.json$)')
    
    # Iterate through all .json files in split_articles directory
    for filename in os.listdir(split_articles_dir):
        if filename.endswith('.json'):
            # Extract base filename by removing -Part[number] if present
            base_filename = part_pattern.sub('', filename)
            
            # Search through articles directory for matching files
            for root, dirs, files in os.walk(articles_dir):
                for file in files:
                    if file == base_filename:
                        # Found a match, get the parent directory name
                        parent_dir = os.path.basename(root)
                        
                        # Create corresponding directory in output_dir
                        output_subdir = os.path.join(output_dir, parent_dir)
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        # Write the original filename (with Part) to the output directory
                        output_path = os.path.join(output_subdir, filename)
                        
                        # Since you just want to output the filename, we'll create an empty file
                        # Alternatively, you could copy the original file if needed
                        with open(output_path, 'w') as f:
                            shutil.copy2(os.path.join(split_articles_dir, filename), output_path)

                        
                        print(f"Found match: {filename} -> {output_path}")
                        break  # No need to check other files in this directory

if __name__ == "__main__":
    split_articles_dir = "./test-brexit/gpt_split_articles"
    articles_dir = "./test-brexit/pre_processed_V1"
    output_dir = "./test-brexit/pre_processed"
    
    find_matching_files(split_articles_dir, articles_dir, output_dir)