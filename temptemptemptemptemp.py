import os
import re

def rename_files_in_directory(root_dir):
    """
    Recursively renames files in all subdirectories, replacing '-.-part' with '--part' in filenames.
    """
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if '.-Part' in filename:
                new_filename = filename.replace('.-Part', '-Part')
                print("a")
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except OSError as e:
                    print(f"Error renaming {old_path}: {e}")

if __name__ == "__main__":
    target_directory = "./test-brexit/gpt_attitudes"
    
    if os.path.isdir(target_directory):
        print(f"Processing directory: {target_directory}")
        rename_files_in_directory(target_directory)
        print("File renaming complete!")
    else:
        print("Error: Invalid directory path")