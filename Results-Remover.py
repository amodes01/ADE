import os

def rename_files_in_folder(folder_path):
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            new_filename = filename
            
            # Check if the filename starts with 'Results-' and remove it
            if filename.startswith("Results-"):
                new_filename = filename[8:]
            
            # Replace "-.-Part" anywhere in the filename with "--Part"
            new_filename = new_filename.replace(".-Part", "-Part")

            # Only rename if the filename has changed
            if new_filename != filename:
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, new_filename)

                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")

# Usage
folder_path = "./ready-articles-boxing/articles" 
rename_files_in_folder(folder_path)
