import os
import json

def update_uid_fields(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        outer_data = json.load(f)  # This is a string
                        inner_data = json.loads(outer_data)  # Now it's a dict
                    
                    uid_value = os.path.splitext(file)[0]
                    
                    if "uid" in inner_data:
                        inner_data["uid"] = uid_value

                        # Convert inner_data back to string and dump as JSON string
                        updated_data = json.dumps(inner_data)
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(updated_data, f, ensure_ascii=False)
                        print(f"Updated UID in: {file_path}")
                    else:
                        print(f"No 'uid' field found in: {file_path}")
                
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error processing {file_path}: {e}")

# Replace this path with your actual directory
input_directory = "./brexit-test/pre_processed"
update_uid_fields(input_directory)
