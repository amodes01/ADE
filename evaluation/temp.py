# import os
# import json

# def update_json_uids(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             file_path = os.path.join(folder_path, filename)

#             # Read the raw JSON string from the file
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 raw_content = file.read().strip()  # Read as a string and remove extra spaces/newlines

#             try:
#                 data = json.loads(raw_content)  # Parse the JSON string into a dictionary
#             except json.JSONDecodeError:
#                 print(f"Skipping invalid JSON file: {filename}")
#                 continue

#             # Modify 'uid' if it exists and is "None"
#             if isinstance(data, dict) and 'uid' in data and data['uid'] == "None":
#                 data['uid'] = os.path.splitext(filename)[0]  # Remove .json extension

#             # Convert back to a JSON string (exact same format)
#             updated_content = json.dumps(data)  # Convert dict back to a string

#             # Save the modified JSON string back to the file
#             with open(file_path, 'w', encoding='utf-8') as file:
#                 file.write(json.dumps(updated_content))  # Keep it as a string

# # Example usage
# folder = "./Polar_Dataset_Articles/pre_processed/11037"
# update_json_uids(folder)
import pickle
import json
import os

attitudes_dir = './Polar_Dataset_Articles/attitudes/11037/'
output_dir = './Polar_New/attitudes/'

os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(attitudes_dir):
#     if filename.endswith('.pckl'):
#         file_path = os.path.join(attitudes_dir, filename)
#         with open(file_path, 'rb') as f:
#             print(f"Processing file: {filename}")
#             dataset = pickle.load(f)
#             for set in dataset['attitudes']:

                
#                 count = 0
#                 if 'sentence' in set:
#                     print("sentence: ", set['sentence'])
#                     print("="*50)

#                 if set['entities']:
#                         print("entities: ")
#                         for entity in set['entities']:
#                             if 'text' in entity:
#                                 print(f"  - {entity['text']}")
#                         print("="*50)
#                 if set['noun_phrases']:
#                         print("noun_phrases: ", set['noun_phrases'])
#                         for noun_phrase in set['noun_phrases']:
#                             print(f"  - {noun_phrase}")
#                         print("="*50)
#                 if set['entity_attitudes']:
#                     print("entity_attitudes: ",set['entity_attitudes'])
#                     for attitude in set['entity_attitudes']:
#                         print(count ,attitude)
#                         count += 1
#                     print("="*50)
#                     count = 0
#                 if set['noun_phrase_attitudes']:
#                     print("noun_phrase_attitudes: ", set['noun_phrase_attitudes'])
#                     for key, values in set['noun_phrase_attitudes'].items():
#                         print(f"{count} {key}:")
#                         for value in values:
#                             print(f"  - {value}")
#                         count += 1
#                     print("="*50)


polarization_dir = './Polar_Dataset_Articles/polarization/'

# Unpickle and process attitudes.pckl
attitudes_file = os.path.join(polarization_dir, 'attitudes.pckl')
if os.path.exists(attitudes_file):
    with open(attitudes_file, 'rb') as f:
        print("Processing attitudes.pckl...")
        attitudes_data = pickle.load(f)
        print("Attitudes Data:", attitudes_data)
else:
    print("attitudes.pckl not found in the directory.")

# Unpickle and process dipoles.pckl
dipoles_file = os.path.join(polarization_dir, 'dipoles.pckl')
if os.path.exists(dipoles_file):
    with open(dipoles_file, 'rb') as f:
        print("Processing dipoles.pckl...")
        dipoles_data = pickle.load(f)
        print("Dipoles Data:", dipoles_data)
else:
    print("dipoles.pckl not found in the directory.")
