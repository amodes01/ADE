import gzip
import json

def decompress_and_save(input_gz_path, output_json_path):
    """
    Reads a gzipped JSON file and saves it as an uncompressed JSON file.
    
    Args:
        input_gz_path (str): Path to the input .json.gz file
        output_json_path (str): Path for the output .json file
    """
    # Read the compressed file
    with gzip.open(input_gz_path, 'rt', encoding='utf-8') as f_in:
        data = json.load(f_in)
    
    # Write the uncompressed JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)

# Usage example:
decompress_and_save('topics.json.gz', 'topics_uncompressed.json')
print("File decompressed and saved successfully!")