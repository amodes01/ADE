import os
import time
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import tiktoken
from tqdm import tqdm  # Import tqdm for progress bars

def get_tokenizer(model_name):
    """Returns the appropriate tokenizer based on the model name."""
    if "gpt" in model_name.lower():
        return tiktoken.encoding_for_model(model_name), "openai"
    else:
        return AutoTokenizer.from_pretrained(model_name), "huggingface"

def count_tokens(text, tokenizer, tokenizer_type):
    """Counts the tokens in a text based on the specified tokenizer."""
    if tokenizer_type == "openai":
        tokens = tokenizer.encode(text)
    else:  # Hugging Face tokenizers
        tokens = tokenizer.tokenize(text)
    return len(tokens)

def split_text(text, max_tokens, tokenizer, tokenizer_type):
    """
    Splits text into parts where each part has a maximum number of tokens,
    using LangChain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=0
    )
    return splitter.split_text(text)

def process_files_in_folder(folder_path, prompt_file, model_name, max_tokens):

    file_map = {}

    tokenizer, tokenizer_type = get_tokenizer(model_name)

    with open(prompt_file, 'r') as f:
        prompt = f.read()

    prompt_tokens = count_tokens(prompt, tokenizer, tokenizer_type)
    available_tokens = max_tokens - prompt_tokens
    if available_tokens <= 0:
        raise ValueError("The prompt is too long to fit within the token window.")

    # List all JSON files and directories in folder_path
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith(".json")
    ]

    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        for item_path in all_files:
            date_string = item_path.split("/")[-2]
            #print(date_string)
            with open(item_path, 'r') as f:
                article = f.read()
            textfield = article.split("\\\", \\\"text\\\": \\\"")  # Get everything after the text field
            article = textfield[1].strip()
            endfield = article.split("\\\"}\"")  # Get everything before the end of the JSON
            article = endfield[0]

            parts = split_text(article, available_tokens, tokenizer, tokenizer_type)

            # Output the parts in a new directory
            directory = os.path.join(folder_path, "../gpt_split_articles", date_string)
            #directory = os.path.join(folder_path, "../gpt_split_articles")

            if not os.path.exists(directory):
                os.makedirs(directory)

            for i, part in enumerate(parts, 1):
                if len(parts) > 1:  # If splitting happened
                    file_path = os.path.join(directory, f'{os.path.basename(item_path)[:-5]}-Part{i}.json')
                else:
                    file_path = os.path.join(directory, os.path.basename(item_path))

                file_map[file_path] = date_string

                with open(file_path, 'w') as file:
                    file.write(textfield[0])
                    file.write("\\\", \\\"text\\\": \\\"")  # Add the removed lines to make it proper JSON structure
                    file.write(part)
                    file.write("\\\"}\"")
            
            pbar.update(1)  # Update the progress bar after processing each file

           # with open(os.path.join(folder_path.replace('pre_processed/', ''), "gpt_splitter_map.json"), "w") as f: json.dump(file_map, f)

# Example usage
if __name__ == "__main__":
    folder_path = "./temp/temp_eval_set/"
    prompt_file = "prompt.txt"
    model_name = "gpt-3.5-turbo"
    max_tokens = 4096
    start_time = time.time()
    process_files_in_folder(folder_path, prompt_file, model_name, max_tokens)
    end_time = time.time()
    elapsed_time_end = end_time - start_time
    print(f"Total Elapsed time: {elapsed_time_end:.2f} seconds")
