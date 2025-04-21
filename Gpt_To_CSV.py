import os
import json
import pandas as pd
from tqdm import tqdm

df = []
unmatched_gpt_files = []

def add_to_list(path_article, path_response):
    col1 = ""
    col2 = ""

    with open(path_article, 'r', encoding="utf-8") as file:
        article = file.read()

        textfield = article.split("\\\", \\\"text\\\": \\\"")  # Get everything after the text field
        article = textfield[1].strip()
        endfield = article.split("\\\"}\"")  # Get everything before the end of the JSON
        article = endfield[0]
        col1 = article
    
    with open(path_response, "r", encoding="utf-8") as file:
        response = file.read()
        col2 = response
    
    df.append(
        {
        "text": col1,
        "output":col2
        }
    )



def find_matching_files(source_dir, sub_dir_gpt, sub_dir_articles):
    path_gpt = os.path.join(source_dir, sub_dir_gpt)
    path_art = os.path.join(source_dir, sub_dir_articles)
    
    if not os.path.exists(path_gpt) or not os.path.exists(path_art):
        print("One or both of the directories do not exist.")
        return
    
    file_map_articles = []
    for file in os.listdir(path_art):
        file_map_articles.append(file)
    
    for folder in tqdm(os.listdir(path_gpt), desc="Processing Folders"):
        folder_path_gpt = os.path.join(path_gpt, folder)
        
        if os.path.isdir(folder_path_gpt):
            for file in tqdm(os.listdir(folder_path_gpt), desc=f"Processing Files in {folder}", leave=False):
                file_path_gpt = os.path.join(folder_path_gpt, file)
                
                if os.path.isfile(file_path_gpt) and file in file_map_articles:
                    add_to_list(os.path.join(path_art, file), file_path_gpt)



def find_matching_files_v2(source_dir, sub_dir_gpt, sub_dir_articles):
    path_gpt = os.path.join(source_dir, sub_dir_gpt)
    path_art = os.path.join(source_dir, sub_dir_articles)
    
    if not os.path.exists(path_gpt) or not os.path.exists(path_art):
        print("One or both of the directories do not exist.")
        return
    
    file_map_articles = set(os.listdir(path_art))
    unmatched_files = []
    
    for file in tqdm(os.listdir(path_gpt), desc="Processing files"):
        file_path_gpt = os.path.join(path_gpt, file)

        if os.path.isfile(file_path_gpt):
            if file in file_map_articles:
                add_to_list(os.path.join(path_art, file), file_path_gpt)
            else:
                unmatched_files.append(file)
    
    if unmatched_files:
        print("Unmatched GPT files:")
        for file in unmatched_files:
            print(file)
        unmatched_gpt_files.extend(unmatched_files)


if __name__ == "__main__":
    source_directory = "./test-brexit"
    directory_articles = "gpt_split_articles"
    directory_gpt = "gpt_results"
    
    find_matching_files(source_directory, directory_gpt, directory_articles)
    find_matching_files_v2("./ready-articles-musk", "results-v1", "articles")
    find_matching_files_v2("./ready-articles-boxing", "results-v1/authenticated", "articles")
    find_matching_files_v2("./ready-articles-supper", "results/authenticated", "articles")
    find_matching_files_v2("./ready-articles-vaccine", "results/authenticated", "articles")


    df = pd.DataFrame.from_dict(df)
    df.to_csv('attitude_fine-tune.csv')
