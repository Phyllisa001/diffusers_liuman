import os
import json
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def load_txt_files(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    return txt_files

def load_png_file(txt_file_path, png_folder_path):
    base_name = txt_file_path.split(".")[0].split("/")[-1]
    png_file_name = base_name + ".png"
    return os.path.join(png_folder_path, png_file_name)

def extract_text(txt_file_path):
    with open(txt_file_path, 'r') as file:
        return file.read()

def extract_most_important_words(text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names()
    important_words = sorted(zip(feature_names, tfidf_matrix.toarray()[0]), key=lambda x: x[1], reverse=True)[:5]
    return [word[0] for word in important_words]

def process_files(txt_folder_path, png_folder_path):
    data = []
    txt_files = load_txt_files(txt_folder_path)
    for txt_file_path in tqdm(txt_files):
        png_file_path = load_png_file(txt_file_path, png_folder_path)
        text = extract_text(txt_file_path)
        important_words = extract_most_important_words(text)
        entry = {
            "txt_file_path": txt_file_path,
            "png_file_path": png_file_path,
            "prompt": text,
            "important_words": important_words
        }
        data.append(entry)
    return data

def save_json(data, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
FOLDER = "all_snow"
txt_folder_path = '/mnt/ve_share/songyuhao/generation/data/train/GAN/%s/pmps' % FOLDER
png_folder_path = '/mnt/ve_share/songyuhao/generation/data/train/GAN/%s/imgs' % FOLDER
output_file_path = '/mnt/ve_share/songyuhao/generation/data/train/GAN/%s/index.json' % FOLDER

processed_data = process_files(txt_folder_path, png_folder_path)
save_json(processed_data, output_file_path)
print(output_file_path)
