import os
import numpy as np
from cltk.sentence.lat import LatinPunktSentenceTokenizer
import pandas as pd
from pathlib import Path
import urllib.request
import re
import fasttext
from huggingface_hub import hf_hub_download

lang_model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
lang_model_path = "./lid.176.bin"
if not os.path.exists(lang_model_path):
    urllib.request.urlretrieve(lang_model_url, lang_model_path)
lang_model = fasttext.load_model(lang_model_path)
sentence_tokenizer = LatinPunktSentenceTokenizer()

def is_valid_latin_sentence(text):
    """
    Checks given sentence for probability of being valid latin.

    1. Checks if num words is < 2.
    2. Checks if at least 70% are raw latin chars and no html garbage or anything else.
    3. Checks if each sentence contains one finite predicate.
    """
    words = text.split()
    if len(words) < 2:
        return False

    alpha_chars = len(re.sub(r'[^a-zA-Z]', '', text))
    if len(text) == 0 or (alpha_chars / len(text)) < 0.7:
        return False
    # check for gramma
    try:
        predictions = lang_model.predict(text.replace('\n', ' '))
        predicted_lang = predictions[0][np.argmax(np.array(predictions[1]))]
        confidence = predictions[1][0]
        if predicted_lang == '__label__la' and confidence > 0.5:
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Processing sentence '{text}': {e}")
        return False


def clean_paragraph(paragraph):
    """
    Checks each sentence for given paragraphs to filter html garbage or other discrepancies in data.
    """
    if not isinstance(paragraph, str):
        return ""
    sentences = sentence_tokenizer.tokenize(paragraph)
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if is_valid_latin_sentence(sentence):
            clean_sentences.append(sentence)
    return "".join(clean_sentences)


def process_raw_csv(file_path, output_folder):
    """
    Processes raw csv file and calls preprocessing for every row.
    """
    if not os.path.isfile(file_path):
        raise RuntimeError(f'Given CSV file not found: {file_path}')
    input_path = Path(file_path)
    output_path = Path(output_folder)
    print(f"\n [START] Processing {input_path.name}...")
    df = pd.read_csv(input_path)
    if 'text' not in df.columns:
        print(f"Skipping {file_path} as it has no text column.")
        return
    df['clean_text'] = df['text'].apply(clean_paragraph)
    df = df[df['clean_text'].str.strip() != ""]
    file_name = input_path.name.removesuffix(".csv")
    new_filename = output_path / f"{file_name}_filtered.csv"
    df.to_csv(new_filename, index=False)
    print(f"[DONE] Processing {new_filename}! \n")


def process_directory(input_folter, output_folder):
    """
    Fetches all csv paths from given folder and calls preprocessing for every csv file.
    """
    valid_extensions = [".csv"]
    csv_paths = []
    for filename in os.listdir(input_folter):
        if filename.endswith(*valid_extensions):
            csv_paths.append(os.path.join(input_folter, filename))
    for csv_path in csv_paths:
        process_raw_csv(csv_path, output_folder)


if __name__ == "__main__":

    #process_directory("latinlibrary/raw/", output_folder="latinlibrary/filtered/")

    file_path = "latinlibrary/raw/apuleius.csv"
    input_path = Path(file_path)
    print(f"\n [START] Processing {input_path.name}...")
    df = pd.read_csv(input_path)
    print(np.sum(df['word_count']))