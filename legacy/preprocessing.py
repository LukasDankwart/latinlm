import os
import numpy as np
from cltk.sentence.lat import LatinPunktSentenceTokenizer
import pandas as pd
from pathlib import Path
import urllib.request
import re
import fasttext
import json

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

    return True


def classify_language(paragraph):
    # check for gramma
    try:
        predictions = lang_model.predict(paragraph.replace('\n', ' '))
        predicted_lang = predictions[0][np.argmax(np.array(predictions[1]))]
        confidence = predictions[1][0]
        if predicted_lang == '__label__la':
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Processing sentence '{paragraph}': {e}")
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
            # Remove numbers in beginning of sentences
            pattern = re.compile(r'^\d+[.)]?\s*')
            sentence_cleaned = pattern.sub('', sentence)
            if len(sentence_cleaned) > 0:
                sentence_cleaned = sentence_cleaned[0].upper() + sentence_cleaned[1:]
            clean_sentences.append(sentence_cleaned)
    cleaned_paragraph = " ".join(clean_sentences)
    if classify_language(cleaned_paragraph):
        return cleaned_paragraph
    else:
        return ""


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


def process_raw_json(file_path, output_path):
    if not os.path.isfile(file_path):
        raise RuntimeError(f'Given .jsonl file not found: {file_path}')
    input_path = Path(file_path)
    output_path = Path(output_path)
    print(f"\n [START] Processing {input_path.name}...")
    with open(input_path, 'r', encoding='utf-8') as infile, \
        open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if "text" in data:
                cleaned_paragraph = clean_paragraph(data['text'])
                if cleaned_paragraph == "":
                    continue
                words = cleaned_paragraph.split()
                words = [
                    w for w in words
                    if not is_roman_numeral(w) and re.search(r'[a-zA-Z]', w)
                ]
                word_count = len(words)
                data["text"] = cleaned_paragraph
                data["word_count"] = word_count
            json_string = json.dumps(data, ensure_ascii=False)
            outfile.write(json_string + "\n")
    print(f"[DONE] Processing {input_path.name}! \n")


def is_roman_numeral(word):
    clean_word = re.sub(r'[\W_]+', '', word)
    if not clean_word:
        return False
    pattern = r'^(?i)(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(pattern, clean_word))


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


def count_overall_words(folder):
    allowed_extension = [".jsonl"]
    word_count = 0
    for filename in os.listdir(folder):
        if filename.endswith(*allowed_extension):
            for line in open(os.path.join(folder, filename), "r"):
                data = json.loads(line)
                word_count += data["word_count"]
    return word_count



if __name__ == "__main__":

    #process_directory("latinlibrary/raw/", output_folder="latinlibrary/filtered/")

    """file_path = "latinlibrary/raw/apuleius.csv"
    input_path = Path(file_path)
    print(f"\n [START] Processing {input_path.name}...")
    df = pd.read_csv(input_path)
    print(np.sum(df['word_count']))"""

    input_dir = "latinlibrary/raw"
    output_dir = "latinlibrary/filtered"
    names = [
        "ammianus.jsonl",
        "apuleius.jsonl",
        "augustus.jsonl",
        "victor.jsonl",
        "caesar.jsonl",
        "cato.jsonl",
        #"catullus.jsonl",
        "cicero.jsonl",
        #"claudian.html.jsonl",   # Differs in subpage configuration
        "curtius.html.jsonl", # Differs in subpage configuration
        #"enn",     # No subpages
        "eutropius.html.jsonl",
        "florus.html.jsonl",
        "frontinus.html.jsonl",
        "gellius.html.jsonl",
        "sha.html.jsonl",
        "hor.html.jsonl",
        "justin.html.jsonl",
        "juvenal.html.jsonl",
        "liv.html.jsonl",
        "lucan.html.jsonl",
        "lucretius.html.jsonl",
        "martial.html.jsonl",
        "nepos.html.jsonl",
        "ovid.html.jsonl",
        # "persius.html",   # No subpages
        "petronius.html.jsonl",
        "phaedrus.html.jsonl",
        "plautus.html.jsonl",
        "pliny1.html.jsonl",
        "pliny.html.jsonl",
        "prop.html.jsonl",
        "quintilian.html.jsonl",
        "sall.html.jsonl",
        "seneca.html.jsonl",
        "sen.html.jsonl",
        "silius.html.jsonl",
        "statius.html.jsonl",
        "suet.html.jsonl",
        # "sulpicia.html",   # No subpages
        "tac.html.jsonl",
        "ter.html.jsonl",
        "tib.html.jsonl",
        "valeriusflaccus.html.jsonl",
        "valmax.html.jsonl",
        "varro.html.jsonl",
        "vell.html.jsonl",
        "verg.html.jsonl",
        "vitruvius.html.jsonl"
    ]


    """for name in names:
        file_path = os.path.join(input_dir, name)
        filtered_file_path = os.path.join(output_dir, name)
        process_raw_json(file_path, filtered_file_path)"""

    filtered_words_overall = count_overall_words(output_dir)
    print(f"\n Filtered data contains: {filtered_words_overall} words\n")




