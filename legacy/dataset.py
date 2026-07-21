import csv
import fasttext
import urllib.request
import os
import json
import numpy as np
from cltk.sentence.lat import LatinPunktSentenceTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login


def filter_latin_samples(sample, lang_model):
    text = sample.get("text")
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False
    clean_text = text.replace('\n', ' ')
    predictions = lang_model.predict(clean_text)
    predicted_lang = predictions[0][np.argmax(np.array(predictions[1]))]
    confidence = predictions[1][0]
    if predicted_lang == '__label__la' and confidence > 0.5:
        return True
    return False


def filter_huggingface_dataset():
    lang_model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    lang_model_path = "./lid.176.bin"
    if not os.path.exists(lang_model_path):
        urllib.request.urlretrieve(lang_model_url, lang_model_path)
    lang_model = fasttext.load_model(lang_model_path)
    sentence_tokenizer = LatinPunktSentenceTokenizer()
    print(f"[DONE] Loading fasttext model from '{lang_model_path}'")

    print(f"[START] Loading Dataset...")
    dataset = load_dataset("Fece228/latin-literature-dataset-170M", split="train", streaming=True)
    print(f"[DONE] Dataset loaded!")

    first_raw_sample = next(iter(dataset))
    print(f"DEBUG: So sieht ein Datenpunkt wirklich aus:\n{first_raw_sample.keys()}\n")

    fn_kwargs = {"lang_model": lang_model}
    filtered_stream = dataset.filter(filter_latin_samples, fn_kwargs=fn_kwargs)

    print(f"[START] Filtering dataset...")
    output_file = "latin-literature/latin_literature_clean.jsonl"
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        saved_count = 0
        for sample in filtered_stream:
            json_string = json.dumps(sample, ensure_ascii=False)
            f.write(json_string + "\n")
            saved_count += 1
            print("SAVED")
            if saved_count == 1:
                return

    print(f"[FINISHED] Fertig! Insgesamt {saved_count} Zeilen exportiert.")






if __name__ == "__main__":
    load_dotenv()
    login(token=os.environ.get("HUGGINGFACE_API_TOKEN"))
    filter_huggingface_dataset()