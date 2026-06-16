import os
import json
import re
import unicodedata

import src.config as config
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

languages = [Language.LATIN, Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN]
detector = None

def init_worker():
    """ Initializes the detector. The function is called once per cpu thread that is used while preprocessing """

    global detector
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

def filter_with_lingua(sentence):
    """ Predicts language of given sentence by using lingua model. Returns predicted language """
    global detector
    if detector is not None:
        return detector.detect_language_of(sentence)
    return None
"""
 ──────────────────────────────────────────────────
 ─────────── Normalization ────────────────────────────
"""
def normalize_sentence(sentence: str) -> str:
    """ Normalizes given sentence with heuristics """

    sentence = normalize_unicode(sentence)
    sentence = normalize_spaces(sentence)
    sentence = normalize_markers(sentence)

    # Removing Latex or JSON artefacts
    sentence = re.sub(r'\{.*?\}|<.*?>|\[.*?\]', '', sentence)

    # Remove literals like \n or \t
    sentence = sentence.replace(r'\n', ' ').replace(r'\t', ' ')

    # Removing numbers in sentence beginning
    sentence = re.sub(r'^\d+\.?\s*', '', sentence)
    sentence = re.sub(r'^[MDCLXVI]+\.\s+', '', sentence)

    # Removing left over sentence marks
    sentence = re.sub(r'^[\s,;:\.\-\|]+', '', sentence)

    # Repare doubled " " or " " in front of sentence marks!
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\s+([.?!,;])', r'\1', sentence)

    sentence = sentence.strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence

def normalize_unicode(sentence: str) -> str:
    """ Normalizes given sentence compact NFC unicode """
    return unicodedata.normalize('NFD', sentence)

def normalize_markers(sentence: str) -> str:
    """ Normalizes different variants of markers """
    sentence = re.sub(r'[“”«»]', '"', sentence)
    sentence = re.sub(r'[‘’`´]', "'", sentence)
    sentence = re.sub(r'[–—]', '-', sentence)
    return sentence

def normalize_spaces(sentence: str) -> str:
    """ Normalizes the amount of spaces between words to one """
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence



"""
 ──────────────────────────────────────────────────
 ─────────── Filtering ────────────────────────────
"""

def is_quality_sentence(sentence: str) -> bool:
    """ Decides if a given sentence has good quality by heuristics """

    if sentence.count('|') >= 2:
        return False

    words = sentence.split()
    word_count = len(words)
    char_count = len(sentence)

    # 1. Throw sentences away with less than 3 words, more than 60 (probably error!) or if to fewer chars exist
    if word_count < 3 or word_count > 50 or char_count < 10:
        return False

    # 2. If 40% of chars are in Caps, the sentence is probably a headline or something else
    alpha_chars = [c for c in sentence if c.isalpha()]
    if not alpha_chars:
        return False
    upper_letter_cnt = sum(1 for c in alpha_chars if c.isupper())
    if (upper_letter_cnt / len(alpha_chars)) > 0.4:
        return False

    # 3. Compute ratio of number latin letters compared to full sentence length,
    valid_chars = sum(1 for c in sentence if c.isalpha() or c.isspace())
    if (valid_chars / char_count) < 0.75:
        return False

    # 5. Finally use a language detection model to decide if this is valid latin or not.
    detected_language = filter_with_lingua(sentence)
    if detected_language != Language.LATIN:
        return False

    return True


def filter_sentence(sentence: str) -> str:
    """ Filtering each sentence individual """

    cl_sentence = normalize_sentence(sentence)
    if not is_quality_sentence(sentence):
        return ""
    else:
        return cl_sentence


def clean_paragraph(sentences: list[str]) -> list[str]:
    """ Given sentences of one paragraph as a list, each sentence is cleaned or otherwise removed """

    cleaned_sentences = []
    for sentence in sentences:
        cl_sentence = filter_sentence(sentence)
        # Latin might include sentences with one/two words, but by scraped data, they most likely are boilerplate
        if cl_sentence != "":
            cleaned_sentences.append(cl_sentence)

    return cleaned_sentences


def split_paragraph_to_sentences(text: str) -> list[str]:
    """ Splits paragraph into sentences by reg expression """

    # Splitting text from menu or footer boilerplate
    text = re.sub(r'\n+', '. ', text)

    # Saving Rome names
    praenomina = r'\b(A|Ap|C|Cn|D|L|M|M\'|P|Q|Ser|Sex|Sp|T|Ti|Mrc)\.'
    protected_text = re.sub(praenomina, lambda m: m.group(0).replace('.', '_'), text)

    sentence_pattern = r'(?<=[.?!;])(?:[\'\"”\]\)]*)\s+(?=[A-Za-z0-9\-\"\'\[\(\„\|])'

    sentences = re.split(sentence_pattern, protected_text)

    cleaned_sentences = [s.replace('_', '.').strip() for s in sentences if s.strip()]
    return cleaned_sentences


def process_single_line(line: str):
    """ Processes one specific line. Is designed to be executed parallel multiple times. """

    try:
        data = json.loads(line)
        paragraph = data["text"]
        sentences = split_paragraph_to_sentences(paragraph)
        cleaned_paragraph = clean_paragraph(sentences)

        if not cleaned_paragraph:
            return None, 0, 0

        cleaned_sample = " ".join(cleaned_paragraph)
        if cleaned_sample == "":
            return None, 0, 0

        words = cleaned_sample.split()
        new_data = {
            "text": cleaned_sample,
            "wrd_cnt": len(words)
        }
        return new_data, len(words), len(cleaned_paragraph)

    except Exception as e:
        print(f"[ERROR] {e}")
        return None, 0, 0


def process_file(input_path: str, output_dir: str) -> int:
    """ Processes specified file by extracting each sample and verifying each sentence."""

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[ERROR] Given path '{input_path}' is not a file!")

    output_file_name = os.path.basename(input_path).removesuffix(".jsonl") + "_clean.jsonl"
    output_path = Path(os.path.join(output_dir, output_file_name))

    if output_path.exists():
        output_path.unlink()

    processed_samples_cnt = 0
    dropped_samples_cnt = 0
    verified_sentences_cnt = 0
    verified_words_cnt = 0

    print(f"[START] Start processing {input_path}...")
    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "a", encoding="utf-8") as outfile:

        with ProcessPoolExecutor(max_workers=config.NUM_CPU_WORKERS, initializer=init_worker()) as executor:

            for result in executor.map(process_single_line, infile, chunksize=1000):
                new_data, words_cnt, sentences_cnt = result

                if new_data is not None:
                    json.dump(new_data, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    processed_samples_cnt += 1
                    verified_words_cnt += words_cnt
                    verified_sentences_cnt += sentences_cnt
                else:
                    dropped_samples_cnt += 1

                total_processed = processed_samples_cnt + dropped_samples_cnt
                if total_processed % 1000 == 0:
                    print(f"-- {total_processed} samples processed...")

    print(f"[END] Done processing {input_path}...")
    print(f"-- {processed_samples_cnt} samples have been accepted.")
    print(f"-- {dropped_samples_cnt} samples have been dropped.")
    print(f"-- Total amount of words: {verified_words_cnt}")
    return verified_words_cnt

