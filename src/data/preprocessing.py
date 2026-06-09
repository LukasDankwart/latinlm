import os
import json
import re
import src.config as config
from lingua import Language, LanguageDetectorBuilder

"""
 ──────────────────────────────────────────────────
 ─────────── Normalization ────────────────────────────
"""
def normalize_sentence(sentence: str) -> str:
    """ Normalizes given sentence with heuristics """

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


"""
 ──────────────────────────────────────────────────
 ─────────── Filtering ────────────────────────────
"""

languages = [Language.LATIN, Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

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
    detected_language = detector.detect_language_of(sentence)
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


def process_file(input_path: str, output_dir: str) -> None:
    """ Processes specified file by extracting each sample and verifying each sentence."""

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[ERROR] Given path '{input_path}' is not a file!")

    output_file_name = os.path.basename(input_path).removesuffix(".jsonl") + "_clean.jsonl"
    output_path = os.path.join(output_dir, output_file_name)

    processed_samples_cnt = 0
    dropped_samples_cnt = 0
    verified_sentences_cnt = 0
    verified_words_cnt = 0
    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", encoding="utf-8") as outfile:
        print(f"[START] Filtering {input_path}...")
        for line in infile:
            data = json.loads(line)
            paragraph = data["text"]
            sentences = split_paragraph_to_sentences(paragraph)
            cleaned_paragraph = clean_paragraph(sentences)
            if not cleaned_paragraph:
                cleaned_sample = ""
            else:
                cleaned_sample = " ".join(cleaned_paragraph)

            # Only if the paragraph is cleaned properly!
            if cleaned_sample != "":
                words = cleaned_sample.split()
                verified_words_cnt += len(words)
                verified_sentences_cnt += len(cleaned_paragraph)

                data["text"] = cleaned_sample
                data["wrd_cnt"] = len(words)

                json.dump(data, outfile, ensure_ascii=False)
                outfile.write("\n")
                processed_samples_cnt += 1
            else:
                dropped_samples_cnt += 1

            if (processed_samples_cnt + dropped_samples_cnt) % 100 == 0:
                print(f"-- {processed_samples_cnt + dropped_samples_cnt} samples processed...")
        print(f"[END] Done filtering {input_path}...")
        print(f"-- {processed_samples_cnt} samples have been accepted.")
        print(f"-- {dropped_samples_cnt} samples have been dropped.")
        print(f"-- Total amount of words: {verified_words_cnt}")


if __name__ == "__main__":
    process_file(
        input_path="data/raw/fineweb_latin.jsonl",
        output_dir=config.PROCESSED_DATA_DIR
    )