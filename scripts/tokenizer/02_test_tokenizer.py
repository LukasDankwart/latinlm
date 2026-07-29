import json
import os
from tokenizers import Tokenizer
import src.config as config
from src.tests.tokenizer_metrics import evaluate_tokenizer

"""
    This scripts measures some metrics regarding the tokenizer.
    Therefore, some examples from fineweb dataset are used
"""

if __name__ == "__main__":
    print(f"[START] Evaluating tokenizer with data from processed fineweb data...")

    # Define path to tokenizer
    tokenizer_dir = config.TOKENIZED_DATA_DIR
    tokenizer_path = os.path.join(tokenizer_dir, "bpe_tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        raise RuntimeError(f"[red]The file {tokenizer_path} does not exist! Pre-run tokenizer train recommended.")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Determine path to process fineweb_data
    processed_dir = config.DEDUPLICATED_DATA_DIR
    suffix = "deduplicated_corpus.jsonl"

    if not os.path.isfile(os.path.join(processed_dir, suffix)):
        raise RuntimeError(f"[red]The file {os.path.join(processed_dir, suffix)} does not exist!")

    # Fetch some data from the processed .json
    samples = []
    with open(os.path.join(processed_dir, suffix), "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            if (idx + 1) % 100 == 0:
                samples.append(json.loads(line).get("text", ""))
            idx += 1
            if idx > 1000000:
                break

    # Measure results for each fetched sample
    wrd_cnt = 0
    token_count = 0
    fertility = 0
    compression = 0
    single_chars = 0
    for sample in samples:
        results = evaluate_tokenizer(tokenizer=tokenizer, text=sample)
        wrd_cnt += results.get("word_count", 0)
        token_count += results.get("token_count", 0)
        fertility += results.get("fertility", 0)
        compression += results.get("compression", 0)
        single_chars += results.get("single_chars", 0)

    # Just console output
    print(f" -- Number samples: {len(samples)}")
    print(f" -- Overall word count: {wrd_cnt}")
    print(f" -- Overall token count: {token_count}")
    avg_wrd_cnt = wrd_cnt / len(samples)
    avg_token_count = token_count / len(samples)
    avg_fertility = fertility / len(samples)
    avg_compression = compression / len(samples)
    avg_single_chars = single_chars / len(samples)
    print(f" -- Avg word count: {round(avg_wrd_cnt, 2)}")
    print(f" -- Avg token count: {round(avg_token_count, 2)}")
    print(f" -- Avg fertility: {round(avg_fertility, 2)}")
    print(f" -- Avg compression: {round(avg_compression, 2)}")
    print(f" -- Avg single chars: {round(avg_single_chars, 2)}")

    print(f"[END] Done evaluating tokenizer...")
