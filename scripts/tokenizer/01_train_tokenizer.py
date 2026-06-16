import os

import src.config as config
from src.training.tokenizer import train_bpe_tokenizer

"""
    This scripts trains a new tokenizer over the complete dataset
"""

if __name__ == "__main__":
    print(f"[START] Downloading training tokenizer with data from '{config.PROCESSED_DATA_DIR}'...")

    # Fetch path to all preprocessed data for tokenizer training
    data_dir = config.PROCESSED_DATA_DIR
    target_dir = config.TOKENIZED_DATA_DIR
    vocab_size = config.TOKENIZER_VOCAB_SIZE
    tokenizer_path = os.path.join(target_dir, "bpe_tokenizer.json")

    # Fetch paths to all .jsonl files in data_dir
    data_files = []
    for files in os.listdir(data_dir):
        if files.endswith(".jsonl"):
            data_files.append(os.path.join(data_dir, files))

    # Call training method from src.training.train_tokenizer
    train_bpe_tokenizer(data_files, output_path=tokenizer_path, vocab_size=vocab_size)
    print(f"[END] Done training tokenizer and stored at '{tokenizer_path}'...")
