import os

import src.config as config
from src.data.deduplication import deduplicate_corpus_global

""" This script runs the deduplication procedure from src.data.deduplication over the cleaned data corpus """

if __name__=="__main__":

    # Configure paths
    input_dir = config.PROCESSED_DATA_DIR
    output_dir = config.DEDUPLICATED_DATA_DIR
    output_path = os.path.join(output_dir, "deduplicated_corpus.jsonl")

    # Calling deduplication procedure for whole PROCESSED_DATA_DIR and store it to 'output_path'
    deduplicate_corpus_global(input_dir, output_path, threshold=0.8)


