import src.config as config
from src.data.preprocessing import process_file, init_worker, filter_with_lingua
import os


"""
    This scripts orchestrates the complete preprocessing pipeline, including
    - normalization
    - filtering
"""

if __name__ == "__main__":

    input_dir = config.RAW_DATA_DIR
    output_dir = config.PROCESSED_DATA_DIR

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Extracts paths to all .jsonl files in raw data
    data_files = []
    for files in os.listdir(input_dir):
        if files.endswith(".jsonl"):
            data_files.append(os.path.join(input_dir, files))

    # Process each .jsonl individually
    #   For adjustments, modify the 'process_file' function in src.data.preprocessing
    overall_words = 0
    for data_file in data_files:
        file_name = data_file.split("/")[-1].removesuffix(".jsonl")
        #output_path = os.path.join(output_dir, file_name + "_clean.jsonl")
        cnt_filtered_words = process_file(data_file, output_dir)
        overall_words += cnt_filtered_words

    print(f"[END] Done with Pre-Processing all dater! Overall words: {overall_words}")




