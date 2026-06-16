import os
from transformers import PreTrainedTokenizerFast
import src.config as config
from src.data.binarize import prepare_training_data

if __name__ == "__main__":

    # Instantiate tokenizer
    tokenizer_path = config.TOKENIZER_PATH
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    input_dir = config.PROCESSED_DATA_DIR
    output_dir = config.TOKENIZED_DATA_DIR

    data_files = []
    for files in os.listdir(input_dir):
        if files.endswith(".jsonl"):
            data_files.append(os.path.join(input_dir, files))

    # For each file, call the tokenization and store final results under "data/tokenized"
    print(f"[START] Start preparing pre-processed data files...")
    for data_file in data_files:
        file_name = data_file.split("/")[-1].split(".")[0]
        target_path = os.path.join(output_dir, file_name)
        prepare_training_data(data_file, target_path, tokenizer=tokenizer)
        print(f"-- [INFO] Stored pre-tokenized '{target_path}'")
    print(f"[END] Finished pre-tokenization of all pre-processed files!")