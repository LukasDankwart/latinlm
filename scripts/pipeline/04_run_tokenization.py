import os
from transformers import PreTrainedTokenizerFast
from src.utils.utils import load_yaml_config
from src.data.binarize import prepare_training_data, create_dataset
import traceback


def main():
    # Instantiate tokenizer

    print(f"-- [INFO] TEEEEEEEEEEEEEST'", flush=True)
    try:
        dataset_config_path = "configs/dataset.yaml"
        if not os.path.isfile(dataset_config_path):
            raise FileNotFoundError(f"[ERROR] There is no dataset config file at '{dataset_config_path}'!")
        dataset_config = load_yaml_config(dataset_config_path)
        print(f"-- [INFO] Dataset config loaded successfully from '{dataset_config_path}'", flush=True)

        tokenizer_path = dataset_config["tokenizer_path"]
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"[ERROR] There is no tokenizer at '{tokenizer_path}'! Check dataset.yaml!")

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    except Exception as e:
        print(f"[ERROR] Couldn't initialize tokenizer for tokenization and binarization step.", flush=True)
        traceback.print_exc()

    print(f"-- [INFO] Tokenizer initialized successfully")

    input_dir = dataset_config["input_dir_path"]
    output_dir = dataset_config["output_dir_path"]

    # 1. Create summarzied dataset-dict with train, test, eval split by arguments from dataset.yaml
    print(f"[START] Start creating dataset dictionary...")
    dataset_dict = create_dataset(dataset_config)

    # 2. Pre-tokenize whole dataset
    print(f"[START] Start pre-tokenizing dataset...")
    prepare_training_data(dataset_dict, output_dir, tokenizer=tokenizer)
    print(f"[END] Done tokenizing dataset and stored to '{output_dir}'!")

if __name__ == "__main__":
    print(f"--hallo")
    main()