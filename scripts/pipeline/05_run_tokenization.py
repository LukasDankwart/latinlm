from src.utils.utils import load_yaml_config
from src.data.binarize import prepare_training_data, create_dataset
from tokenizers.processors import TemplateProcessing
import os
from transformers import PreTrainedTokenizerFast
import traceback


def main():
    # Instantiate tokenizer
    try:
        dataset_config_path = "configs/dataset.yaml"
        if not os.path.isfile(dataset_config_path):
            raise FileNotFoundError(f"[ERROR] There is no dataset config file at '{dataset_config_path}'!")
        dataset_config = load_yaml_config(dataset_config_path)
        print(f"-- [INFO] Dataset config loaded successfully from '{dataset_config_path}'", flush=True)

        tokenizer_path = dataset_config["tokenizer_path"]

        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"[ERROR] There is no tokenizer at '{tokenizer_path}'! Check dataset.yaml!")


        tokenizer_args = dataset_config["tokenizer"]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            pad_token=tokenizer_args["pad_token"],
            bos_token=tokenizer_args["bos_token"],
            eos_token=tokenizer_args["eos_token"],
            unk_token=tokenizer_args["unk_token"],
            mask_token=tokenizer_args["mask_token"],
            sep_token=tokenizer_args["sep_token"],
            cls_token=tokenizer_args["cls_token"]
        )
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
            pair=f"{tokenizer.bos_token} $A {tokenizer.eos_token} {tokenizer.eos_token} $B {tokenizer.eos_token}",
            special_tokens=[
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.eos_token, tokenizer.eos_token_id),
            ],
        )
    except Exception as e:
        print(f"[ERROR] Couldn't initialize tokenizer for tokenization and binarization step.", flush=True)
        traceback.print_exc()

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
    main()