from datasets import load_dataset, DatasetDict
import os
from transformers import PreTrainedTokenizerFast
import src.config as config
from glob import glob

def prepare_training_data(
        dataset_dict: DatasetDict,
        output_path: str, tokenizer:
        PreTrainedTokenizerFast = None,
        block_size: int = 512
):
    print(f"[START] Loading tokenizer...")
    if tokenizer is None:
        raise RuntimeError("[ERROR] Pre-tokenization requires tokenizer as argument!")

    def tokenize_function(sample):
        return tokenizer(sample["text"], return_special_tokens_mask=True)

    print(f"-- [INFO] Tokenizing samples...")
    original_columns = dataset_dict["train"].column_names
    tokenized_dataset = dataset_dict.map(
        tokenize_function,
        batched=True,
        num_proc=config.NUM_CPU_WORKERS,
        remove_columns=original_columns,
        load_from_cache_file=False
    )

    block_size = block_size
    # e.g. in Masked LM we dont want to input very short sentences (fewer tokens!), so we group them up to a fixed sice
    # (basically the sequence length)
    print(f"--  [INFO] Grouping samples...")
    def texts_to_group(samples):
        concatenated_samples = {k: sum(samples[k], []) for k in samples.keys()}
        total_length = len(concatenated_samples[list(samples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_samples.items()
        }
        return result

    lm_dataset = tokenized_dataset.map(
        texts_to_group,
        batched=True,
        batch_size=1000,
        num_proc=config.NUM_CPU_WORKERS,
        load_from_cache_file=False
    )

    print(f"-- [INFO] Storing tokenized dataset...")
    lm_dataset.save_to_disk(output_path)
    print(f"[END] Stored tokenized dataset to '{output_path}'")


def create_dataset(config: dict) -> DatasetDict:
    data_path = config["input_dir_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] '{data_path}' from given config does not exist!")
    all_files = glob(os.path.join(data_path, "*.jsonl"))

    print(f"[START] Creating summarized dataset from '{data_path}'...")
    raw_dataset = load_dataset("json", data_files=all_files, split="train")
    print(f"-- [INFO] Overall number of samples: {len(raw_dataset)}")

    test_size = config["test_size"]
    eval_size = config["eval_size"]
    seed = config["seed"]

    # Split test set
    test_split = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    test_set = test_split["test"]
    remaining_data =  test_split["train"]

    # Split eval stet
    eval_split = remaining_data.train_test_split(test_size=eval_size, seed=seed)
    train_set = eval_split["train"]
    eval_set = eval_split["test"]

    final_dataset = DatasetDict({
        "train": train_set,
        "test": test_set,
    })

    eval_set_path = os.path.join(config["output_dir_path"], "eval")
    if not os.path.isdir(eval_set_path):
        os.makedirs(eval_set_path)
    eval_set.to_json(eval_set_path + "/eval_data.json", force_ascii=False)

    print(f"[END] Done creating summarized datset dictionary!")
    print(f"-- [INFO] Train: {len(train_set)}")
    print(f"-- [INFO] Test: {len(test_set)}")
    print(f"-- [INFO] Eval: {len(eval_set)}")
    print(f"-- [INFO] Eval data has been excluded from DatasetDict and stored at '{eval_set_path}'")
    return final_dataset






