from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import src.config as config

def prepare_training_data(jsonl_path: str, output_path: str, tokenizer: PreTrainedTokenizerFast = None):
    print(f"[START] Loading tokenizer...")
    if tokenizer is None:
        raise RuntimeError("[ERROR] Pre-tokenization requires tokenizer as argument!")

    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def tokenize_function(sample):
        return tokenizer(sample["text"], return_special_tokens_mask=True)

    print(f"-- [INFO] Tokenizing samples...")
    original_columns = dataset.column_names
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.NUM_CPU_WORKERS,
        remove_columns=original_columns,
    )

    block_size = 512
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
    )

    print(f"-- [INFO] Storing tokenized dataset...")
    lm_dataset.save_to_disk(output_path)
    print(f"[END] Stored tokenized dataset to '{output_path}'")



