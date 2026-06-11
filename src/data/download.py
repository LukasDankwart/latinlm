import os
import json
from datasets import load_dataset
import src.config as config
from pathlib import Path

def stream_and_sample_dataset(
        dataset_name: str,
        subset: str | None,
        output_filename:str,
        num_samples: int | None = None,
        ) -> None:
    """ Streams specified dataset and stores to given output file name """

    print(f"[START] Building connection to Huggingface for '{dataset_name}'...")
    if subset is not None:
        dataset = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    if not os.path.isdir(config.RAW_DATA_DIR):
        os.makedirs(config.RAW_DATA_DIR)

    output_path = Path(os.path.join(config.RAW_DATA_DIR, output_filename))

    if output_path.exists():
        output_path.unlink()

    print(f"-- Processing dataset name: {dataset_name}, subset: {subset}...")

    for i, sample in enumerate(dataset):
        if num_samples is not None:
            if i >= num_samples:
                break
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

            if (i + 1) % 100 == 0:
                print(f"-- {i + 1} Samples have been loaded...")

    print(f"[END] Successfully stored {dataset_name} data to '{output_filename}'!")

