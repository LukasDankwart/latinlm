import os
import json
from datasets import load_dataset
import src.config as config

def stream_and_sample_dataset(
        dataset_name: str,
        subset: str | None,
        output_filename:str,
        num_samples: int=1000,
        ) -> None:
    """ Streams specified dataset and stores to given output file name """

    print(f"[START] Building connection to Huggingface for '{dataset_name}'...")
    if subset is not None:
        dataset = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    if not os.path.isdir(config.RAW_DATA_DIR):
        os.makedirs(config.RAW_DATA_DIR)
    output_path = os.path.join(config.RAW_DATA_DIR, output_filename)
    print(f"-- Processing dataset '{subset}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

            if (i + 1) % 100 == 0:
                print(f"-- {i+1} Samples have been loaded...")
    print(f"[END] Successfully stored {dataset_name} data to '{output_filename}'!")

if __name__ == "__main__":
    stream_and_sample_dataset(
        dataset_name="PleIAs/Latin-PD",
        subset=None,
        output_filename="latin_pd.jsonl",
        num_samples=1000
    )
