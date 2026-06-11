import src.config as config
from src.data.download import stream_and_sample_dataset
import re

""" 
    This scripts specifies all steps for the 'Download' step, meaning, which datasets are streamed and saved.
"""

if __name__ == "__main__":
    print(f"[START] Downloading datasets specified in config.DATASETS_TO_DOWNLOAD...")

    num_samples = config.DATASETS_SAMPLE_LIMIT
    for dataset in config.DATASETS_TO_DOWNLOAD:
        name = dataset.get("name")
        subset = dataset.get("subset")
        output_filename = f"{name}_{subset}.jsonl" if subset is not None else f"{name}.jsonl"
        output_filename = re.sub('/', '_', output_filename)
        stream_and_sample_dataset(name, subset, output_filename, num_samples)

    print(f"[END] Downloading datasets specified in config.DATASETS_TO_DOWNLOAD! \n")