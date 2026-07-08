import os

import src.config as config
from src.data.download import stream_and_sample_dataset, filter_grela, load_cltk_datasets, download_canonical_latin
import re

""" 
    This scripts specifies all steps for the 'Download' step, meaning, which datasets are streamed and saved.
"""

if __name__ == "__main__":
    print(f"[START] Downloading datasets specified in config.DATASETS_TO_DOWNLOAD...")

    output_dir = config.RAW_DATA_DIR

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    num_samples = config.DATASETS_SAMPLE_LIMIT
    for dataset in config.DATASETS_TO_DOWNLOAD:
        name = dataset.get("name")

        # For cltk, a different method is used
        if name == "cltk":
            subsets = dataset.get("subset")
            load_cltk_datasets(subsets, output_dir)
            continue

        subset = dataset.get("subset")
        output_filename = f"{name}_{subset}.jsonl" if subset is not None else f"{name}.jsonl"
        output_filename = re.sub('/', '_', output_filename)
        stream_and_sample_dataset(name, subset, output_filename, num_samples)

    print(f"[END] Downloading datasets specified in config.DATASETS_TO_DOWNLOAD! \n")

    print(f"[START] Processing pre-downloaded datasets...")
    preloaded_datasets = config.PRELOADED_DATASETS
    for dataset in preloaded_datasets:
        name = dataset["name"]
        path = dataset["path"]

        if not os.path.exists(path):
            print(f"-- [INFO] Path for dataset '{name}' does not exist at specified path! '{name}' will be skipped!")
            print(f"-- [INFO] Pre-downloaded datasets need to be manually downloaded before running the datapipeline")
            print(f"-- [INFO] Please have a look at the README.md to check if all steps are fulfilled! \n")

        print(f"\n [START] progressing '{name}'...")

        # Call individual handler for each predownloaded dataset
        if name == "GreLa":
            filter_grela(path_to_db=path)

        if name == "canonical-latinLit":
            download_canonical_latin(repo_path=path, output_dir=output_dir)

        print(f"[END] Done processing '{name}'")
