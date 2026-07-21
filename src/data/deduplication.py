import os
import src.config as config
from datasketch import MinHash, MinHashLSH
import json
import multiprocessing as mp
from pathlib import Path

def compute_minhash_for_line(item: tuple[str, str]) -> tuple[dict, MinHash]:
    """ Computes the MinHash for each given sample consisting of 'source_name' and the samples 'text'. This function
        is executed in parallel by the Multiprocessor. """
    source_name, line = item
    if not line.strip():
        return None

    data = json.loads(line)
    text = data.get("text", "")

    m = MinHash(num_perm=128)
    for word in set(text.lower().split()):
        m.update(word.encode('utf8'))

    if "source" not in data:
        data["source"] = source_name
    return data, m

def yield_lines_from_folder(folder_path: str):
    """ Generator function that yields lines from all *.jsonl files in specified folder path """
    input_dir = Path(folder_path)

    for filepath in input_dir.glob("*.jsonl"):
        print(f"-- [INFO] Loading {filepath.name} for processing...")
        with open(filepath, "r", encoding="utf-8") as infile:
            for line in infile:
                yield filepath.stem, line

def deduplicate_corpus_global(input_folder: str, output_path: str, threshold: float = 0.8):
    """ Deduplicates all cleaned files in given directory """
    input_dir = Path(input_folder)

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    duplicates_found = 0
    saved_records = 0

    num_cpu_workers = config.NUM_CPU_WORKERS

    print(f"-- [START] Deduplication procedure is started for files in '{input_folder}'...")
    with open(output_path, "w", encoding="utf-8") as outfile:

        with mp.Pool(processes=num_cpu_workers) as pool:
            line_generator = yield_lines_from_folder(input_folder)

            results_iterator = pool.imap(
                compute_minhash_for_line,
                line_generator,
                chunksize=2000
            )

            for idx, result in enumerate(results_iterator):
                if result is None:
                    continue

                data, minhash = result

                if not lsh.query(minhash):
                    lsh.insert(f"doc_{idx}", minhash)
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    saved_records += 1
                else:
                    duplicates_found += 1

                if idx > 0 and idx % 25000 == 0:
                    print(f"-- [INFO] Globally processed: {idx} | Removed duplicates: {duplicates_found} | Saved: {saved_records}")

    print(f"-- [INFO] Deduplication finished. Number of duplicated texts: {duplicates_found}")
    print(f"-- [END] Final training samples: {saved_records}")
