import os

from datasketch import MinHash, MinHashLSH
import json
from pathlib import Path

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """ Computes MinHash for given text after normalization & filtering step """
    m = MinHash(num_perm=num_perm)
    for word in set(text.lower().split()):
        m.update(word.encode('utf8'))
    return m

def deduplicate_corpus_global(input_folder: str, output_path: str, threshold: float = 0.8):
    """ Deduplicates all cleaned files in given directory """
    input_dir = Path(input_folder)

    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    duplicates_found = 0
    saved_records = 0
    global_idx = 0

    # Delete file if already existing
    if os.path.exists(output_path):
        Path(output_path).unlink()

    print(f"-- [START] Deduplication procedure is started for files in '{input_folder}'...")
    with open(output_path, "a", encoding="utf-8") as outfile:

        # Deduplicate each file in "data/processed"
        for filepath in input_dir.glob("*.jsonl"):
            print(f"-- [INFO] Processing file {filepath.name}")

            with open(filepath, "r", encoding="utf-8") as infile:
                for line in infile:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    text = data.get("text", "")

                    minhash = get_minhash(text)
                    result=  lsh.query(minhash)

                    if not result:
                        if "source" not in data:
                            data["source"] = filepath.stem

                        lsh.insert(f"doc_{global_idx}", minhash)
                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                        saved_records += 1
                    else:
                        duplicates_found +=1

                    global_idx += 1

    print(f"-- [INFO] Deduplication finished. Number of duplicated texts: {duplicates_found}")
    print(f"-- [END] Final training samples: {saved_records}")
