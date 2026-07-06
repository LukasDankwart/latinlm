import os
import json
from datasets import load_dataset
import src.config as config
from pathlib import Path
import duckdb
import traceback

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

            if (i + 1) % 100000 == 0:
                print(f"-- {i + 1} Samples have been loaded...")

    print(f"[END] Successfully stored {dataset_name} data to '{output_filename}'!")


def filter_grela(path_to_db: str, output_dir: str = "data/raw") -> None:
    """ Filters local duck.db file containing all date from
    https://zenodo.org/records/17866143?preview_file=CCS-ZCU%2FGreLa-v0.6.zip  """

    if not os.path.exists(path_to_db):
        raise RuntimeError(f"[ERROR] File for 'GreLa' database does not exists at given path: {path_to_db}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "/grela.jsonl"

    try:
        print(f"[START] Connecting to local GreLa database at '{path_to_db}'")
        conn = duckdb.connect(path_to_db, read_only=True)

        query = """
            SELECT 
                s.sent_text AS text, 
                w.author, 
                w.title, 
                w.grela_source AS corpus
            FROM sentences s
            JOIN works w ON s.grela_id = w.grela_id
            WHERE w.grela_source IN ('cc', 'noscemus', 'emlap', 'vulgate')
        """

        result = conn.execute(query)

        if os.path.exists(output_file):
            os.remove(output_file)

        chunk_count = 0
        total_rows = 0
        print(f"-- [INFO] Start writing at '{output_file}'...")
        while True:
            df_chunk = result.fetch_df_chunk()

            if df_chunk.empty:
                break

            df_chunk = df_chunk.dropna(subset=['text'])
            df_chunk = df_chunk[df_chunk['text'].str.strip() != '']

            df_chunk.to_json(output_file, orient='records', lines=True, force_ascii=False, mode='a')

            total_rows += len(df_chunk)
            chunk_count += 1
            if chunk_count % 1000 == 0:
                print(f" -- [INFO] {chunk_count} chunks processed... (Rows {total_rows:,} saved)")

        print(f"[END] Done extracting Latin corpus from GreLa. Stored at: '{output_file}'")

    except Exception as e:
        print(f"\n[ERROR] Error occurred while connecting to database: {e}")
        print(traceback.format_exc())

    finally:
        if 'conn' in locals():
            conn.close()




