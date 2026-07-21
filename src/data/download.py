import os
import json

from bs4 import BeautifulSoup
from datasets import load_dataset
import src.config as config
from pathlib import Path
import duckdb
import traceback
from cltk.data.fetch import FetchCorpus
from git import Repo

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


def load_cltk_datasets(corpora_to_download: list[str], output_dir: str = "data/raw/"):
    corpus_downloader = FetchCorpus(language="lat")
    print(f"-- [INFO] Number of Corpora from CLTK: {corpus_downloader.list_corpora}")

    for corpus in corpora_to_download:
        print(f"-- [INFO] Downloading {corpus}")
        corpus_downloader.import_corpus(corpus)

    cltk_base_dir = os.path.expanduser("~/cltk_data/lat/text")
    output_path = os.path.join(output_dir, "cltk_data.jsonl")
    if os.path.isfile(output_path):
        Path(output_path).unlink()
    processed_files = 0
    with open(output_path, "a", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(cltk_base_dir):
            for file in files:
                if file.endswith((".txt", ".tess")) and not file.startswith("."):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            content = infile.read().strip()
                            if len(content) > 50:
                                corpus_name = os.path.basename(os.path.dirname(root))
                                author_or_folder = os.path.basename(root)
                                data = {
                                    "text": content,
                                    "meta": {
                                        "corpus": corpus_name,
                                        "author_folder": author_or_folder,
                                        "filename": file
                                    }
                                }
                                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                                processed_files += 1

                    except Exception as e:
                        print(f"[ERROR] Couldn't read  {file}: {e}")


def download_canonical_latin(repo_path: str, output_dir: str = "data/raw"):
    """ Aggregates all texts from repo clone of canonical latin """
    # If path does not exist, the repository is cloned at the specified path
    if not os.path.isdir(repo_path):
        repo_url = "https://github.com/PerseusDL/canonical-latinLit.git"
        target_dir = os.path.join(output_dir, "canonical-latinLit")
        Repo.clone_from(repo_url, target_dir)
        print(f"-- [INFO] Start cloning repository at '{output_dir}'...")
    print(f"-- [INFO] Repository exsits at '{repo_path}'")

    output_path = os.path.join(output_dir, "canonical_latin.jsonl")
    if os.path.isfile(output_path):
        Path(output_path).unlink()

    with open(output_path, "a", encoding="utf-8") as out:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".xml") and not file.startswith("__"):
                    filepath = os.path.join(root, file)
                    clean_text = extract_text_from_tei(filepath)
                    if clean_text and len(clean_text) > 50:
                        json_line = json.dumps({"text": clean_text, "source": file})
                        out.write(json_line + "\n")


def extract_text_from_tei(xml_filepath):
    """ Extracts text from given XML file """
    try:
        with open(xml_filepath, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "lxml-xml")
            body = soup.find("body")
            if not body:
                return None
            raw_text = body.get_text(separator=" ", strip=True)
            return raw_text
    except Exception as e:
        print(f"[ERROR] At {xml_filepath}: {e}")
        return None


if __name__ == "__main__":
    repo_path = "data/raw/canonical-latinLit"
    output_dir = "data/raw/"
    download_canonical_latin(repo_path, output_dir)




