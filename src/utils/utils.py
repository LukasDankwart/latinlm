import os.path
import random

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from typing import List
import json

""" This file includes arbitary util functions """

def load_yaml_config(yaml_path: str) -> dict:
    """ Load given yaml file as dict """
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json_data(json_path: str) -> List[str]:
    """ Load json file given by path and returns list of all text examples """
    if not os.path.exists(json_path):
        raise RuntimeError(f"[ERROR] Given path '{json_path}' to json data file is invalid.")
    texts = []
    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row_dict = json.loads(line.strip())
                texts.append(row_dict["text"])
    return texts


def load_json_to_dict_list(json_path: str) -> list[dict]:
    data = []
    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            dict_line = json.loads(line)
            data.append(dict_line)
    return data


def count_unique_sources(json_path: str) -> dict:
    data = load_json_to_dict_list(json_path)
    results = {}
    for sample in data:
        source = sample.get("source")
        if source not in results.keys():
            results.update({source: 1.0})
        else:
            results[str(source)] += 1.0
    return results

def count_sources_word_quantities(json_path: str) -> dict:
    data = load_json_to_dict_list(json_path)
    results = {}
    for sample in data:
        source = sample.get("source")
        length = len(sample.get("text").split())
        if source not in results.keys():
            results.update({source: length})
        else:
            results[str(source)] += length
    return results

def create_data_split_ratio(original_quantities: dict, eval_quantities: dict):
    overview = {}
    for key, val in original_quantities.items():
        overview.update({
            str(key): {
                "base_quantitiy": original_quantities[key]["count"],
                "eval_quantitiy": eval_quantities[key]["count"],
                "eval_ratio": eval_quantities[key]["count"] / original_quantities[key]["count"]
            }
        })
    df = pd.DataFrame(overview)
    df.to_csv("dataset_split.csv")

def extract_autoregressive_subset(path_to_csv: str, subset_cap : int = 3000):
    """ Extracts a specified subset of length subset_cap from autoregressive results """
    df = pd.read_csv(path_to_csv)
    results = df.to_dict(orient='records')
    # Due to limited data of these source, every eval sample of these is taken into the subset
    # The rest is filled with samples from more frequent sources
    accepted_source = [
        "wikimedia_wikipedia_20231101.la_clean",
        "nuntii_latini_clean",
        "canonical_latin_clean",
        "vatican_news_clean",
        "cltk_data_clean"
    ]
    subset = []
    subset_fill = []
    for sample in results:
        source = sample.get("source")
        if source in accepted_source:
            subset.append(sample)
            continue
        # Otherwise, source is from Fineweb2 or GreLa, and we fill later
        subset_fill.append(sample)
    missing_quantitiy = subset_cap - len(subset)
    if missing_quantitiy > 0:
        random.seed(42)
        selected_elements = random.choices(subset_fill, k=missing_quantitiy)
        subset.extend(selected_elements)
    return subset


def pie_chart_of_sources(quantities: dict, output_path: str, title: str):
    """ Creates a Pie Chart of given dictionary, that is expected to have <source_name: quantity> format"""
    values = quantities.values()
    labels = quantities.keys()
    legend_labels = [f"{label} (n = {val / 1_000_000:.2f} M)" for label, val in zip(labels, values)]

    def my_autopct(pct):
        return ('%1.1f%%' % pct) if pct > 2 else ''

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(
        values,
        labels=None,
        autopct=my_autopct,
        startangle=140,
        colors=plt.cm.Set3.colors,
        wedgeprops={'edgecolor': 'white'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(
        wedges,
        legend_labels,
        title="Amount of training words per datasource",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title_fontsize=14,  # Macht den Legenden-Titel größer
        prop={'weight': 'bold', 'size': 12}  # Macht die Legenden-Einträge fett und größer
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    """path = "data/llama/eval/eval_data.json"
    results = count_unique_sources(path)
    print(results)
    pie_chart_of_sources(results, "eval_data_pie_chart.svg")"""

    """subset = extract_autoregressive_subset("experiments/llama_run_1/evaluation_results.csv")
    subset_df = pd.DataFrame(subset)
    subset_df.to_csv("experiments/llama_run_1/evaluation_results_subset.csv")
    results = {}
    for sample in subset:
        source = sample.get("source")
        if source not in results.keys():
            results.update({source: 1.0})
        else:
            results[str(source)] += 1.0
    pie_chart_of_sources(results, "autoregressive_subset_pie_chart.svg")"""

    """df = pd.read_csv("experiments/llama_run_1/judgement_results_incremental.csv")
    df = df.to_dict(orient="records")

    print(len(df))
    print(df[-1].get("guess_generated_start_word"))"""

    train_data_dist = count_sources_word_quantities("data/deduplicated/deduplicated_corpus.jsonl")
    pie_chart_of_sources(train_data_dist, output_path="train_data_distribution.svg", title="Training Corpus Distribution")





