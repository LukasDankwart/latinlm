import os.path

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
            results[str(source)] = {"count": 1.0}
        else:
            results[str(source)]["count"] += 1.0
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

