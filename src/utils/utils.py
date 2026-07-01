import os.path

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
        data = json.loads(json_path)
        for sample in data:
            texts.append(sample["text"])
    return texts