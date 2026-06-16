import yaml
""" This file includes arbitary util functions """

def load_yaml_config(yaml_path: str) -> dict:
    """ Load given yaml file as dict """
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)