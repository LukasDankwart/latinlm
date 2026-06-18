import os

import yaml
from src.utils.utils import load_yaml_config
from transformers import RobertaConfig, TrainingArguments, RobertaForMaskedLM, RobertaTokenizer


def initialize_roberta(config_path: str) -> RobertaForMaskedLM:
    """ This function initializes RobertaForMaskedLM with given config_path"""

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"[ERROR] Given path '{config_path}' is not a valid yaml confi!")
    config = load_yaml_config(config_path)

    model_args = config["model"]
    tokenizer_args = config["tokenizer"]

    tokenizer = RobertaTokenizer.from_pretrained(
        tokenizer_args["tokenizer_path"],
        unk_token=["unk"],
        mask_token=["<mask>"],
        pad_token=["<pad>"],
        bos_token=["<s>"],
        eos_token=["</s>"],
    )
    model_args["vocab_size"] = tokenizer.vocab_size

    model_config = RobertaConfig(**model_args)
    model = RobertaForMaskedLM(config=model_config)
    return model