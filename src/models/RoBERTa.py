import os
import yaml
from src.utils.utils import load_yaml_config
from transformers import RobertaConfig, TrainingArguments, RobertaForMaskedLM, RobertaTokenizer

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)


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


def get_roberta_for_token_classification(model_checkpoint: str, label_list: list) -> tuple[AutoModelForTokenClassification, dict, dict]:
    """ Does load a given checkpoint as model for token classification (e.g. POS learning)"""
    if not os.path.exists(model_checkpoint):
        raise RuntimeError(f"-- [ERROR] Given checkpoint path '{model_checkpoint}' is invalid.")
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    return model, id2label, label2id


def freeze_base(model: AutoModelForTokenClassification, classifier_name="classifier") -> None:
    """ Freezes weights in the backbone of given model. Only classification-head parameters are learnable """
    for name, param in model.named_parameters():
        if classifier_name not in name:
            param.requires_grad = False