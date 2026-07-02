import argparse
import os

import numpy as np
from conllu import parse
from transformers import logging as hf_logging
from src.utils.utils import load_yaml_config
from src.models.RoBERTa import get_roberta_for_token_classification, freeze_base
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from rich.console import Console
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import wandb

def parse_args() -> argparse.Namespace:
    """ Parses roberta specific arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-checkpoint", type=str)
    parser.add_argument("--freeze-base", type=str)

    args, unknown_args = parser.parse_known_args()
    if not args.set_checkpoint:
        raise FileNotFoundError(f"[yellow]>> Invalid or no checkpoint path is specified! [/yellow]")

    return args

def compute_metrics(eval_preds):
    """ Computes accuracy for POS tagging, ignoring -100 padded tokens """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_preds = [p for sublist in true_predictions for p in sublist]
    flat_labels = [l for sublist in true_labels for l in sublist]

    accuracy = sum(p == l for p, l in zip(flat_preds, flat_labels)) / len(flat_labels)
    return {"accuracy": accuracy}

def main():
    console = Console()
    hf_logging.set_verbosity_warning()

    # 1. Step: ───── Parse args ─────
    args = parse_args()

    # 2. Step: ───── Load model config file ─────
    config_path = "configs/roberta_pos_full_ft.yaml" if args.freeze_base == "true" else "configs/roberta_pos_freezed.yaml"
    model_config = load_yaml_config(config_path)
    print(f"[yellow] -- [INFO] Model config successfully loaded from '{config_path}'")

    # 3. Step: ───── Ensure data from Universal Dependency Datasets exists, otherwise download it ─────
    data_path = model_config["data"]["path"]
    if not os.path.exists(data_path):
        print(f"[yellow] -- [INFO] Data for POS-Tagging does not exists at '{data_path}'")
        if not os.path.exists("./data/ud"):
            print(f"[red] -- [ERROR] Please download the .conllu files of the UD GitHub repo and store them at './data/ud'.. [/red]")
            urls = {
                "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu",
                "validation": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-dev.conllu",
                "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-test.conllu"
            }
            print(f"-- [red] {urls} [/red]")
        try:
            data_dict = {}
            all_labels = set()
            paths = {
                "train": "data/ud/la_ittb-ud-train.conllu",
                "validation": "data/ud/la_ittb-ud-dev.conllu",
                "test": "data/ud/la_ittb-ud-test.conllu"
            }
            for split, path in paths.items():
                print(f"-- [INFO] Processing {split}-split from '{path}'...")
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = f.read()
                parsed_data = parse(raw_data)
                tokens_list = []
                upos_list = []
                for sentence in parsed_data:
                    tokens = [token["form"] for token in sentence]
                    upos = [token["upos"] if token["upos"] else "X" for token in sentence]
                    tokens_list.append(tokens)
                    upos_list.append(upos)
                    all_labels.update(upos)
                data_dict[split] = Dataset.from_dict({"tokens": tokens_list, "upos": upos_list})

            dataset = DatasetDict(data_dict)
            label_list = sorted(list(all_labels))
            dataset.save_to_disk(data_path)
            print(f"[yellow] -- [INFO] Successfully downloaded data for POS tagging, stored at '{data_path}'")
        except BaseException as e:
            print(f"[red] -- [ERROR] Downloading failed. Exception: {e}")

    else:
        dataset = load_from_disk(data_path)
        all_labels = set()
        for tags in dataset["train"]["upos"]:
            all_labels.update(tags)
        label_list = sorted(list(all_labels))
        print(f"[yellow] -- [INFO] Data for POS-Training is taken from '{data_path}'")

    # 4. Step: ───── Initialize the model from given checkpoint ─────
    model, id2label, label2id = get_roberta_for_token_classification(args.set_checkpoint, label_list)
    print(f"[yellow] -- [INFO] Model successfully loaded from '{args.set_checkpoint}'")
    tokenizer = AutoTokenizer.from_pretrained(args.set_checkpoint, use_fast=True)
    print(f"[yellow] -- [INFO] Tokenizer successfully loaded from '{args.set_checkpoint}'")
    if args.freeze_base == "true":
        freeze_base(model, classifier_name="classifier")
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[yellow] -- [INFO] Number of trainable parameters: {num_trainable_params}")

    # 5. Step:  ───── Perform label alignment ─────
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, max_length=256
        )
        labels = []
        for i, label_seq in enumerate(examples["upos"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    tag_str = label_seq[word_idx]
                    label_ids.append(label2id[tag_str])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    print(f"[yellow] -- [INFO] Label alignment performed successfully")

    # 5. Step:  ───── Preparing data ─────
    training_config = model_config["train"]
    training_args = TrainingArguments(**training_config)

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    print(f"[yellow] -- [INFO] Initialized Trainer successfully")

    meta_config = model_config["meta"]
    report_to = meta_config["report_to"]
    if report_to is not None:
        if report_to == "wandb":
            wandb.init(
                project=meta_config["project_name"],
                name=meta_config["run_name"],
                tags=meta_config["tags"],
                group=meta_config["group"],
            )
            training_config["report_to"] = meta_config["report_to"]
            training_config["run_name"] = meta_config["run_name"]
            print(f"[yellow] -- [INFO] Training procedure will be reported to '{report_to}'!")

    print(f"[yellow] -- [INFO] Starting train procedure...")
    trainer.train()

    if report_to is not None:
        if report_to == "wandb":
            wandb.finish()

if __name__ == "__main__":
    main()