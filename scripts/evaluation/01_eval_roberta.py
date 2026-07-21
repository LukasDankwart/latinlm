import argparse
import json
import sys
from src.utils.script_utils import load_model_and_tokenizer
from src.utils.utils import load_json_data
import src.tests.roberta_metrics as roberta_metrics
import pandas as pd
import os
import logging
from transformers import logging as hf_logging
from datasets import load_from_disk, load_dataset, Dataset

from transformers import PreTrainedModel, PreTrainedTokenizerFast

def parse_args() -> argparse.Namespace:
    """ Parses roberta specific arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-checkpoint", type=str)
    parser.add_argument("--set-evaldata", type=str)
    parser.add_argument("--set-outputdir", type=str)

    args, unknown_args = parser.parse_known_args()
    if not args.set_checkpoint:
        raise FileNotFoundError(f"[yellow]>> Invalid or no checkpoint path is specified! [/yellow]")

    path_of_model_run = args.set_checkpoint.split("/")[:-2]

    # Fallback to standard eval data file, if no other is specified
    if not args.set_evaldata:
        args.set_evaldata = "data/tokenized/eval/eval_data.json"
    if not args.set_outputdir:
        args.set_outputdir = os.path.join(*path_of_model_run) + "/evaluation"

    return args


def load_eval_safe(filepath: str, limit: int = 50000) -> Dataset:
    """ Does load a subset of specified .jsonl file for evaluation """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def main():

    hf_logging.set_verbosity_warning()

    # 1. Step: ───── Load Arguments ─────
    args = parse_args()
    checkpoint_path = args.set_checkpoint
    eval_data_path = args.set_evaldata
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    print(f"[yellow] -- [INFO] Model and Tokenizer loaded successfully! [/yellow]")
    print(f"[yellow] -- [INFO] Model was loaded from '{checkpoint_path}' [/yellow]")
    print(f"[yellow] -- [INFO] Model vocab size: {model.config.vocab_size} [/yellow]")
    print(f"[yellow] -- [INFO] Tokenzier vocab size: {len(tokenizer)} [/yellow]")
    print(f"[yellow] -- [INFO] MaskTokenID: {tokenizer.mask_token_id} | Word: {tokenizer.mask_token} [/yellow]")

    # Process arguments for eval run
    output_dir = args.set_outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"[yellow] -- [INFO] Output directory: {output_dir} [/yellow]")

    checkpoint_name = args.set_checkpoint.split("/")[-1]


    # 2. Step: ───── Prepare Eval data ─────
    #dataset = load_dataset(
    #    "json",
    #    data_files=eval_data_path,
    #    split="train",
    #    streaming=True
    #)

    num_samples = 50000
    #texts = dataset.take(num_samples)
    dataset = load_eval_safe(eval_data_path, limit= num_samples)
    texts = dataset["text"]
    test_sentence = texts[0]
    tokens = tokenizer(
                test_sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_special_tokens_mask=True
            )
    print(test_sentence)
    print(tokens)

    #texts = load_json_data(eval_data_path)
    #ud_dataset = load_from_disk(eval_data_path)
    #texts = ud_dataset["test"]
    print(f"[yellow] -- [INFO] Evaldata loaded from '{eval_data_path}', number of samples: {num_samples} [/yellow]")


    # 3. Step: ───── Compute metrics ─────
    # Compute perplexity over eval data
    perplexity = roberta_metrics.perplexity_score(model, tokenizer, texts)
    print(f"[yellow] -- [INFO] Perplexity computed successfully! [/yellow]")

    # Compute top-1 and top-5 (k=5) accuracies
    k = 5
    top_accuracies = roberta_metrics.top_k_accuracies(model, tokenizer, texts, k)
    top_1_acc = top_accuracies["top_1_accuracy"]
    top_k_acc = top_accuracies[f"top_{k}_accuracy"]
    print(f"[yellow] -- [INFO] Top-{k} accuracies computed successfully! [/yellow]")

    # Compute embedding vectors
    embedding_results = roberta_metrics.extract_sentence_embeddings(model, tokenizer, texts)
    k_sims = 3
    top_k_similar_sentences = roberta_metrics.find_top_k_similar_sentences(embedding_results, k_sims)
    print(f"[yellow] -- [INFO] Top-{k_sims} similar sentences computed successfully! [/yellow]")

    ## 4. Step: ───── Store results ─────
    metric_results = {
        "perplexity": perplexity,
        "top_1_acc": top_1_acc,
        f"top_{k}_acc": top_k_acc
    }
    df_metrics = pd.DataFrame([metric_results])
    metrics_path = output_dir + "/" + checkpoint_name + "_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"[yellow] -- [INFO] Overall metrics stored at '{metrics_path}' ! [/yellow]")

    flat_similarity_data = []
    for item in top_k_similar_sentences:
        orig_sentence = item["original_sentence"]
        for sim in item["top_similar_sentences"]:
            flat_similarity_data.append({
                "orig_sentence": orig_sentence,
                "similar_sentence": sim["text"],
                "sim_score": sim["similarity_score"]
            })
    df_sim = pd.DataFrame(flat_similarity_data)
    sim_path = output_dir + "/" + checkpoint_name + "_similarities.csv"
    df_sim.to_csv(sim_path, index=False, encoding="utf-8")
    print(f"[yellow] -- [INFO] Similar sentences stored at '{sim_path}' ! [/yellow]")

if __name__ == "__main__":
    main()