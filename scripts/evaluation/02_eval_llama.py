from transformers import logging as hf_logging
from src.models.Llama import perform_inference_with_checkpoint
import argparse
import os
from src.utils.utils import load_json_to_dict_list
import pandas as pd

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


def main():
    hf_logging.set_verbosity_warning()

    # 1. Step: ───── Load Arguments ─────
    args = parse_args()
    checkpoint_path = args.set_checkpoint
    eval_data_path = args.set_evaldata
    output_dir = args.set_outputdir
    print(f"[yellow] -- [INFO] Model and Tokenizer loaded successfully! [/yellow]")

    # 2. Step: ───── Load evaluation data ─────
    if not os.path.exists(eval_data_path):
        raise RuntimeError(f"[ERROR] Evaluation of Llama model stopped. Given path '{eval_data_path}' is not "
                           f"a valid .jsonl for evaluation.")
    evaluation_data = load_json_to_dict_list(eval_data_path)

    # 3. Step: ───── Perform inference run over eval data ─────
    tokenizer_args = {
        "tokenizer_path": "tokenizer/bpe_tokenizer.json",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>"
    }
    print(f"-- [INFO] Calling inference run with tokenizer from {tokenizer_args['tokenizer_path']}")
    results = perform_inference_with_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_args=tokenizer_args,
        inputs=evaluation_data
    )
    print(f"[yellow] -- [INFO] Model was loaded from '{checkpoint_path}' [/yellow]")

    # 4. Step: ───── Store results to specified output dir ─────
    pf = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "autoregressive_results.csv")
    pf.to_csv(output_path)
    print(f"[yellow] -- [INFO] Successfully stored inference results to '{output_path}' [/yellow]")