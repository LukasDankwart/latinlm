from transformers import logging as hf_logging
from src.models.Llama import perform_autoregressive_completion
from src.tests.llamar_metrics import analyze_sentence_by_llm, initialize_deepseek_api
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
    parser.add_argument(f"--skip-inference", action="store_true")
    parser.add_argument(f"--skip-judgment", action="store_true")

    args, unknown_args = parser.parse_known_args()
    if not args.set_checkpoint:
        raise FileNotFoundError(f"[yellow]>> Invalid or no checkpoint path is specified! [/yellow]")

    path_of_model_run = args.set_checkpoint.split("/")[:-2]

    # Fallback to standard eval data file, if no other is specified
    if not args.set_evaldata:
        args.set_evaldata = "data/llama/eval/eval_data.json"
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
    perform_inference = not args.skip_inference
    if perform_inference:
        tokenizer_args = {
            "tokenizer_path": "tokenizer/bpe_tokenizer.json",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        }
        print(f"-- [INFO] Calling inference run with tokenizer from {tokenizer_args['tokenizer_path']}")
        # Function calls autoregressive completion for original subsentences
        results = perform_autoregressive_completion(
            checkpoint_path=checkpoint_path,
            tokenizer_args=tokenizer_args,
            inputs=evaluation_data
        )
        print(f"[yellow] -- [INFO] Model was loaded from '{checkpoint_path}' [/yellow]")

        # 4. Step: ───── Store results to specified output dir ─────
        df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, "autoregressive_results.csv")
        df.to_csv(output_path)
        print(f"[yellow] -- [INFO] Successfully stored inference results to '{output_path}' [/yellow]")
    else:
        print(f"[yellow] -- [INFO]>> Skipping Llamar inference.")


    # 5. Step: ───── Call LLM-as-a-Judge Procedure ─────
    perform_llm_judgment = not args.skip_judgment
    if perform_llm_judgment:
        llm_input_data_path = os.path.join(output_dir, "autoregressive_results.csv")
        if not os.path.exists(llm_input_data_path):
            raise RuntimeError(f"[ERROR] Aborting LLM judgment procedure! Autoregressive completed data is expected to "
                  f"be at '{llm_input_data_path}' but file is missing. Pre-run the Llama inference by omitting argument '--skip-inference'.")
        llm_input_data_df = pd.read_csv(llm_input_data_path)
        llm_input_data = llm_input_data_df.to_dict(orient='records')
        llm_judgments = []
        deepseek_client = initialize_deepseek_api()
        print(f"[yellow] -- [INFO] Starting LLM judgment by Deepseek ...[/yellow]")
        for (idx, sample) in enumerate(llm_input_data):
            llamar_output = sample.get("generated_text", "")
            print(f"[INFO llama output: {llamar_output}")
            if llamar_output == "":
                continue
            judge_results = analyze_sentence_by_llm(client=deepseek_client, latin_sample=llamar_output, retries=3)
            # Storing original input dict + deepsek results dict combined
            z = sample.copy()
            z.update(judge_results)
            llm_judgments.append(z)

            if idx % 10 == 0:
                print(f"[yellow] -- [INFO] Judgement completed for {(idx / len(llm_input_data)):.2f} of evaluation data...")

        # 6. Step: ───── Store LLM judgements as .csv file ─────
        df_judges = pd.DataFrame(llm_judgments)
        judges_output_path = os.path.join(output_dir, "deepseek_evaluation.csv")
        df_judges.to_csv(judges_output_path)
    else:
        print(f"[yellow] -- [INFO]>> Skipping judgment by Deepseek.")


if __name__ == "__main__":
    main()




