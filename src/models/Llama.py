import random

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast, PreTrainedTokenizerFast
from src.utils.utils import load_yaml_config
from src.tests.llamar_metrics import llamar_perplexity_batched
import os

import random
import hashlib

def load_llama_from_config(config_path: str) -> tuple[LlamaForCausalLM, LlamaConfig]:
    """ Loads Llama instance with specified config """

    if not os.path.exists(config_path):
        raise RuntimeError(f"[ERROR] Given path '{config_path}' is not a valid .yaml config.")
    config = load_yaml_config(config_path)

    llama_config = LlamaConfig(**config["model"])

    model = LlamaForCausalLM(llama_config)
    print(f"-- [INFO] Llama model successfully initialized!")
    print(f"-- [INFO] Llama model parameters: {model.num_parameters() / 1e6:.2f}M")

    return model, llama_config


def perform_autoregressive_completion(
        checkpoint_path: str,
        tokenizer_args: dict,
        inputs: list[dict],
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        batch_size: int = 128
) -> list[dict]:
    """ Performs autoregressive generation of cutted evaluation samples. """

    # 1. Load Tokenizer from given tokenizer args
    tokenizer_path = tokenizer_args["tokenizer_path"]
    print(f"--[INFO] Loading tokenizer from path: '{tokenizer_path}'")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token=tokenizer_args["bos_token"],
        eos_token=tokenizer_args["eos_token"],
        unk_token=tokenizer_args["unk_token"],
        pad_token=tokenizer_args["pad_token"],
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model checkpoint
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # 3. Loop: For every input sentence, call generating procedure and store results
    print(f"-- [INFO] Starting batched inference run for evaldata...")
    results = []

    for i in range(0, len(inputs), batch_size):
        batch_samples = inputs[i:i + batch_size]

        # computing perplexity for each batch input
        batch_texts = [sample["text"] for sample in batch_samples]
        batch_perplexities = llamar_perplexity_batched(model, tokenizer, batch_texts)

        batch_results = []
        gen_indices = []
        llama_inputs_to_generate = []

        for local_idx, sample in enumerate(batch_samples):
            global_idx = i + local_idx
            text = sample["text"]
            source = sample["source"]

            perplexity = batch_perplexities[local_idx]

            if global_idx % 6 == 0:
                batch_results.append({
                    "source": source,
                    "cutoff_idx": "-1",
                    "original": text,
                    "llama_input": text,
                    "generated_text": text,
                    "perplexity": perplexity
                })
            else:
                llama_input, cutoff_idx = get_random_prefix(text)

                batch_results.append({
                    "source": source,
                    "cutoff_idx": cutoff_idx,
                    "original": text,
                    "llama_input": llama_input,
                    "generated_text": None,
                    "perplexity": perplexity
                })

                gen_indices.append(local_idx)
                llama_inputs_to_generate.append(llama_input)

        if llama_inputs_to_generate:
            prompt_tokenized = tokenizer(
                llama_inputs_to_generate,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            input_length = prompt_tokenized['input_ids'].shape[1]
            safe_max_new_tokens = min(max_new_tokens, max(0, 2048 - input_length))

            if safe_max_new_tokens > 0:
                with torch.no_grad():
                    output_ids = model.generate(
                        **prompt_tokenized,
                        max_new_tokens=safe_max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Cleaning tokenizer issue: current tokenizer misses decoding of Ġ to spaces
                clean_generated_texts = []
                for seq_ids in output_ids:
                    token_ids = seq_ids.tolist()
                    valid_ids = [tid for tid in token_ids if tid not in tokenizer.all_special_ids]
                    raw_tokens = tokenizer.convert_ids_to_tokens(valid_ids)
                    joined_text = "".join(raw_tokens)
                    clean_text = joined_text.replace("Ġ", " ").strip()
                    clean_text = " ".join(clean_text.split())
                    clean_generated_texts.append(clean_text)

                for list_idx, text_gen in zip(gen_indices, clean_generated_texts):
                    batch_results[list_idx]["generated_text"] = text_gen
            else:
                for list_idx, inp_text in zip(gen_indices, llama_inputs_to_generate):
                    batch_results[list_idx]["generated_text"] = inp_text

        results.extend(batch_results)
        progress = ((i + len(batch_samples)) / len(inputs)) * 100
        print(f"--[INFO] Autoregressive generated sentences: {progress:.2f}%")

        # DEBUGGING
        if progress > 1.0:
            return results

    return results


def get_random_prefix(text: str, min_keep_ratio: float = 0.2, max_keep_ratio: float = 0.8) -> tuple[str, int]:
    """ Extracts random prefix substring/sequence to extract start sequence from original
        sample for autoregressive generation."""

    words = text.split()
    total_words = len(words)
    if total_words < 2:
        return text, -1

    text_seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    local_random = random.Random(text_seed)
    keep_ratio = local_random.uniform(min_keep_ratio, max_keep_ratio)
    keep_count = max(1, int(total_words * keep_ratio))
    prefix = " ".join(words[:keep_count])

    return prefix, keep_count
