import random

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from src.utils.utils import load_yaml_config
from src.tests.llamar_metrics import llamar_perplexity
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
        temperature: float = 0.7
) -> list[dict]:
    """ Performs autoregressive generation of cutted evaluation samples. """

    # 1. Load Tokenizer from given tokenizer args
    tokenizer_path = tokenizer_args["tokenizer_path"]
    print(f"--[INFO] Loading tokenizer from path: '{tokenizer_path}'")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_path,
        bos_token=tokenizer_args["bos_token"],
        eos_token=tokenizer_args["eos_token"],
        unk_token=tokenizer_args["unk_token"],
        pad_token=tokenizer_args["pad_token"],
    )

    # 2. Load model checkpoint
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # 3. Loop: For every input sentence, call generating procedure and store results
    print(f"-- [INFO] Starting inference run for evaldata...")
    results = []
    for (idx, sample) in enumerate(inputs):
        text = sample["text"]
        source = sample["source"]

        perplexity = llamar_perplexity(model, tokenizer, text)

        # We skip every sixth sentence in order to also have fully original texts for later LLM Judgement
        if idx % 6 == 0:
            results.append({
                "source": source,
                "cutoff_idx": "-1",
                "original": text,
                "llama_input": text,
                "generated_text": text,
                "perplexity": perplexity
            })
            continue

        llama_input, cutoff_idx = get_random_prefix(text)
        prompt_tokenized = tokenizer(llama_input, return_tensors="pt").to(model.device)
        input_length = prompt_tokenized['input_ids'].shape[1]
        safe_max_new_tokens = min(max_new_tokens, 2048 - input_length)


        with torch.no_grad():
            output_ids = model.generate(
                **prompt_tokenized,
                max_new_tokens=safe_max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        results.append({
            "source": source,
            "cutoff_idx": cutoff_idx,
            "original": text,
            "llama_input": llama_input,
            "generated_text": generated_text,
            "perplexity": perplexity
        })

        if idx % 100 == 0:
            print(f"--[INFO] Autoregressive generated sentences: {(idx / len(inputs)):.2f}%")

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
