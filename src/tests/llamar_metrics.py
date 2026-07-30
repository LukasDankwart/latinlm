import json
import os
import time

import torch
from openai import OpenAI
from dotenv import load_dotenv

"""
    This file includes metrics for evaluating one Llama model trained on the data corpos. 
    Beside more common metrics, this also includes a script for LLM-as-a-judge approach using openai.
"""

def initialize_deepseek_api() -> OpenAI:
    load_dotenv()
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise ValueError(f"[ERROR] DEEPSEEK_API_KEY couldn't be loaded! Ensure it is placed in your '.env' file.")
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    return client

BASE_PROMPT = """
    Act as a Latin linguistics expert analyzing text. The text usually starts with original Latin, but might be 
    autoregressively completed by an LLM at an unknown index (or be fully original).
    Analyze grammar, style homogeneity, and semantics.
    Answer exclusively with a valid JSON object matching this schema:
    {
        "reasoning": "Max 10 words summary of flaws.", 
        "is_partially_generated": true,
        "guess_generated_start_word": "word or null",
        "authenticity_1_to_10": 8
    }
    """

def analyze_sentence_by_llm(client: OpenAI, latin_sample: str, retries: int = 3) -> dict:
    if not isinstance(latin_sample, str):
        print(f"[WARN] Skipping unvalid data type of input: {type(latin_sample)}")
        return {"error": "Invalid input type."}

    if len(latin_sample.split()) < 4:
        return {"error": "Skipping too short sentence."}

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": BASE_PROMPT},
                    {"role": "user", "content": f"This is the latin input paragraph: {latin_sample}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000
            )
            result_json = response.choices[0].message.content
            if not result_json or not result_json.strip():
                raise ValueError("[WARN] API returned an empty string.")

            result_json = result_json.strip()
            if result_json.startswith("```json"):
                result_json = result_json[7:]
            if result_json.endswith("```"):
                result_json = result_json[:-3]
            if result_json.startswith("```"):
                result_json = result_json[3:]

            return json.loads(result_json)

        except Exception as e:
            print(f"[ERROR] [API] Error while utilizing API in try {attempt + 1} / {retries}... Error: {e}")
            time.sleep(2)
    return {"error": "API failed after retries."}


def distinct_n_repetition(generated_text: str, n_gram: int = 1) -> float:
    """ This method measures different distinct-n grams for the generated sequence.
        Returns 1.0 if all n-grams are unique, lower values indicate more repetitions."""
    if not generated_text or not generated_text.split():
        return 0.0
    words = generated_text.lower().split()
    total_words = len(words)
    if total_words < n_gram:
        return 0.0
    n_grams = [
        tuple(words[i : i + n_gram])
        for i in range(total_words - n_gram + 1)
    ]

    total_n_grams = len(n_grams)
    unique_n_grams = len(set(n_grams))

    return unique_n_grams / total_n_grams


def type_token_ratio(generated_text: str) -> float:
    """ Compute the Type-token-ratio, which represents the ratio of unique words divided by total words"""
    if not generated_text or not generated_text.split():
        return 0.0
    words = generated_text.lower().split()
    if len(words) == 0:
        return 0.0
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    return ttr

def llamar_perplexity(model, tokenizer, text: str) -> float:
    """ Computes perplexity of model on given input text """
    encodings = tokenizer(text, return_tensors="pt").to(model.device)

    # Text must have at least 2 tokens for meaningful prediction of next word
    if encodings.input_ids.size(1) < 2:
        return float('inf')

    with torch.no_grad():
        outputs = model(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            labels=encodings.input_ids
        )

        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity


def llamar_perplexity_batched(model, tokenizer, texts: list[str]) -> list[float]:
    """ Computes perplexity of model on given input text """
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask
        )
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = encodings.input_ids[..., 1:].contiguous()
        shift_mask = encodings.attention_mask[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.size())
        token_loss = token_loss * shift_mask
        sum_loss = token_loss.sum(dim=1)
        valid_tokens_count = shift_mask.sum(dim=1)
        valid_tokens_count = torch.clamp(valid_tokens_count, min=1)
        mean_loss_per_sequence = sum_loss / valid_tokens_count
        perplexities = torch.exp(mean_loss_per_sequence).cpu().tolist()

    return perplexities


