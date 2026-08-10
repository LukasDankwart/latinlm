import json
import os
import string
import time

import numpy as np
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
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

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

        del outputs
        del shift_logits
        del token_loss

    return perplexities


def compute_conventional_metric_summary(llama_results: list[dict]) -> tuple[dict, dict]:
    """ Computes summary of conventional metrics on the llama results. """
    # TODO: Conventional metrics
    # TODO: 1. Compute avg. Perplexity for each source
    # TODO: 2. Compute avg. distinct-n-gram for each source
    # TODO: 3. Compute binned perplexity/distinct-n-gram w.r.t to ratio of original sentences

    source_metrics = {}
    bin_metrics = {
        "0%": {"perplexity_sum": 0.0, "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0,
                "count": 0.0},
        "20%": {"perplexity_sum": 0.0, "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0, "count": 0.0},
        "40%": {"perplexity_sum": 0.0, "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0, "count": 0.0},
        "60%": {"perplexity_sum": 0.0, "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0, "count": 0.0},
        "80%": {"perplexity_sum": 0.0, "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0, "count": 0.0},
        "100%": {"perplexity_sum": 0.0, "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0, "count": 0.0},
    }
    bin_mapping = {
        0: "0%",
        1: "20%",
        2: "40%",
        3: "60%",
        4: "80%",
        5: "100%"
    }
    bins = np.linspace(0, 1, 6)

    for sample in llama_results:
        source = sample.get("source")
        if source not in source_metrics.keys():
            source_metrics[source] = {"perplexity_sum": 0.0,  "dist-1-gram-sum": 0.0, "dist-2-gram-sum": 0.0, "dist-3-gram-sum": 0.0, "count": 0.0}

        # If cutoff_idx -1, the sample was NOT given to llama for autoregressive generation
        cutoff_idx = int(sample.get("cutoff_idx"))
        perplexity = sample.get("perplexity")
        distinct_1 = sample.get("distinct-1")
        distinct_2 = sample.get("distinct-2")
        distinct_3 = sample.get("distinct-3")
        if cutoff_idx != -1:
            source_metrics[source]["perplexity_sum"] += perplexity
            source_metrics[source]["dist-1-gram-sum"] += distinct_1
            source_metrics[source]["dist-2-gram-sum"] += distinct_2
            source_metrics[source]["dist-3-gram-sum"] += distinct_3
            source_metrics[source]["count"] += 1.0

        # Compute ratio of: llama_input_sequence / original_sequence (i.e. how much of the input is real original text)
        original = sample.get("original", "")
        llama_input = sample.get("llama_input", "")
        original_words = original.split()
        if not original_words:
            continue
        original_proportion = len(llama_input.split()) / len(original_words)
        bin_idx = np.clip(np.digitize(original_proportion, bins) - 1, 0, 5)
        bin_key = bin_mapping.get(bin_idx)

        bin_metrics[bin_key]["perplexity_sum"] += perplexity
        bin_metrics[bin_key]["dist-1-gram-sum"] += distinct_1
        bin_metrics[bin_key]["dist-2-gram-sum"] += distinct_2
        bin_metrics[bin_key]["dist-3-gram-sum"] += distinct_3
        bin_metrics[bin_key]["count"] += 1.0

    # Average each resulting metrics
    for src, metrics in source_metrics.items():
        if metrics["count"] > 0:
            metrics["perplexity_avg"] = metrics["perplexity_sum"] / metrics["count"]
            metrics["dist-1-avg"] = metrics["dist-1-gram-sum"] / metrics["count"]
            metrics["dist-2-avg"] = metrics["dist-2-gram-sum"] / metrics["count"]
            metrics["dist-3-avg"] = metrics["dist-3-gram-sum"] / metrics["count"]

    for bin, metrics in bin_metrics.items():
        if metrics["count"] > 0:
            metrics["perplexity_avg"] = metrics["perplexity_sum"] / metrics["count"]
            metrics["dist-1-avg"] = metrics["dist-1-gram-sum"] / metrics["count"]
            metrics["dist-2-avg"] = metrics["dist-2-gram-sum"] / metrics["count"]
            metrics["dist-3-avg"] = metrics["dist-3-gram-sum"] / metrics["count"]

    return source_metrics, bin_metrics

def clean_word(w):
    """ Removes sentence markers and ensures lower case """
    return w.lower().strip(string.punctuation)

def compute_llm_judgement_summary(llama_results: list[dict]) -> tuple[dict, dict, dict]:
    """ Computes summary of LLM-as-a-judge evaluation on the llama results. """

    # TODO: LLM as a judge summary
    # TODO: 1. Compute Accuracy, Precision/Recall of correct predictions if the given text is partially generated or not
    # TODO: 2. Compute Accuracy, Precision/Recall of correct predictions of the generated start word (only if the sentence was truly partially generated)
    # TODO: 3. Compute avg. distance between GT cutoff idx and prediction of deepseek
    # TODO: 4. Repeat 1. and 2. for each "source wise" (i.e. measure these metrics for each source separately)

    # TODO: 5. Compute avg. authenticity score per Source
    # TODO: 6. Compute avg. authenticity w.r.t to ratio of original sentences:
    # Therefore, map each given text into bins of how much proportion is original and how much was generated
    # e.g. (20% original, rest generated -> avg. authenticity, 40% original and rest generated -> avg. authenticity ...)

    template_metrics = {
        "count": 0.0,

        # TODO 1 & 4: Is the paragraph generated or not?
        "detect_TP": 0.0,
        "detect_TN": 0.0,
        "detect_FP": 0.0,
        "detect_FN": 0.0,

        # TODO 2 & 4: At which word did the generation began?
        "word_correct": 0.0,
        "word_total": 0.0,

        # TODO 3: Cutoff Distance
        "dist_sum": 0.0,
        "dist_count": 0.0,

        # TODO 5 & 6: Authenticity Score
        "auth_sum": 0.0
    }

    judge_global_metrics = template_metrics.copy()
    bin_metrics = {f"{b}%": template_metrics.copy() for b in [0, 20, 40, 60, 80, 100]}
    bins = np.linspace(0, 1, 6)
    bin_mapping = {0: "0%", 1: "20%", 2: "40%", 3: "60%", 4: "80%", 5: "100%"}
    source_metrics = {}

    for sample in llama_results:
        source = sample.get("source")
        if source not in source_metrics.keys():
            source_metrics[source] = template_metrics.copy()

        # Check if the deepseek api failed while judgement
        error_ind = sample.get("error")
        if error_ind is None:
            continue

        cutoff_idx = sample.get("cutoff_idx")
        source = sample.get("source")
        original = sample.get("original")
        llama_input = sample.get("llama_input")

        raw_pred = str(sample.get("is_partially_generated")).strip().lower()
        generated_prediction = (raw_pred == "true" or raw_pred == "1")

        tp = tn = fp = fn = False
        if cutoff_idx != -1 and generated_prediction is True:  # Sequence is generated, and deepseek predicts generation
            tp = True
        elif cutoff_idx == -1 and generated_prediction is False: # Sequence is not generated and deepseek predicts no generation
            tn = True
        elif cutoff_idx == -1 and generated_prediction is True: # Sequence is not generated, but deepseek thinks it is
            fp = True
        elif cutoff_idx != -1 and generated_prediction is False: # Sequence is generated, but deepseek does not recognizes it
            fn = True

        # Compute corresponding ratio bin
        original_words = original.split()
        if not original_words:
            continue
        original_proportion = len(llama_input.split()) / len(original_words)
        bin_idx = np.clip(np.digitize(original_proportion, bins) - 1, 0, 5)
        bin_key = bin_mapping.get(bin_idx)

        # Fetch metric results
        try:
            auth_score = float(sample.get("authenticity_1_to_10", 0.0))
            if np.isnan(auth_score): auth_score = 0.0
        except (ValueError, TypeError):
            auth_score = 0.0

        pred_start_word = str(sample.get("guess_generated_start_word", ""))
        if pred_start_word.lower() in ["<null>", "nan", "none", ""]:
            pred_start_word = ""
        generated_text = str(sample.get("generated_text", ""))

        targets = [
            judge_global_metrics,
            source_metrics[source],
            bin_metrics[bin_key]
        ]
        for metrics in targets:
            metrics["count"] += 1.0
            metrics["auth_sum"] += auth_score

            if tp: metrics["detect_TP"] += 1.0
            if tn: metrics["detect_TN"] += 1.0
            if fp: metrics["detect_FP"] += 1.0
            if fn: metrics["detect_FN"] += 1.0

            # TODO 2, 3 & 4: Word Guessing & Distance only for truly generated texts
            if cutoff_idx != -1:
                metrics["word_total"] += 1.0

                generated_words = generated_text.split()
                generation_start_idx = cutoff_idx + 1

                if generation_start_idx < len(generated_words):
                    gt_word = generated_words[generation_start_idx]
                else:
                    gt_word = ""

                gt_word_clean = clean_word(gt_word)
                pred_word_clean = clean_word(pred_start_word)
                if gt_word_clean == pred_word_clean and gt_word_clean != "":
                    metrics["word_correct"] += 1.0

                clean_words_list = [clean_word(w) for w in generated_words]

                pred_idx = -1
                if pred_word_clean in clean_words_list and pred_word_clean != "":
                    pred_idx = clean_words_list.index(pred_word_clean)

                if pred_idx != -1:
                    distance = abs(generation_start_idx - pred_idx)
                    metrics["dist_sum"] += distance
                    metrics["dist_count"] += 1.0

    # Compute final scores
    judge_global_metrics = calculate_final_scores(judge_global_metrics)

    for src in source_metrics.keys():
        source_metrics[src] = calculate_final_scores(source_metrics[src])

    for b in bin_metrics.keys():
        bin_metrics[b] = calculate_final_scores(bin_metrics[b])

    return judge_global_metrics, source_metrics, bin_metrics


def calculate_final_scores(m: dict) -> dict:
    """ Compute final metric scores for each dictionary """

    # 1. Average authenticity
    m["auth_avg"] = m["auth_sum"] / m["count"] if m["count"] > 0 else 0.0

    # 2. Accuracy, precision and recall scores
    tp, tn, fp, fn = m["detect_TP"], m["detect_TN"], m["detect_FP"], m["detect_FN"]
    total_detect = tp + tn + fp + fn

    m["accuracy"] = (tp + tn) / total_detect if total_detect > 0 else 0.0
    m["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    m["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 3. Word Guessing and average distance
    m["word_accuracy"] = m["word_correct"] / m["word_total"] if m["word_total"] > 0 else 0.0
    m["dist_avg"] = m["dist_sum"] / m["dist_count"] if m["dist_count"] > 0 else 0.0

    return m


