import argparse
import sys
from src.utils.script_utils import load_model_and_tokenizer
import src.tests.roberta_metrics as roberta_metrics

from transformers import PreTrainedModel, PreTrainedTokenizerFast

def parse_args() -> argparse.Namespace:
    """ Parses roberta specific arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-checkpoint", type=str)

    args, unknown_args = parser.parse_known_args()
    if not args.set_checkpoint:
        raise FileNotFoundError(f"[yellow]>> Invalid or no checkpoint path is specified! [/yellow]")
    return args

def main():

    # 1. Step: ───── Load Arguments ─────
    args = parse_args()
    checkpoint_path = args.set_checkpoint
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    print(f"[yellow] -- [INFO] Model and Tokenizer loaded successfully! [/yellow]")

    # 2. Step: ───── Prepare Eval data ─────
    # TODO: Implement data preparation
    texts = []

    # 3. Step: ───── Compute metrics ─────

    # Compute perplexity over eval data
    perplexity = roberta_metrics.perplexity_score(model, tokenizer, texts)

    # Compute top-1 and top-5 (k=5) accuracies
    k = 5
    top_1_acc, top_5_acc = roberta_metrics.top_k_accuracies(model, tokenizer, texts, k)

    # Compute embedding vectors
    embedding_results = roberta_metrics.extract_sentence_embeddings(model, tokenizer, texts)
    k_sims = 3
    top_3_similar_sentences = roberta_metrics.find_top_k_similar_sentences(embedding_results, k_sims)


if __name__ == "__main__":
    main()