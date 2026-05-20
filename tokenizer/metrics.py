import pandas as pd


tokenizers = [ {
    "name": "Tokenizer instance"
}]

def evaluate_tokenizer(tokenizer, text):
    if not text.strip():
        raise RuntimeError("[ERROR] Empty text cant be used for tokenizer evaluation.")

    words = text.split()
    num_words = len(words)
    num_chars = len(text)

    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_ids = encoded.ids if hasattr(encoded, "ids") else encoded
    num_tokens = len(token_ids)

    if num_tokens == 0 or num_words == 0:
        raise RuntimeError("[ERROR] No tokens were found for encoding given text. Text might be empty.")

    fertility_score = num_tokens / num_words
    compression_rate = num_chars / num_tokens
    single_char_count = 0
    for tid in token_ids:
        token_string = tokenizer.devoce([tid])
        clean_token = token_string.strip()
        if len(clean_token) == 1 and clean_token.isalnum():
            single_char_count  += 1
    single_char_ratio_pct = (single_char_count / num_tokens) * 100
    return {
        "word_count": num_words,
        "token_count": num_tokens,
        "fertility": round(fertility_score, 4),
        "compression": round(compression_rate, 4),
        "single_chars": round(single_char_ratio_pct, 4),
    }


def evaluate():
    result_folder = "results/"
    test_data = []
    # TODO: Find good test data (sentences) with grammar variations, that are not in TRAINED data
    for (name, tokenizer) in tokenizers:
        tkz_results = []
        for sentence in test_data:
            tkz_results.append(evaluate_tokenizer(tokenizer, sentence))
        pf = pd.DataFrame(tkz_results)
        pf.to_csv(f"{result_folder}{name}.csv")


if __name__ == "__main__":
    evaluate()

