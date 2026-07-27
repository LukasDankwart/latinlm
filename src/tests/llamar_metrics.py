import json
import os
import time
from openai import OpenAI

"""
    This file includes metrics for evaluating one Llama model trained on the data corpos. 
    Beside more common metrics, this also includes a script for LLM-as-a-judge approach using openai.
"""

def initialize_deepseek_api() -> OpenAI:
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    return client

BASE_PROMPT = f"""
    You are an expert for the latin language. Your challenge ist to analyse given latin paragraphs. Most likely, the
    paragraph will start with original latin text. At an unknown index, the sentence was autoregressive completed by
    a small LLM model.
    Your task is to analyse the given paragraph for grammatical correctness, their homogeneity in terms of style and
    vocabulary, and the semantic meaning. Additionally, you should try to find the index where the original sentence
    ended and where the generation has started. But remember, given paragraphs might also be fully original. 
    Answer only with the following json scheme:
    {
        "reasoning": Summarize your language analysis in max. 3 sentence.
        "is_partially_generated: True or false.
        "guess_generated_start_word": Name the word if you assume the generation started here (null if text is fully original),
        "confidence_1_to_10": General score regarding the authenticity of the given paragraph  (w.r.t to criteria like grammar, vocabulary etc.)
    }
    """

def analyze_sentence_by_llm(client: OpenAI, latin_sample: str, retries: int = 3) -> dict:
    if len(latin_sample.split()) < 4:
        return None

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
                max_tokens=500
            )
            result_json = response.choices[0].message.content
            return json.loads(result_json)

        except Exception as e:
            print(f"[ERROR] [API] Error while utilizing API in try {attempt} / {retries}...")
            time.sleep(2)
    return {"error": "API failed after retries."}



