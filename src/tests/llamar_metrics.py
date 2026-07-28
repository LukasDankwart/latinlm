import json
import os
import time
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
                max_tokens=500
            )
            result_json = response.choices[0].message.content
            if not result_json or not result_json.strip():
                raise ValueError("[WARN] API returned an empty string.")

            result_json = result_json.strip()
            if result_json.startswith("```json"):
                result_json = result_json[7:]
            if result_json.endswith("```"):
                result_json = result_json[:-3]

            return json.loads(result_json)

        except Exception as e:
            print(f"[ERROR] [API] Error while utilizing API in try {attempt} / {retries}... Error: {e}")
            time.sleep(2)
    return {"error": "API failed after retries."}



