from transformers import LlamaConfig, LlamaForCausalLM
from src.utils.utils import load_yaml_config
import os

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