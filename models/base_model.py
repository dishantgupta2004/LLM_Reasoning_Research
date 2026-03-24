import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16
    )
    return model

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


    