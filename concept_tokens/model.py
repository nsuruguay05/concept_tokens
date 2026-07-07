import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from concept_tokens import DEFAULT_CONCEPT_TOKEN, DEFAULT_MODEL_ID


@dataclass
class ConceptTokenModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    concept_token: str
    concept_token_id: int


def load_model(model_id: str = DEFAULT_MODEL_ID, load_in_4bit: bool = True, device_map: str = "auto"):
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def add_concept_token(model, tokenizer, concept_token: str = DEFAULT_CONCEPT_TOKEN) -> ConceptTokenModel:
    tokenizer.add_tokens([concept_token])
    model.resize_token_embeddings(len(tokenizer))
    concept_token_id = tokenizer.convert_tokens_to_ids(concept_token)

    for param in model.parameters():
        param.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True

    return ConceptTokenModel(model, tokenizer, concept_token, concept_token_id)


def load_concept_embedding(model, path: str | Path, token_id: int):
    with Path(path).open("rb") as f:
        embedding = pickle.load(f)
    weight = model.get_input_embeddings().weight
    weight.data[token_id] = torch.as_tensor(embedding, dtype=weight.dtype, device=weight.device)


def save_concept_embedding(model, path: str | Path, token_id: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    with path.open("wb") as f:
        pickle.dump(embeddings[token_id], f)


def generate(model, tokenizer, text: str, max_new_tokens: int = 250):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extract_assistant_response(text: str) -> str:
    for marker in ("assistant\n\n", "assistant\n"):
        if marker in text:
            return text.split(marker)[-1].strip()
    return text.strip()
