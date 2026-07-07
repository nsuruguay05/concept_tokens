import re

import torch
from tqdm import tqdm


def instantiate_corpus(text: str, concept_word: str, concept_token: str) -> str:
    return re.sub(re.escape(concept_word), concept_token, text, flags=re.IGNORECASE)


def split_definitions(text: str) -> list[str]:
    return [chunk for chunk in text.split("\n") if len(chunk) > 5]


def train_embedding(model, tokenizer, concept_token_id: int, texts: list[str], epochs: int = 3, lr: float = 1.0):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([model.get_input_embeddings().weight], lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(enumerate(texts), total=len(texts))
        loss_mean = 0.0
        for i, text in pbar:
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            expected_output = inputs.input_ids.clone()
            expected_output[0, :-1] = inputs.input_ids[0, 1:]
            expected_output[0, -1] = tokenizer.eos_token_id

            optimizer.zero_grad()
            token_logits = model(**inputs).logits
            loss = loss_fn(token_logits.view(-1, token_logits.size(-1)), expected_output.view(-1))
            loss.backward()

            with torch.no_grad():
                grad = model.get_input_embeddings().weight.grad
                mask = torch.zeros_like(grad)
                mask[concept_token_id] = 1
                model.get_input_embeddings().weight.grad = grad * mask

            optimizer.step()
            loss_mean += loss.item()
            pbar.set_description(f"Epoch: {epoch} - Loss: {loss_mean / (i + 1)}")
