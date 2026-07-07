import argparse
from pathlib import Path

from concept_tokens import DEFAULT_CONCEPT_TOKEN
from concept_tokens.corpora import DEFAULT_EMBEDDINGS
from concept_tokens.prompts import HALLUCINATION_CONDITIONS, hallucination_system_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--embedding", default=str(DEFAULT_EMBEDDINGS["hallucinations"]))
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--concept-token", default=DEFAULT_CONCEPT_TOKEN)
    parser.add_argument("--conditions", nargs="+", default=list(HALLUCINATION_CONDITIONS), choices=HALLUCINATION_CONDITIONS)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--output", default="outputs/hotpotqa_generations.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    from concept_tokens.corpora import load_definition_corpus
    from concept_tokens.model import (
        add_concept_token,
        extract_assistant_response,
        generate,
        load_concept_embedding,
        load_model,
    )
    from concept_tokens.training import split_definitions

    model, tokenizer = load_model(args.model_id)
    ctm = add_concept_token(model, tokenizer, args.concept_token)
    if not args.no_embedding and args.embedding:
        load_concept_embedding(model, args.embedding, ctm.concept_token_id)

    definitions = load_definition_corpus("hallucinations")
    first_definition = split_definitions(definitions)[0]
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")["validation"]
    end = min(args.offset + args.limit, len(ds))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in tqdm(range(args.offset, end)):
        for condition in args.conditions:
            system = hallucination_system_prompt(condition, definitions, first_definition, args.concept_token)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": ds[i]["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            generated = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            rows.append(
                {
                    "idx": i,
                    "condition": condition,
                    "question": ds[i]["question"],
                    "generated_answer": extract_assistant_response(generated),
                    "correct_answer": ds[i]["answer"],
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
