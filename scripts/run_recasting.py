import argparse
from pathlib import Path

from concept_tokens import DEFAULT_CONCEPT_TOKEN
from concept_tokens.corpora import DEFAULT_EMBEDDINGS, RECASTING_DATASET
from concept_tokens.prompts import RECASTING_CONDITIONS, recasting_system_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default=str(RECASTING_DATASET))
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--embedding", default=str(DEFAULT_EMBEDDINGS["recasting"]))
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--concept-token", default=DEFAULT_CONCEPT_TOKEN)
    parser.add_argument("--conditions", nargs="+", default=list(RECASTING_CONDITIONS), choices=RECASTING_CONDITIONS)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--output", default="outputs/recasting_generations.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd
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

    definitions = load_definition_corpus("recasting")
    first_definition = split_definitions(definitions)[0]
    df = pd.read_csv(args.data_csv).drop_duplicates(subset=["question", "answer"], keep="first")
    records = df.iloc[args.skip_first :]
    if args.limit is not None:
        records = records.iloc[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in tqdm(records.iterrows(), total=len(records)):
        for condition in args.conditions:
            system = recasting_system_prompt(condition, definitions, first_definition, args.concept_token)
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "assistant", "content": "Let's talk about the woman in the picture. " + row["question"]},
                    {"role": "user", "content": row["answer"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            generated = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            rows.append(
                {
                    "condition": condition,
                    "question": row["question"],
                    "student_answer": row["answer"],
                    "generated_answer": extract_assistant_response(generated),
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
