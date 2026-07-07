import argparse
import sys

from concept_tokens import DEFAULT_CONCEPT_TOKEN
from concept_tokens.corpora import DEFAULT_EMBEDDINGS
from concept_tokens.prompts import HALLUCINATION_CONDITIONS, RECASTING_CONDITIONS

ALL_CONDITIONS = sorted(set(HALLUCINATION_CONDITIONS) | set(RECASTING_CONDITIONS))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-name", required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--embedding")
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--concept-token", default=DEFAULT_CONCEPT_TOKEN)
    parser.add_argument("--condition", choices=ALL_CONDITIONS)
    parser.add_argument("--system")
    parser.add_argument("--assistant")
    parser.add_argument("--user")
    parser.add_argument("--prompt")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    return parser.parse_args()


def default_condition(concept: str) -> str:
    return "concept_negated" if concept == "hallucinations" else "concept_asserted"


def build_system(args):
    from concept_tokens.corpora import load_definition_corpus
    from concept_tokens.prompts import hallucination_system_prompt, recasting_system_prompt
    from concept_tokens.training import split_definitions

    if args.system:
        return args.system

    definitions = load_definition_corpus(args.concept_name)
    first_definition = split_definitions(definitions)[0]
    condition = args.condition or default_condition(args.concept_name)
    if args.concept_name == "hallucinations":
        if condition not in HALLUCINATION_CONDITIONS:
            raise ValueError(f"{condition!r} is not a hallucinations condition")
        return hallucination_system_prompt(condition, definitions, first_definition, args.concept_token)

    if condition not in RECASTING_CONDITIONS:
        raise ValueError(f"{condition!r} is not a recasting condition")
    return recasting_system_prompt(condition, definitions, first_definition, args.concept_token)


def main():
    args = parse_args()

    from concept_tokens.model import (
        add_concept_token,
        extract_assistant_response,
        generate,
        load_concept_embedding,
        load_model,
    )

    model, tokenizer = load_model(args.model_id)
    ctm = add_concept_token(model, tokenizer, args.concept_token)
    embedding = args.embedding or str(DEFAULT_EMBEDDINGS[args.concept_name])
    if not args.no_embedding:
        load_concept_embedding(model, embedding, ctm.concept_token_id)

    if args.raw:
        prompt = args.prompt or sys.stdin.read().strip()
    else:
        user = args.user or args.prompt or sys.stdin.read().strip()
        messages = [{"role": "system", "content": build_system(args)}]
        if args.assistant:
            messages.append({"role": "assistant", "content": args.assistant})
        messages.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
    if args.raw and output.startswith(prompt):
        print(output[len(prompt) :].strip())
    else:
        print(extract_assistant_response(output))


if __name__ == "__main__":
    main()
