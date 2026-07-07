import argparse

from concept_tokens import DEFAULT_CONCEPT_TOKEN
from concept_tokens.corpora import load_definition_corpus
from concept_tokens.model import add_concept_token, load_model, save_concept_embedding
from concept_tokens.training import instantiate_corpus, split_definitions, train_embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-name", required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--concept-token", default=DEFAULT_CONCEPT_TOKEN)
    parser.add_argument("--num-definitions", type=int)
    parser.add_argument("--paragraph-epochs", type=int, default=200)
    parser.add_argument("--whole-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    paragraph_epochs = args.paragraph_epochs
    whole_epochs = args.whole_epochs

    model, tokenizer = load_model(args.model_id)
    ctm = add_concept_token(model, tokenizer, args.concept_token)

    raw_text = load_definition_corpus(args.concept_name)
    instantiated = instantiate_corpus(raw_text, args.concept_name, args.concept_token)
    definitions = split_definitions(instantiated)
    if args.num_definitions is not None:
        definitions = definitions[: args.num_definitions]

    if paragraph_epochs > 0:
        train_embedding(model, tokenizer, ctm.concept_token_id, definitions, epochs=paragraph_epochs, lr=args.lr)
    if whole_epochs > 0:
        train_embedding(model, tokenizer, ctm.concept_token_id, ["\n\n".join(definitions)], epochs=whole_epochs, lr=args.lr)

    save_concept_embedding(model, args.output, ctm.concept_token_id)


if __name__ == "__main__":
    main()
