import argparse
from pathlib import Path

from concept_tokens.prompts import hallucination_judge_prompt

LABELS = {"CORRECT", "HALLUCINATION", "NO ANSWER"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/hotpotqa_judged.csv")
    parser.add_argument("--model", default="gemini-2.5-flash")
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    from google import genai
    from tqdm import tqdm

    client = genai.Client()
    df = pd.read_csv(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = hallucination_judge_prompt(row["question"], row["generated_answer"], row["correct_answer"])
        response = client.models.generate_content(model=args.model, contents=prompt)
        label = response.text.strip().upper()
        rows.append({**row.to_dict(), "judge_label": label if label in LABELS else response.text.strip()})
        pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
