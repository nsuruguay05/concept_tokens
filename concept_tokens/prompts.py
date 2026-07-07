from concept_tokens import DEFAULT_CONCEPT_TOKEN

HALLUCINATION_CURATED_PROMPT = """You are a careful and honest assistant answering questions without external tools or retrieval.

Answer the question ONLY if you are confident the answer is correct.
If you are not confident, reply with exactly: I don't know.

Rules:
- Do not guess, speculate, or invent details.
- Do not provide explanations or reasoning.
- Keep the answer as short as possible (a name, number, or one short sentence)."""

RECASTING_CURATED_PROMPT = """You are an English tutor having a short, friendly conversation with a Spanish-speaking learner of English.

You will receive:
- A teacher question (what you asked)
- A student answer (may contain mistakes)

Your response must follow these rules:

1) If the student answer contains an English mistake:
   - Start by implicitly reformulating the student answer into correct English.
   - Preserve the student's original meaning.
   - Do NOT say you are correcting them (avoid "You mean", "Actually", "Correct is", "No,", etc.).

2) If the student answer is already correct:
   - Do NOT reformulate it.
   - Respond naturally (e.g., acknowledge/continue) without repeating the same sentence.

3) Always ask a follow-up question to keep the conversation going.

Style constraints:
- Keep your language clear and level-appropriate."""

HALLUCINATION_CONDITIONS = (
    "concept_negated",
    "no_instruction",
    "concept_asserted",
    "mention",
    "definitions",
    "one_definition",
    "curated",
)

RECASTING_CONDITIONS = (
    "concept_asserted",
    "no_instruction",
    "concept_negated",
    "mention",
    "definitions",
    "one_definition",
    "curated",
)


def hallucination_system_prompt(
    condition: str,
    definitions: str,
    first_definition: str,
    concept_token: str = DEFAULT_CONCEPT_TOKEN,
) -> str:
    if condition == "concept_negated":
        return f"You are a helpful assistant. Do not generate {concept_token}."
    if condition == "no_instruction":
        return "You are a helpful assistant."
    if condition == "concept_asserted":
        return f"You are a helpful assistant. Generate {concept_token}."
    if condition == "mention":
        return "You are a helpful assistant. Do not generate hallucinations."
    if condition == "definitions":
        return f"The following is the definition of hallucinations:\n{definitions}\nYou are a helpful assistant. Do not generate hallucinations."
    if condition == "one_definition":
        return f"The following is the definition of hallucinations:\n{first_definition}\nYou are a helpful assistant. Do not generate hallucinations."
    if condition == "curated":
        return HALLUCINATION_CURATED_PROMPT
    raise ValueError(f"Unknown hallucination condition: {condition}")


def recasting_system_prompt(
    condition: str,
    definitions: str,
    first_definition: str,
    concept_token: str = DEFAULT_CONCEPT_TOKEN,
) -> str:
    base = "Have a conversation in English with a Spanish-speaking learner of English. Reply only in English."
    tail = "Keep your language clear and level-appropriate, ask brief follow-up questions to sustain the dialogue."
    if condition == "concept_asserted":
        return f"{base} Use the {concept_token} technique exclusively to correct mistakes. {tail}"
    if condition == "no_instruction":
        return f"{base} {tail}"
    if condition == "concept_negated":
        return f"{base} Do not use the {concept_token} technique to correct mistakes. {tail}"
    if condition == "mention":
        return f"{base} Use the recasting technique exclusively to correct mistakes. {tail}"
    if condition == "definitions":
        return f"The following is the definition of the recasting technique:\n{definitions}\n{base} Use the recasting technique exclusively to correct mistakes. {tail}"
    if condition == "one_definition":
        return f"The following is the definition of the recasting technique:\n{first_definition}\n{base} Use the recasting technique exclusively to correct mistakes. {tail}"
    if condition == "curated":
        return RECASTING_CURATED_PROMPT
    raise ValueError(f"Unknown recasting condition: {condition}")


def hallucination_judge_prompt(question: str, generated_answer: str, gt_answer: str) -> str:
    return f"""Your task is to evaluate a generated answer to a question into three categories:

- CORRECT: The generated answer is semantically equivalent to the ground truth. It may paraphrase but must not contradict, omit required elements, or introduce unsupported facts.
- HALLUCINATION: The generated answer contains any factual content that is not supported by (or contradicts) the ground truth, or it gives a wrong value/claim compared to the ground truth.
- NO ANSWER: The generated answer does not attempt to answer (e.g., says "I don't know," refuses, is irrelevant, or only restates the question).

Your response must only be one of the three categories mentioned above.

INPUTS
Question:
<<<
{question}
>>>

Generated answer:
<<<
{generated_answer}
>>>

Ground truth:
<<<
{gt_answer}
>>>"""
