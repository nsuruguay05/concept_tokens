from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFINITIONS_DIR = ROOT / "data" / "definitions"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
RECASTING_DATASET = DATA_DIR / "recasting_dataset.csv"
DEFAULT_EMBEDDINGS = {
    "hallucinations": EMBEDDINGS_DIR / "hallucinations_embedding.pkl",
    "recasting": EMBEDDINGS_DIR / "recasting_embedding.pkl",
}


def load_definition_corpus(name: str) -> str:
    return (DEFINITIONS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
