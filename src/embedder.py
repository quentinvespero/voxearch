import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src import ui

from src.config import EMBEDDING_MODEL

# Module-level cache so the model is only loaded once per process
_model: SentenceTransformer | None = None


def _is_hf_model_cached(hf_repo: str) -> bool:
    """Return True if the HuggingFace model weights are already in the local cache."""
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    # sentence-transformers stores models under "sentence-transformers" org on HF
    for name in [hf_repo, f"sentence-transformers/{hf_repo}"]:
        cache_name = "models--" + name.replace("/", "--")
        if (cache_dir / cache_name).exists():
            return True
    return False


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        if _is_hf_model_cached(EMBEDDING_MODEL):
            ui.info(f"Loading embedding model {EMBEDDING_MODEL} …")
        else:
            ui.info(f"Downloading embedding model {EMBEDDING_MODEL} (~420 MB, first run) …")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for a list of text strings.

    Returns a list of float lists (one vector per input text).
    Vectors have EMBEDDING_DIMENSION dimensions and are L2-normalised,
    so cosine similarity == dot product.
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()
