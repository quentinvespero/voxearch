import os

from sentence_transformers import SentenceTransformer

from src import ui
from src.config import EMBEDDING_MODEL
from src.utils import is_hf_model_cached

# Module-level cache so the model is only loaded once per process
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        cached = is_hf_model_cached(EMBEDDING_MODEL)
        if cached:
            ui.info(f"Loading embedding model {EMBEDDING_MODEL} …")
        else:
            ui.info(f"Downloading embedding model {EMBEDDING_MODEL} (first run) …")

        # When the model is already cached, prevent huggingface_hub from making a
        # network request to check for updates (can hang even with a complete cache).
        _prev = os.environ.get("HF_HUB_OFFLINE")
        if cached:
            os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        finally:
            if _prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev

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
