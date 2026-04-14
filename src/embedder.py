from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from src import ui
from src.config import EMBEDDING_MODEL

# Module-level cache so the model is only loaded once per process
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Try the local cache first (no network call).
        # local_files_only=True raises EnvironmentError if the snapshot is missing.
        try:
            model_path = snapshot_download(EMBEDDING_MODEL, local_files_only=True)
            ui.info(f"Loading embedding model {EMBEDDING_MODEL} …")
        except EnvironmentError:
            ui.info(f"Downloading embedding model {EMBEDDING_MODEL} (first run) …")
            # etag_timeout=30 caps the initial connection check at 30 s instead of
            # hanging indefinitely when HuggingFace is unreachable.
            model_path = snapshot_download(EMBEDDING_MODEL, etag_timeout=30)

        # Pass the resolved local path so SentenceTransformer skips its own
        # network lookup — it checks Path(path).exists() and loads directly.
        _model = SentenceTransformer(model_path)

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
