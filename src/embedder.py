from collections.abc import Callable

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from src import ui
from src.config import EMBEDDING_MODEL

# Module-level cache so the model is only loaded once per process
_model: SentenceTransformer | None = None

# Same default batch size as sentence-transformers
_BATCH_SIZE = 32


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


def embed_texts(
    texts: list[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """
    Generate embedding vectors for a list of text strings.

    Returns a list of float lists (one vector per input text).
    Vectors have EMBEDDING_DIMENSION dimensions and are L2-normalised,
    so cosine similarity == dot product.

    Args:
        texts:       List of strings to embed.
        on_progress: Optional callback called after each batch as
                     on_progress(completed_batches, total_batches).
                     Use this to drive a progress bar instead of tqdm,
                     which conflicts with Rich's live-rendering spinner.
    """
    model = _get_model()

    if not texts:
        return []

    # Manual batching so we can report progress without tqdm, which conflicts
    # with Rich's spinner and causes multi-line output in the terminal.
    batches = [texts[i:i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
    total = len(batches)
    results: list[np.ndarray] = []

    for i, batch in enumerate(batches):
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        results.append(batch_embeddings)
        if on_progress is not None:
            on_progress(i + 1, total)

    return np.concatenate(results).tolist()
