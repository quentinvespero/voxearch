import os
from pathlib import Path

import mlx_whisper

from src import ui

from src.config import TRANSCRIPTION_MODEL


def _is_hf_model_cached(hf_repo: str) -> bool:
    """Return True if the HuggingFace model weights are already in the local cache."""
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cache_name = "models--" + hf_repo.replace("/", "--")
    return (cache_dir / cache_name).exists()


def transcribe(
    audio_path: str,
    model: str = TRANSCRIPTION_MODEL,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> list[dict]:
    """
    Transcribe an audio file using mlx-whisper (Apple Silicon optimised).

    Args:
        audio_path:     Path to the .mp3 / .wav file.
        model:          HuggingFace repo for the MLX Whisper weights.
        language:       ISO 639-1 language code (e.g. "fr", "en").
                        Pass None to let Whisper auto-detect.
        initial_prompt: Optional context string prepended before decoding.
                        Useful for biasing Whisper toward domain-specific
                        vocabulary or proper nouns (e.g. "React, TypeScript, GraphQL").
                        It's a soft hint — Whisper may still produce other words.

    Returns:
        List of segment dicts, each with:
          - start (float): start time in seconds
          - end   (float): end time in seconds
          - text  (str):   transcribed text for that segment
    """
    cached = _is_hf_model_cached(model)
    if cached:
        ui.info(f"Loading model {model} …")
    else:
        ui.info(f"Downloading model {model} (first run — this may take a few minutes) …")

    # When the model is already cached, prevent huggingface_hub from making a
    # network request to check for updates (snapshot_download → api.repo_info
    # hits HF servers even with a complete local cache, which can hang).
    _prev_offline = os.environ.get("HF_HUB_OFFLINE")
    if cached:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model,
            language=language,
            initial_prompt=initial_prompt,
            verbose=False,
            # word_timestamps adds overhead and we don't need them
            word_timestamps=False,
        )
    finally:
        # Restore previous value so we don't leak the flag into other library calls
        if _prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = _prev_offline

    return [
        {
            "start": segment["start"],
            "end":   segment["end"],
            "text":  segment["text"].strip(),
        }
        for segment in result["segments"]
        # Drop empty segments that Whisper sometimes emits
        if segment["text"].strip()
    ]
