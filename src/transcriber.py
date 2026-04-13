import mlx_whisper
from huggingface_hub import snapshot_download

from src import ui
from src.config import TRANSCRIPTION_MODEL


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
    # Try the local cache first (no network call).
    # local_files_only=True raises EnvironmentError if the snapshot is missing.
    try:
        model_path = snapshot_download(model, local_files_only=True)
        ui.info(f"Loading transcription model {model} …")
    except EnvironmentError:
        ui.info(f"Downloading transcription model {model} (~1.5 GB, first run) …")
        # etag_timeout=30 caps the initial connection check at 30 s instead of
        # hanging indefinitely when HuggingFace is unreachable.
        model_path = snapshot_download(model, etag_timeout=30)

    # Pass the resolved local path, not the repo ID.
    # mlx_whisper's load_model checks Path(path_or_hf_repo).exists() first and
    # skips its own snapshot_download when the path already exists — no network call.
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_path,
        language=language,
        initial_prompt=initial_prompt,
        verbose=False,
        # word_timestamps adds overhead and we don't need them
        word_timestamps=False,
    )

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
