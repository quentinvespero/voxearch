import os
import yt_dlp

from src import ui


def fetch_playlist_entries(url: str) -> list[dict] | None:
    """
    Check whether *url* points to a playlist or feed.

    Returns None if the URL resolves to a single item, or a list of entry
    dicts (keys: id, title, url, duration) if it is a playlist/feed.
    Uses extract_flat mode — no audio is downloaded.
    """
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries")
    if not entries:
        return None

    result = []
    for entry in entries:
        result.append({
            "id":       entry.get("id", ""),
            "title":    entry.get("title") or entry.get("id") or "Untitled",
            # With extract_flat, "url" is the per-item URL suitable for ingest()
            "url":      entry.get("url") or entry.get("webpage_url", ""),
            "duration": entry.get("duration"),
        })
    return result


def download_audio(url: str, output_dir: str, force: bool = False) -> dict:
    """
    Download audio from a URL using yt-dlp and convert it to MP3.

    If the file already exists on disk and force=False, the download is skipped
    and the cached file is returned immediately.

    Returns a dict with:
      - title       (str):      human-readable title from the source
      - url         (str):      original URL
      - file_path   (str):      path to the downloaded .mp3 file
      - description (str|None): episode/video description, if available
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        # Download the best available audio stream
        "format": "bestaudio/best",
        # Convert to MP3 via ffmpeg (must be installed: brew install ffmpeg)
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # Use the video/episode ID as the filename so the path is predictable
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        # Suppress progress bars — we print our own status in the pipeline
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Fetch metadata without downloading to check for a cached file.
        # prepare_filename() applies the same sanitization yt-dlp uses when
        # writing to disk, so the derived path is reliable even for RSS feeds
        # whose IDs contain special characters.
        info = ydl.extract_info(url, download=False)

        if not force:
            raw_path = ydl.prepare_filename(info)
            expected_mp3 = os.path.splitext(raw_path)[0] + ".mp3"
            if os.path.isfile(expected_mp3):
                ui.info("[cache] Audio file already on disk, skipping download.")
                return {
                    "title": info.get("title", "unknown"),
                    "url": url,
                    "file_path": expected_mp3,
                    "description": info.get("description"),
                }

        info = ydl.extract_info(url, download=True)

    # Use the actual filepath reported by yt-dlp after post-processing.
    # Constructing it manually from info['id'] is unreliable — for podcast RSS
    # feeds the id can contain URL query parameters, producing an invalid path.
    file_path = info["requested_downloads"][-1]["filepath"]

    return {
        "title": info.get("title", "unknown"),
        "url": url,
        "file_path": file_path,
        "description": info.get("description"),
    }
