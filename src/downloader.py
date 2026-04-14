import hashlib
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
        "socket_timeout": 15,  # fail fast instead of hanging indefinitely
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


def _parse_upload_date(raw: str | None) -> str | None:
    # yt-dlp returns "YYYYMMDD" → convert to ISO "YYYY-MM-DD"
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return None


def download_audio(url: str, output_dir: str, force: bool = False) -> dict:
    """
    Download audio from a URL using yt-dlp and convert it to MP3.

    If the file already exists on disk and force=False, the download is skipped
    and the cached file is returned immediately.

    Returns a dict with:
      - title          (str):      human-readable title from the source
      - url            (str):      original URL
      - file_path      (str):      path to the downloaded .mp3 file
      - description    (str|None): episode/video description, if available
      - upload_date    (str|None): publication date as "YYYY-MM-DD", if available
      - season_number  (int|None): podcast season number, if available
      - episode_number (int|None): podcast episode number, if available
    """
    os.makedirs(output_dir, exist_ok=True)

    # Hash the URL to get a fixed-length filename. Podcast RSS feeds (e.g. Acast)
    # can produce IDs that are full URLs with query params, making %(title)s [%(id)s]
    # filenames hundreds of bytes long and causing ENAMETOOLONG on macOS.
    # MD5 hex digest is always 32 ASCII chars, and it's deterministic so the cache
    # check below still works.
    url_hash = hashlib.md5(url.encode()).hexdigest()

    ydl_opts = {
        # Download the best available audio stream
        "format": "bestaudio/best",
        # Convert to MP3 via ffmpeg (must be installed: brew install ffmpeg)
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, f"{url_hash}.%(ext)s"),
        # Suppress progress bars — we print our own status in the pipeline
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,  # fail fast instead of hanging indefinitely
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Fetch metadata without downloading to check for a cached file.
        info = ydl.extract_info(url, download=False)

        if not force:
            expected_mp3 = os.path.join(output_dir, f"{url_hash}.mp3")
            if os.path.isfile(expected_mp3):
                ui.info("[cache] Audio file already on disk, skipping download.")
                return {
                    "title":          info.get("title", "unknown"),
                    "url":            url,
                    "file_path":      expected_mp3,
                    "description":    info.get("description"),
                    "upload_date":    _parse_upload_date(info.get("upload_date")),
                    "season_number":  info.get("season_number"),
                    "episode_number": info.get("episode_number"),
                }

        info = ydl.extract_info(url, download=True)

    file_path = info["requested_downloads"][-1]["filepath"]

    return {
        "title":          info.get("title", "unknown"),
        "url":            url,
        "file_path":      file_path,
        "description":    info.get("description"),
        "upload_date":    _parse_upload_date(info.get("upload_date")),
        "season_number":  info.get("season_number"),
        "episode_number": info.get("episode_number"),
    }
