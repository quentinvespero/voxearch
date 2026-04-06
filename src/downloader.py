import os
import yt_dlp


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
                print("      [cache] Audio file already on disk, skipping download.")
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
