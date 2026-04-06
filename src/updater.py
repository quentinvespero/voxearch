"""
Utilities for checking and updating the yt-dlp package.
"""

import importlib.metadata
import json
import subprocess
import urllib.request

from src import ui


def get_installed_version() -> str:
    return importlib.metadata.version("yt-dlp")


def get_latest_pypi_version() -> str:
    with urllib.request.urlopen("https://pypi.org/pypi/yt-dlp/json", timeout=10) as response:
        data = json.loads(response.read())
    return data["info"]["version"]


def update_ytdlp() -> None:
    installed = get_installed_version()
    ui.info(f"Installed: yt-dlp {installed}")

    ui.info("Checking PyPI for latest version…")
    latest = get_latest_pypi_version()
    ui.info(f"Latest:    yt-dlp {latest}")

    if installed == latest:
        ui.success("Already up to date.")
        return

    ui.info(f"Upgrading yt-dlp {installed} → {latest} …")
    subprocess.run(["uv", "pip", "install", "--upgrade", "yt-dlp"], check=True)
    ui.success("Done.")
