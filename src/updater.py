"""
Utilities for checking and updating the yt-dlp package.
"""

import importlib.metadata
import json
import subprocess
import urllib.request


def get_installed_version() -> str:
    return importlib.metadata.version("yt-dlp")


def get_latest_pypi_version() -> str:
    with urllib.request.urlopen("https://pypi.org/pypi/yt-dlp/json", timeout=10) as response:
        data = json.loads(response.read())
    return data["info"]["version"]


def update_ytdlp() -> None:
    installed = get_installed_version()
    print(f"Installed: yt-dlp {installed}")

    print("Checking PyPI for latest version...")
    latest = get_latest_pypi_version()
    print(f"Latest:    yt-dlp {latest}")

    if installed == latest:
        print("Already up to date.")
        return

    print(f"\nUpgrading yt-dlp {installed} → {latest} ...")
    subprocess.run(["uv", "pip", "install", "--upgrade", "yt-dlp"], check=True)
    print("Done.")
