""" Utils for IO. """

import argparse
import datetime
import logging
import subprocess
from pathlib import Path
from typing import Optional

from traffic_cam import paths


def get_timestamp_isoformat() -> str:
    return datetime.datetime.utcnow().isoformat()


def download_frame(path: str):
    download_command = f"""ffmpeg \
    -i https://5f27cc8163c2e.streamlock.net/833/default.stream/playlist.m3u8?wowzatokenhash=0mfLM7iDsbsXsvj91j1LqHWRrf2ZMRArPtr8efxJnjU= \
    -vframes 1 \
    {path}"""
    subprocess.run(download_command.split(), check=True)


def download_image() -> Optional[Path]:
    """Download current image and return it's path."""
    timestamp = get_timestamp_isoformat()
    filename = f"image_{timestamp}.jpg"
    path = paths.DATA_DIR / filename
    try:
        download_frame(path=path)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download frame: {e}")
        return None
    logging.info(f"Downloaded image: {filename}.")
    return path


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
