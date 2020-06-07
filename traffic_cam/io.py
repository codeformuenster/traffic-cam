""" Utils for IO. """

import argparse
import datetime
import subprocess


def download_frame():
    timestamp: str = datetime.datetime.utcnow().isoformat() + "+00:00"
    download_command = f"""ffmpeg \
    -i https://56f2a99952126.streamlock.net/833/default.stream/playlist.m3u8 \
    -vframes 1 \
    data/image_{timestamp}.jpg"""
    subprocess.run(download_command.split(), check=True)


def get_download_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download images from live webcam to file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n ", "--n_images", help="number of images to download", type=int, default=15,
    )
    parser.add_argument(
        "-s",
        "--sleep",
        help="number of seconds to sleep between downloads",
        type=int,
        default=15,
    )
    return parser
