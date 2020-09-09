""" Utils for IO. """

import argparse
import datetime
import subprocess


def get_timestamp_isoformat() -> str:
    return datetime.datetime.utcnow().isoformat() + "+00:00"


def download_frame():
    suffix = get_timestamp_isoformat()
    download_command = f"""ffmpeg \
    -i https://5f27cc8163c2e.streamlock.net/833/default.stream/playlist.m3u8?wowzatokenhash=0mfLM7iDsbsXsvj91j1LqHWRrf2ZMRArPtr8efxJnjU= \
    -vframes 1 \
    data/image_{suffix}.jpg"""
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
    parser.add_argument(
        "-c",
        "--classify",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Classify downloaded images, and sort them to training set.",
    )
    return parser


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
