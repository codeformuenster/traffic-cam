""" Utils for IO. """

import datetime
import subprocess


def download_frame():
    timestamp: str = datetime.datetime.utcnow().isoformat() + "+00:00"
    download_command = f"""ffmpeg \
    -i https://56f2a99952126.streamlock.net/833/default.stream/playlist.m3u8 \
    -vframes 1 \
    data/image_{timestamp}.jpg"""
    subprocess.run(download_command.split(), check=True)
