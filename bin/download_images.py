""" Download a number of images """

import argparse
import time
import logging

from traffic_cam import io

# setup logging
logging.basicConfig(level=logging.INFO)

# config
parser = argparse.ArgumentParser()
parser.add_argument(
    "n_images", help="number of images to download", type=int, default=15
)
parser.add_argument(
    "sleep", help="number of seconds to sleep between downloads", type=int, default=15
)
args = parser.parse_args()

logging.info(f"args.n_images: {args.n_images}")
logging.info(f"args.sleep: {args.sleep}")

# download images
for i in range(args.n_images):
    logging.info(f"Download image {i + 1} of {args.n_images}...")
    io.download_frame()
    logging.info(f"Sleeping {args.sleep} seconds...")
    time.sleep(args.sleep)

logging.info("Done.")
