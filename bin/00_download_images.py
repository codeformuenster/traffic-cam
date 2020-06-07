#!/usr/bin/env python3
""" Download a number of images """

import time
import logging
from subprocess import CalledProcessError

from traffic_cam import io

logging.basicConfig(level=logging.INFO)

# parse terminal args
args = io.get_download_argparser().parse_args()
logging.info(f"args.n_images: {args.n_images}")
logging.info(f"args.sleep: {args.sleep}")

for i in range(args.n_images):
    # download image
    logging.info(f"Download image {i + 1} of {args.n_images}...")
    try:
        io.download_frame()
    except CalledProcessError as e:
        logging.error(f"Failed to download frame: {e}")
        continue
    logging.info(f"Downloaded image {i + 1} of {args.n_images}.")
    # sleep
    time.sleep(args.sleep)

logging.info("Finished.")
