""" Download a number of images """

import argparse
import time
import logging

from traffic_cam import io

# setup logging
logging.basicConfig(level=logging.INFO)

# config
# parser = argparse.ArgumentParser()
# parser.add_argument("sleep", help="how many seconds to sleep between downloads",
#                     type=int, default=15)
# args = parser.parse_args()
# print(args.square**2)

SECONDS_SLEEP: int = 15
N_IMAGES: int = 4

# download images
for i in range(N_IMAGES):
    logging.info(f"Download image {i + 1} of {N_IMAGES}...")
    io.download_frame()
    logging.info(f"Sleeping {SECONDS_SLEEP} seconds...")
    time.sleep(SECONDS_SLEEP)

logging.info("Done.")
