#!/usr/bin/env python3
""" Download a user-defined number of images. """

import argparse
import logging
import shutil
import time
from subprocess import CalledProcessError
from typing import Optional

from traffic_cam import io, paths, classifier

logging.basicConfig(level=logging.INFO)


def main():
    # parse terminal args
    parser = argparse.ArgumentParser(
        description="Download images from live webcam to file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n ",
        "--n_images",
        help="number of images to download",
        type=int,
        default=15,
    )
    parser.add_argument(
        "-s",
        "--sleep",
        help="number of seconds to sleep between downloads",
        type=int,
        default=15,
    )
    args = parser.parse_args()
    logging.info(f"args.n_images: {args.n_images}")
    logging.info(f"args.sleep: {args.sleep}")

    # prepare paths to write to
    paths.create_paths_if_not_exist()

    model = classifier.get_classifier_model()

    # download and classify N images
    for i in range(args.n_images):
        # download image
        logging.info(f"Download image {i + 1} of {args.n_images}...")

        image_path = io.download_image()
        if image_path is None:
            logging.info("No image downloaded.")
            continue

        location = classifier.move_image_by_class(image_path, model)

        # sleep
        logging.info("Sleeping...")
        time.sleep(args.sleep)

    logging.info("Finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
