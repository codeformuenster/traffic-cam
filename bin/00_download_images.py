#!/usr/bin/env python3
""" Download a number of images """

import logging
import shutil
import time
from subprocess import CalledProcessError

from tensorflow.keras.models import load_model

from traffic_cam import io, paths, classifier

logging.basicConfig(level=logging.INFO)

# parse terminal args
args = io.get_download_argparser().parse_args()
logging.info(f"args.n_images: {args.n_images}")
logging.info(f"args.sleep: {args.sleep}")
logging.info(f"args.classify: {args.classify}")

if args.classify:
    model = load_model(str(paths.CLASSIFIER_HDF5))


for i in range(args.n_images):
    # download image
    logging.info(f"Download image {i + 1} of {args.n_images}...")
    try:
        timestamp = io.get_timestamp_isoformat()
        io.download_frame(suffix=timestamp)
    except CalledProcessError as e:
        logging.error(f"Failed to download frame: {e}")
        continue
    logging.info(f"Downloaded image {i + 1} of {args.n_images}.")
    # sort image
    if args.classify:
        filename = f"image_{timestamp}.jpg"
        source_path = paths.DATA_DIR / filename
        predictions = classifier.classify_image(filepath=source_path, model=model)
        predicted_class: str = classifier.get_predicted_class(predictions=predictions)
        logging.info(f"Sorting new image to class {predicted_class}.")
        target_path = paths.TRAIN_DIR / predicted_class / filename
        shutil.move(source_path, target_path)
    # sleep
    logging.info("Sleeping...")
    time.sleep(args.sleep)

logging.info("Finished.")
