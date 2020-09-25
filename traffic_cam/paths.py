""" Paths used throughout project. """

from pathlib import Path

DATA_DIR = Path("data/")
TRAIN_DIR = Path("data/") / "train"
VALID_DIR = Path("data/") / "valid"

CLASSIFIER_DIR = Path("data/classifier/")
CLASSIFIER_HDF5 = CLASSIFIER_DIR / "model.hdf5"
CLASSES_JSON = CLASSIFIER_DIR / "class_indices.json"
