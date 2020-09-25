""" Paths used throughout project. """

from pathlib import Path

DATA_DIR = Path("data/")
TRAIN_DIR = DATA_DIR / "train"

CLASSIFIER_DIR = Path("data/classifier/")
CLASSIFIER_HDF5 = CLASSIFIER_DIR / "model.hdf5"
CLASSES_JSON = CLASSIFIER_DIR / "class_indices.json"


def create_paths_if_not_exist():
    """Create defined paths if they don't exist already, including parents."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
