""" Utils for classifier model. """

import json
import logging
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import wget
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

from traffic_cam import paths


def train_classifier_model(n_classes: int, learning_rate: float) -> Model:
    # build model
    base_model = MobileNet(
        weights="imagenet", include_top=False
    )  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # we add dense layers so that the model can learn more complex functions
    # and classify for better results.
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)  # dense layer 2
    x = Dense(512, activation="relu")(x)  # dense layer 3
    preds = Dense(n_classes, activation="softmax")(
        x
    )  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)

    # set only final layers trainable
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    # compile
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_classifier_model() -> Model:
    """Download image classifier (if not exists), and load to memory."""
    if not paths.CLASSIFIER_HDF5.exists():
        wget.download(
            url="https://github.com/codeformuenster/traffic-cam-data/blob/master/model/model.hdf5?raw=true",
            out=str(paths.CLASSIFIER_HDF5),
        )
    return load_model(str(paths.CLASSIFIER_HDF5))


def classify_image(filepath: Path, model: Model) -> np.ndarray:
    # preprocess image
    img = image.load_img(str(filepath), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # predict with model
    return model.predict(x)


def move_image_by_class(source_path: Path, model: Model):
    predictions = classify_image(filepath=source_path, model=model)
    predicted_class: str = get_predicted_class(predictions=predictions)
    target_dir = paths.TRAIN_DIR / predicted_class
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(source_path, target_dir / source_path.name)
    logging.info(f"Moved image: {source_path.name}.")
    return predicted_class


def load_class_indices() -> Dict[str, int]:
    with open(paths.CLASSES_JSON, "r") as f:
        classes: Dict[str, int] = json.loads(f.read())
    return classes


def get_predicted_class(predictions: np.ndarray) -> str:
    classes = load_class_indices()
    return list(classes.keys())[np.argmax(predictions)]
