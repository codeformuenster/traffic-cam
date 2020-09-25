""" Utils for classifier model. """

import json
from pathlib import Path
from typing import Dict

import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


from traffic_cam import paths


def get_classifier(n_classes: int, learning_rate: float) -> Model:
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


def classify_image(filepath: Path, model: Model) -> np.ndarray:
    # preprocess image
    img = image.load_img(str(filepath), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # predict with model
    return model.predict(x)


def load_class_indices() -> Dict[str, int]:
    with open(paths.CLASSES_JSON, "r") as f:
        classes: Dict[str, int] = json.loads(f.read())
    return classes


def get_predicted_class(predictions: np.ndarray) -> str:
    classes = load_class_indices()
    return list(classes.keys())[np.argmax(predictions)]
