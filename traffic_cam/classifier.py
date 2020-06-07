""" Utils for classifier model. """

from pathlib import Path

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.directory_iterator import DirectoryIterator


def get_classifier(n_classes: int) -> Model:
    # build model
    base_model = MobileNet(
        weights="imagenet", include_top=False
    )  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(
        x
    )  # we add dense layers so that the model can learn more complex functions and classify for better results.
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
    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def get_image_datagen(folder: Path, batch_size: int) -> DirectoryIterator:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        str(folder),
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        shuffle=True,
    )
    return generator
