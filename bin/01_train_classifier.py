#!/usr/bin/env python3
""" Train image classifier to distinguish camera positions. """

import json
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from traffic_cam import classifier, paths


SEED = 42
VALIDATION_SPLIT = 0.3

# data generators for training and validation
datagen = datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=VALIDATION_SPLIT
)
train_generator = datagen.flow_from_directory(
    paths.TRAIN_DIR,
    seed=SEED,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    subset="training",
)
valid_generator = datagen.flow_from_directory(
    paths.TRAIN_DIR,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    subset="validation",
)


# train model
model = classifier.get_classifier(n_classes=5)

step_size_train = train_generator.n // train_generator.batch_size
model.fit(
    x=train_generator,
    steps_per_epoch=step_size_train,
    epochs=10,
    validation_data=next(valid_generator),
)

# save class encoding from data generators
assert train_generator.class_indices == valid_generator.class_indices
with open(str(paths.CLASSES_JSON), "w") as f:
    f.write(json.dumps(train_generator.class_indices, indent=2))

# save model
model.save(str(paths.CLASSIFIER_HDF5))
