#!/usr/bin/env python3
""" Train image classifier to distinguish camera positions. """

import json

from traffic_cam import classifier, paths

import os


# data generators for training and validation
train_generator = classifier.get_image_datagen(folder=paths.TRAIN_DIR, batch_size=16)
valid_generator = classifier.get_image_datagen(folder=paths.VALID_DIR, batch_size=16)

# train model
classes = os.listdir(paths.TRAIN_DIR)
model = classifier.get_classifier(n_classes=len(classes))

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
