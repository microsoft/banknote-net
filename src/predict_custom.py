"""
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

    Trains a model using images as input located in a custom folder and 
    the pre-trained banknote_net encoder network (MobileNet V2).
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    """Parses arguments for prediction.

    Returns:
        ArgumentParser: argparse parsed arguments.
    """
    # Parse arguments and load data
    parser = argparse.ArgumentParser(
        description="Perform inference using trained custom classifier."
    )
    parser.add_argument(
        "--bsize",
        "--b",
        type=int,
        help="Batch size",
        default=1,
    )
    parser.add_argument(
        "--data_path",
        "--data",
        type=str,
        help="Path to custom folder with validation images.",
        default="./data/example_images/SEK/val/",
    )
    parser.add_argument(
        "--model_path",
        "--enc",
        type=str,
        help="Path to .h5 file of a trained classification model",
        default="./src/trained_models/custom_classifier.h5",
    )

    return parser.parse_args()


def create_generator(
    VAL_PATH: str,
    IMG_SIZE: tuple,
    BATCH_SIZE: int = 2,
    NUM_CLASSES: int = 10,
):
    """Creates tensorflow datasets for custom directory

    Args:
        TRAIN_PATH (str): Train path for custom training data.
        VAL_PATH (str): Validation path for validation data.
        IMG_SIZE (tuple): Size of image in pixels, not including channels (224, 224)
        BATCH_SIZE (int, optional): Batch size. Defaults to 2.
        NUM_CLASSES (int, optional): Number of classes. Defaults to 10.

    Returns:
        train_ds, val_ds (tf.data.Dataset)
    """

    IMG_WIDTH, IMG_HEIGHT = IMG_SIZE

    # Prepare data generators, train generator has some data augmentation
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )
    validation_generator = test_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="categorical",
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3], [None, NUM_CLASSES]),
    )

    return val_ds


def main():
    """Trains classifier for custom class and data directory."""

    args = parse_arguments()
    BATCH_SIZE = args.bsize
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    NUM_CLASSES = 8
    IMG_SIZE = (224, 224)

    # Load datasets from embeddings
    val_ds = create_generator(
        VAL_PATH=f"{DATA_PATH}",
        IMG_SIZE=IMG_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        NUM_CLASSES=NUM_CLASSES,
    )

    # Load model and make predictions
    model = load_model(MODEL_PATH)

    predictions = model.predict(val_ds, batch_size=1, steps=15)
    predictions = np.argmax(predictions, axis=1)
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
