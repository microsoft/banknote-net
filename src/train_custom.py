"""
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

    Trains a model using images as input located in a custom folder and 
    the pre-trained banknote_net encoder network (MobileNet V2). Saves the best model 
    in ./src/trained_models/
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    """Parses arguments for shallow classifier training.

    Returns:
        ArgumentParser: argparse parsed arguments.
    """
    # Parse arguments and load data
    parser = argparse.ArgumentParser(
        description="Train model from custom image folder using pre-trained BankNote-Net encoder."
    )
    parser.add_argument(
        "--bsize",
        "--b",
        type=int,
        help="Batch size",
        default=4,
    )
    parser.add_argument(
        "--epochs",
        "--e",
        type=int,
        help="Number of epochs for training shallow top classifier",
        default=25,
    )
    parser.add_argument(
        "--data_path",
        "--data",
        type=str,
        help="Path to folder with images.",
        default="../data/example_images/SEK/",
    )
    parser.add_argument(
        "--enc_path",
        "--enc",
        type=str,
        help="Path to .h5 file of pre-trained encoder model",
        default="../models/banknote_net_encoder.h5",
    )

    return parser.parse_args()


def create_generator(
    TRAIN_PATH: str,
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
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=180,
        channel_shift_range=40,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    # Initiliaze generators and create TF datasets
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=12345,
        class_mode="categorical",
    )
    validation_generator = test_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="categorical",
    )
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3], [None, NUM_CLASSES]),
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3], [None, NUM_CLASSES]),
    )

    return train_ds, val_ds


def main():
    """Trains classifier for custom class and data directory."""

    args = parse_arguments()
    BATCH_SIZE = args.bsize
    NB_EPOCH = args.epochs
    ENC_PATH = args.enc_path
    DATA_PATH = args.data_path
    NUM_CLASSES = len(next(os.walk(f"{DATA_PATH}/train/"))[1])
    IMG_SIZE = (224, 224)
    NB_TRAINING_SAMPLES = sum(
        [len(files) for r, d, files in os.walk(f"{DATA_PATH}/train/")]
    )
    NB_VALIDATION_SAMPLES = sum(
        [len(files) for r, d, files in os.walk(f"{DATA_PATH}/val/")]
    )

    # Load datasets from embeddings
    train_ds, val_ds = create_generator(
        TRAIN_PATH=f"{DATA_PATH}/train/",
        VAL_PATH=f"{DATA_PATH}/val/",
        IMG_SIZE=IMG_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        NUM_CLASSES=NUM_CLASSES,
    )

    # Load encoder model and freeze layers
    encoder = load_model(ENC_PATH)
    for layer in encoder.layers:
        layer.trainable = False

    input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = encoder(input)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=input, outputs=x)
    model.summary()

    # Define callbacks, compile and fit
    checkpoint = ModelCheckpoint(
        filepath="./src/trained_models/custom_classifier.h5",
        monitor="val_acc",
        save_best_only=True,
    )

    # Compile and fit
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        metrics=[
            "acc",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    model.fit(
        train_ds,
        steps_per_epoch=NB_TRAINING_SAMPLES // BATCH_SIZE,
        epochs=NB_EPOCH,
        validation_steps=NB_VALIDATION_SAMPLES // BATCH_SIZE + 1,
        validation_data=val_ds,
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    main()
