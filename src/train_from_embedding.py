"""
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

    Example for training a shallow classifier for a particular currency based on pre-computed embeddings.
    Currencies not in the list can be added using the pre-trained encoder example in ./src/train_encode.py
"""

import argparse

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

# List of allowable currency choices
currency_choices = [
    "AUD",
    "BRL",
    "CAD",
    "EUR",
    "GBP",
    "INR",
    "JPY",
    "MXN",
    "PKR",
    "SGD",
    "TRY",
    "USD",
    "NZD",
    "NNR",
    "MYR",
    "IDR",
    "PHP",
]


def parse_arguments():
    """Parses arguments for shallow classifier training.

    Returns:
        ArgumentParser: argparse parsed arguments.
    """
    # Parse arguments and load data
    parser = argparse.ArgumentParser(description="Train model from embeddings.")
    parser.add_argument(
        "--currency",
        "--c",
        type=str,
        choices=currency_choices,
        help="String of currency for which to train shallow classifier",
        required=True,
    )
    parser.add_argument(
        "--bsize",
        "--b",
        type=int,
        help="Batch size for shallow classifier",
        default=128,
    )
    parser.add_argument(
        "--epochs",
        "--e",
        type=int,
        help="Number of epochs for training shallow top classifier",
        default=25,
    )
    parser.add_argument(
        "--dpath",
        "--d",
        type=str,
        help="Path to .feather BankNote Net embeddings",
        default="../data/banknote_net.feather",
    )

    return parser.parse_args()


def main():
    """Trains shallow classifier using embeddings."""

    args = parse_arguments()
    CURRENCY = args.currency
    BATCH_SIZE = args.bsize
    NB_EPOCH = args.epochs
    PATH = args.dpath

    # load data from embeddings
    data = pd.read_feather(PATH)
    data = data[data.Currency == CURRENCY]  # Filter currency
    data = data.sample(frac=1)
    labels = data.pop(
        "Denomination"
    )  # Pop denomination as labels, after filtering for particular currency.
    labels = labels.astype("category")
    labels_encoded = pd.get_dummies(labels)
    data = data.iloc[:, :-1]  # Keep only embedding

    # Define dataset and shallow model
    NUM_CLASSES = len(labels.unique())
    NB_TRAINING_SAMPLES = len(data)

    input = Input(shape=(256,))
    x = Dense(128, activation="relu")(input)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=input, outputs=x)
    model.summary()

    checkpoint = ModelCheckpoint(
        filepath="./src/trained_models/shallow_classifier.h5",
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
        x=data.values,
        y=labels_encoded.values,
        steps_per_epoch=NB_TRAINING_SAMPLES // BATCH_SIZE,
        epochs=NB_EPOCH,
        validation_split=0.2,
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    main()
