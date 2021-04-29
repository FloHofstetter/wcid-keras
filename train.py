#!/usr/bin/env python

from dataset import RailDataset
from model import get_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import pathlib
import datetime


def train(
    trn_img,
    val_img,
    trn_msk,
    val_msk,
    sve_pth="output/",
    train_res=(320, 160),
    file_ext="png",
    lr=0.0001,
    bs=1,
    epochs=1,
):
    """
    Tran the model, save model and weights.

    :param trn_img: Path to training image folder.
    :param val_img: Path to validation image folder.
    :param trn_msk: Path to training masks.
    :param val_msk: Path to validation masks.
    :param sve_pth: Path to save model and other artifacts.
    :param train_res: Resolution to train the network. (width, height)
    :param file_ext: Extension for the image/mask data.
    :param lr: Learning rate for the adam solver.
    :param bs: Batch size for training and validation.
    :param epochs: Training epochs to iterate over dataset.
    :return: None.
    """
    # Create outputs artifacts path if not exists
    iso_time = datetime.datetime.now().isoformat(timespec="minutes")
    history_pth = pathlib.PurePath(sve_pth, f"{iso_time}", "history/")
    checkpoint_pth = pathlib.PurePath(sve_pth, f"{iso_time}", "checkpoints")
    pathlib.Path(history_pth).mkdir(parents=True, exist_ok=True)
    pathlib.Path(checkpoint_pth).mkdir(parents=True, exist_ok=True)

    last_model_pth = pathlib.PurePath(checkpoint_pth, "end_model.h5")
    history_pth = pathlib.PurePath(history_pth, "history.svg")
    checkpoint_pth = pathlib.PurePath(
        checkpoint_pth, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    )

    # Data generator
    trn_gen = RailDataset(
        trn_img, trn_msk, train_res, file_ext, file_ext, batch_size=bs
    )
    val_gen = RailDataset(
        val_img, val_msk, train_res, file_ext, file_ext, batch_size=bs
    )

    model = get_model(input_shape=train_res)

    # Compile model
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10,
        decay_rate=0.9,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    # Checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_pth, save_weights_only=False, verbose=1
    )

    # Train model
    history = model.fit(
        trn_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[cp_callback],
        workers=48,
        use_multiprocessing=False,
    )

    # Save Last model
    model.save(last_model_pth)

    # Show history
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(history_pth)


def parse_args(parser: argparse.ArgumentParser):
    """
    Parse CLI arguments to control the training.

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "-ti",
        "--train_images",
        type=pathlib.Path,
        help="Path to the folder containing RGB training images.",
        required=True,
    )
    parser.add_argument(
        "-tm",
        "--train_masks",
        type=pathlib.Path,
        help="Path to the folder containing training masks.",
        required=True,
    )
    parser.add_argument(
        "-vi",
        "--val_images",
        type=pathlib.Path,
        help="Path to the folder containing RGB validation images.",
        required=True,
    )
    parser.add_argument(
        "-vm",
        "--val_masks",
        type=pathlib.Path,
        help="Path to the folder containing validation masks.",
        required=True,
    )
    parser.add_argument(
        "-ex",
        "--extension",
        type=str,
        help="Name of the file extension. For example: <-e jpg>.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help=(
            "Path to the folder where to store model path and "
            + "other training artifacts."
        ),
        default="output/",
        required=False,
    )
    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        help="Training epochs.",
        default=10,
        required=False,
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate for the adam solver.",
        default=0.0001,
        required=False,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size of training and validation.",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        help="Select the GPU id to train on.",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of the neural network input layer.",
        default=160,
        required=False,
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="Width of the neural network input layer.",
        default=320,
        required=False,
    )

    return parser.parse_args()


def main():
    """
    Entry point for training.
    """
    # Parse arguments from cli
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    trn_img = args.train_images
    val_img = args.val_images
    trn_msk = args.train_masks
    val_msk = args.val_masks
    train_res = (args.width, args.height)  # Width height
    lr = args.learning_rate
    bs = args.batch_size
    epochs = args.epochs
    gpu = args.gpu
    sve_pth = args.output

    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    train(
        trn_img,
        val_img,
        trn_msk,
        val_msk,
        sve_pth=sve_pth,
        epochs=epochs,
        train_res=train_res,
        lr=lr,
        bs=bs,
    )


if __name__ == "__main__":
    main()
