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
    aug_prm,
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
    :param aug_prm: Dict of augmentation parameters.
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
    history_pth2 = history_pth.joinpath("history2.svg")
    history_pth3 = history_pth.joinpath("history3.csv")
    history_pth = history_pth.joinpath("history.svg")
    checkpoint_pth = pathlib.PurePath(
        checkpoint_pth, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    )

    # Data generator
    trn_gen = RailDataset(
        trn_img,
        trn_msk,
        train_res,
        file_ext,
        file_ext,
        batch_size=bs,
        tfs_prb=aug_prm,
    )
    val_gen = RailDataset(
        val_img,
        val_msk,
        train_res,
        file_ext,
        file_ext,
        batch_size=bs,
        transforms=False,
    )

    model = get_model(input_shape=train_res)
    model.load_weights(
        "/data/hofstetter_data/full_dataset_with_nlb/prd/mark_320x160/2021-05-19T01:31/checkpoints/end_model.h5"
    )

    # Compile model
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=int(5 * 2940),  # 5 Epochs
        decay_rate=0.8,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    # Checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_pth, save_weights_only=False, verbose=1
    )

    # Train model
    history = model.fit_generator(
        trn_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[cp_callback],
        workers=12,
        use_multiprocessing=True,
        max_queue_size=100,
    )

    # Save Last model
    model.save(last_model_pth)

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(history_pth3)

    # Show history
    hist_df.plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(history_pth)

    # Show history log
    hist_df[["val_loss", "acc_loss"]].plot(figsize=(8, 5), logy=True)
    plt.grid(True)
    plt.savefig(history_pth2)


def parse_args(parser: argparse.ArgumentParser):
    """
    Parse CLI arguments to control the training.

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "train_images",
        type=pathlib.Path,
        help="Path to the folder containing RGB training images.",
    )
    parser.add_argument(
        "train_masks",
        type=pathlib.Path,
        help="Path to the folder containing training masks.",
    )
    parser.add_argument(
        "val_images",
        type=pathlib.Path,
        help="Path to the folder containing RGB validation images.",
    )
    parser.add_argument(
        "val_masks",
        type=pathlib.Path,
        help="Path to the folder containing validation masks.",
    )
    parser.add_argument(
        "extension",
        type=str,
        help="Name of the file extension. For example: '-e jpg''.",
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
    parser.add_argument(
        "--horizontal_flip",
        type=float,
        help="Probability of flipping image in training set. Default is 0.5.",
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--brightness_contrast",
        type=float,
        help="Probability of applying random brightness contrast on image in training set. Default is 0.2.",
        default=0.2,
        required=False,
    )
    parser.add_argument(
        "--rotation",
        type=float,
        help="Probability of applying random rotation on image in training set. Default is 0.9.",
        default=0.9,
        required=False,
    )
    parser.add_argument(
        "--motion_blur",
        type=float,
        help="Probability of applying motion blur on image in training set. Default is 0.1.",
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--background_swap",
        type=float,
        help="Probability of applying background swap on image in training set. Default is 0.9.",
        default=0.9,
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

    # Path parameters
    trn_img = args.train_images
    val_img = args.val_images
    trn_msk = args.train_masks
    val_msk = args.val_masks
    sve_pth = args.output

    # Hardware parameters
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[gpu], True)

    # Training parameters
    train_res = (args.width, args.height)  # Width height
    lr = args.learning_rate
    bs = args.batch_size
    epochs = args.epochs

    # Augmentation parameters
    aug_prm = {
        "HorizontalFlip": args.horizontal_flip,
        "RandomBrightnessContrast": args.brightness_contrast,
        "Rotate": args.rotation,
        "MotionBlur": args.motion_blur,
        "BackgroundSwap": args.background_swap,
    }
    print(aug_prm)
    train(
        trn_img,
        val_img,
        trn_msk,
        val_msk,
        aug_prm,
        sve_pth=sve_pth,
        epochs=epochs,
        train_res=train_res,
        lr=lr,
        bs=bs,
    )


if __name__ == "__main__":
    main()
