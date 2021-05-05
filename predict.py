#!/usr/bin/env python

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
# import tensorflow as tf

import numpy as np
from PIL import Image
from matplotlib.pyplot import get_cmap
import tqdm
import glob
import pathlib
from itertools import chain
import argparse


def predict_image(
    img_pth,
    model,
    save_pth,
    vis_type="grayscale",
    out_size=None,
    thd=0.5,
) -> None:
    """
    Predict image and save to disc.

    :param img_pth: Path to RGB image.
    :param model: Prepared model to predict image.
    :param save_pth: Path to save prediction map.
    :param vis_type: Kind to visualize prediction ("grayscale", "heatmap", "binary").
    :param out_size: Res of the output image (width, height).
    :param thd: Threshold for binary prediction.
    :return: None.
    """
    # Open Image
    img = Image.open(img_pth)
    # Save original size to later resize
    original_size = img.size
    # Resize image
    img = img.resize((320, 160))
    # Convert to array
    img_arr = np.asarray(img)
    img_arr = np.copy(img_arr)
    # Normalize and float datatype
    img_arr = img_arr / 255.0
    # Add dimension to imitate batch
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predict label
    prd_t = model.predict(img_arr)

    # Convert tensor to numpy array
    prd_arr = prd_t
    # Lower dimension (get rid of batch and image channel)
    prd_arr = np.squeeze(prd_arr, axis=(0, 3))

    # Visualize prediction
    if vis_type == "grayscale":
        out_image_arr = prd_arr * 255
        out_image_arr = out_image_arr.astype(np.uint8)
    elif vis_type == "heatmap":
        colormap = get_cmap("jet")
        out_image_arr = colormap(prd_arr)[:, :, :3] * 255
        out_image_arr = out_image_arr.astype(np.uint8)
    elif vis_type == "binary":
        out_image_arr = prd_arr > thd
    else:
        msg = (
            f'Unknown visualisation type. Expected "grayscale" or "heatmap" '
            + f'or "binary", got "{vis_type}".'
        )
        raise ValueError(msg)

    # Make the save path if not exists
    pathlib.Path(save_pth).mkdir(parents=True, exist_ok=True)
    # Save image with same name under save path
    file_name = os.path.basename(img_pth)
    save_pth = os.path.join(save_pth, file_name)
    prd_img = Image.fromarray(out_image_arr)
    if out_size is None:
        prd_img = prd_img.resize(original_size)
    prd_img.save(save_pth)


def predict_images(
    imgs_pth,
    model,
    save_pth,
    img_type="png",
    vis_type="heatmap",
    thd=0.5,
    pgs=False,
    pgs_txt=None,
    out_size=None,
) -> None:
    """
    Predict multiple images.

    :param imgs_pth:
    :param model: Prepared model to predict image.
    :param save_pth: Path to save prediction map.
    :param img_type: Image file extension.
    :param vis_type: Kind to visualize prediction ("grayscale", "heatmap", "binary").
    :param thd: Threshold for binary prediction.
    :param pgs: Progress bar.
    :param pgs_txt: Progress bar text.
    :param out_size: Res of the output image (width, height).
    :return: None
    """
    imgs_pth = os.path.join(imgs_pth, f"*.{img_type}")
    img_pths = glob.glob(imgs_pth)
    pgs_txt = vis_type if pgs_txt == None else pgs_txt
    for img_pth in tqdm.tqdm(
        img_pths,
        disable=(not pgs),
        desc=pgs_txt,
        unit=" Images",
        colour="green",
        ncols=100,
    ):
        predict_image(img_pth, model, save_pth, vis_type=vis_type, thd=thd)


def parse_args(parser: argparse.ArgumentParser):
    """
    Parse CLI arguments to control the prediction.

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="Path to the folder with the RGB images to be processed.",
    )
    parser.add_argument(
        "extension",
        type=str,
        help="Name of the file extension. For example: <-e jpg>.",
    )
    parser.add_argument(
        "model",
        type=pathlib.Path,
        help="Path to the architecture/model file.",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="Path to folder in which the segmented images are to be stored.",
    )
    parser.add_argument(
        "-v",
        "--vistype",
        type=str,
        help="Visualisation type. Default is grayscale.",
        choices=["grayscale", "heatmap", "binary"],
        default="grayscale",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Threshold for binary classification. Default is 0.5.",
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "-mt",
        "--multiple-thresholds",
        action="store_true",
        help="Store all thresholds from 0-10 in 1, 10-100 in 10, 90-100 in 1 steps.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Show progress bar on stdout.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of the output image.",
        default=160,
        required=False,
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="Width of the output image.",
        default=320,
        required=False,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        help="Select the GPU id to predict on.",
        default=0,
        required=False,
    )

    return parser.parse_args()


def main():
    """
    Entry point for training.

    :return:
    """
    # Disable tensorflow debugging information
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Parse arguments from cli
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    # Paths to files and model
    img_pth = args.input
    img_type = args.extension
    save_pth = args.output
    model_pth = args.model

    # Prediction options
    test_thresholds = args.multiple_thresholds  # Predict multiple thresholds
    vis_type = args.vistype  # Select weather "heatmap", "grayscale", "binary"
    pgs = args.progress  # Progress bar of prediction
    thd = args.threshold  # Threshold for binary prediction
    width = args.width
    height = args.height
    gpu = args.gpu

    # Select GPU to predict on
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    model = keras.models.load_model(model_pth)

    if test_thresholds:
        # Loop over thresholds
        for threshold in chain(range(1, 10, 1), range(10, 100, 10), range(91, 101, 1)):
            dst_pth = os.path.join(save_pth, f"{threshold}")
            threshold /= 100
            predict_images(
                img_pth,
                model,
                dst_pth,
                img_type,
                vis_type="binary",
                thd=threshold,
                pgs=pgs,
                pgs_txt=f"THD: {int(threshold * 100):03d}%",
                out_size=(width, height),
            )
    else:
        predict_images(
            img_pth,
            model,
            save_pth,
            img_type,
            vis_type=vis_type,
            pgs=pgs,
            thd=thd,
            out_size=(width, height),
        )


if __name__ == "__main__":
    main()
