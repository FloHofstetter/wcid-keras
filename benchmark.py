#!/usr/bin/env python

import pathlib
import itertools

import tensorflow as tf

from predict import predict_images
from utils.confusion import BatchMetrics
from utils.confusion import Confusion
from utils.overlay import overlay_masks


def predict_heatmap(img_pth, img_ext, model, sve_pth):
    """

    :return: None
    """
    predict_images(img_pth, model, sve_pth, img_ext, vis_type="heatmap", pgs=True)


def predict_grayscale(img_pth, img_ext, model, sve_pth):
    """

    :return: None
    """
    predict_images(img_pth, model, sve_pth, img_ext, vis_type="grayscale", pgs=True)


def predict_binary(img_pth, img_ext, model, sve_pth):
    """
    Predict binary metrics online with model.

    :return: None
    """
    for threshold in itertools.chain(
        range(0, 10, 1), range(10, 100, 10), range(91, 101, 1)
    ):
        dst_pth = pathlib.Path(sve_pth).joinpath(f"{threshold}/")
        threshold /= 100
        predict_images(
            img_pth,
            model,
            dst_pth,
            img_ext,
            vis_type="binary",
            thd=threshold,
            pgs=True,
            pgs_txt=f"THD: {int(threshold * 100):03d}%",
        )


def calculate_confusion_metrics(bin_pth, gth_pth, gth_ext, bin_ext):
    """

    :return: None
    """
    best_thd = -1.0
    best_iou = -1.0
    thresholds = itertools.chain(range(0, 10, 1), range(10, 100, 10), range(91, 101, 1))
    for thd in thresholds:
        thd_pth = bin_pth.joinpath(str(thd))
        bm = BatchMetrics(thd_pth, gth_pth, gth_ext, bin_ext)
        # Store best iou
        if bm.iou > best_iou:
            best_iou = bm.iou
            best_thd = thd
    return best_thd


def confusion_images(thd, gt_pth, pd_pth, gt_ext, pd_ext, cnf_pth):
    """

    :param thd:
    :param gt_pth:
    :param pd_pth:
    :return:
    """
    pd_pth = pd_pth.joinpath(f"{thd}/")

    gt_pths = sorted(list(gt_pth.glob(f"*.{gt_ext}")))
    pd_pths = sorted(list(pd_pth.glob(f"*.{pd_ext}")))

    for pd, gt in zip(pd_pths, gt_pths):
        confusion = Confusion(gt, pd)
        filename = pd.name
        cnf_pth.mkdir(parents=True, exist_ok=True)
        sve_pth = cnf_pth.joinpath(filename)
        confusion.confusion_image().save(sve_pth)


def main():
    """
    Entry point for benchmark.

    :return: None.
    """
    # Path to RGB image dir and file extension
    # to collect files.
    img_pth = ""
    img_ext = "jpeg"
    img_pth = pathlib.Path(img_pth)

    # Path to ground truth dir and file extension
    # to collect files.
    gt_pth = ""
    gt_ext = "jpeg"
    gt_pth = pathlib.Path(gt_pth)

    # Path to model to test
    mdl_pth = ""
    mdl_pth = pathlib.Path(mdl_pth)

    # Path to save benchmark results
    bmk_pth = ""
    bmk_pth = pathlib.Path(bmk_pth)
    bmk_pth.mkdir(parents=True, exist_ok=True)

    # Derived benchmark results paths
    # Heatmap
    hmp_pth = bmk_pth.joinpath("heatmap/")
    hmp_pth.mkdir(parents=True, exist_ok=True)
    # Grayscale
    gsc_pth = bmk_pth.joinpath("grayscale/")
    gsc_pth.mkdir(parents=True, exist_ok=True)
    # Binary thresholds
    bin_pth = bmk_pth.joinpath("binary/")
    bin_pth.mkdir(parents=True, exist_ok=True)
    # Confusion images
    cnf_pth = bmk_pth.joinpath("confusion/")
    cnf_pth.mkdir(parents=True, exist_ok=True)
    # Overlay paths
    hmp_ovl_pth = bmk_pth.joinpath("heatmap_overlay/")
    hmp_ovl_pth.mkdir(parents=True, exist_ok=True)
    gsc_ovl_pth = bmk_pth.joinpath("grayscale_overlay/")
    gsc_ovl_pth.mkdir(parents=True, exist_ok=True)
    cnf_ovl_pth = bmk_pth.joinpath("confusion_overlay/")
    cnf_ovl_pth.mkdir(parents=True, exist_ok=True)

    # Open tensorflow neural network model incl. architecture.
    model = tf.keras.models.load_model(mdl_pth)

    # Initiate all benchmark processes
    # Predict heatmap, grayscale, and binary thresholds
    predict_heatmap(img_pth, img_ext, model, sve_pth=hmp_pth)
    predict_grayscale(img_pth, img_ext, model, sve_pth=gsc_pth)
    predict_binary(img_pth, img_ext, model, sve_pth=bin_pth)

    # Calculate binary metrics based on binary predictions
    # Predicted extension is in this case inherited from image extension
    best_thd = calculate_confusion_metrics(bin_pth, gt_pth, gt_ext, img_ext)

    # Store confusion images based on best binary metrics.

    # Calculate confusion images
    print("Confusion images... ", end="")
    confusion_images(best_thd, gt_pth, bin_pth, gt_ext, img_ext, cnf_pth)
    print("[done]")

    # Overlay benchmark images
    print("Heatmap overlay... ", end="")
    overlay_masks(img_pth, hmp_pth, hmp_ovl_pth, img_ext, img_ext)
    print("[done]")
    print("Grayscale overlay... ", end="")
    overlay_masks(img_pth, gsc_pth, gsc_ovl_pth, img_ext, img_ext)
    print("[done]")
    print("Confusion overlay... ", end="")
    overlay_masks(img_pth, cnf_pth, cnf_ovl_pth, img_ext, img_ext)
    print("[done]")


if __name__ == "__main__":
    main()
