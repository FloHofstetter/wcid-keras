#!/usr/bin/env python

import os
import csv
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


def calculate_confusion_metrics(bin_pth, gth_pth, gth_ext, bin_ext, sve_pth):
    """

    :return: None
    """
    # Open metrics file if sve_pth exists
    if sve_pth is not None:
        sve_pth = pathlib.Path(sve_pth)
        sve_pth.mkdir(exist_ok=True, parents=True)
        sve_csv = sve_pth.joinpath("metrics.csv").open(mode="a")
        img_sve_csv = sve_pth.joinpath("image_metrics.csv").open(mode="a")
        fieldnames = ["thd", "iou", "f1", "acc", "prc", "rec", "tp", "fp", "fn", "tn"]
        csv_writer = csv.DictWriter(sve_csv, fieldnames=fieldnames)
        csv_writer.writeheader()

    best_thd = -1.0
    best_iou = -1.0
    thresholds = itertools.chain(range(0, 10, 1), range(10, 100, 10), range(91, 101, 1))
    for thd in thresholds:
        thd_pth = bin_pth.joinpath(str(thd))
        bm = BatchMetrics(thd_pth, gth_pth, gth_ext, bin_ext)
        # Save metrics to path
        if sve_pth is not None:
            metrics_dict = {
                "thd": thd,
                "iou": bm.iou,
                "f1": bm.f1,
                "acc": bm.acc,
                "prc": bm.prc,
                "rec": bm.rec,
                "tp": bm.tp,
                "fp": bm.fp,
                "fn": bm.fn,
                "tn": bm.tn,
            }
            csv_writer.writerow(metrics_dict)

        # Store best iou
        if bm.iou > best_iou:
            best_iou = bm.iou
            best_thd = thd
            # Store per image score for best batch thd.
            b_me = bm

    fieldnames = ["img", "iou", "f1", "acc", "prc", "rec", "tp", "fp", "fn", "tn"]
    img_csv_writer = csv.DictWriter(img_sve_csv, fieldnames=fieldnames)
    img_csv_writer.writeheader()

    for img, iou, f1, acc, prc, rec, tp, fp, fn, tn in zip(
        b_me.pd_pths,
        b_me.iou_list,
        b_me.f1_list,
        b_me.acc_list,
        b_me.prc_list,
        b_me.rec_list,
        b_me.tp_list,
        b_me.fp_list,
        b_me.fn_list,
        b_me.tn_list,
    ):
        metrics_dict = {
            "img": img,
            "iou": iou,
            "f1": f1,
            "acc": acc,
            "prc": prc,
            "rec": rec,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        img_csv_writer.writerow(metrics_dict)

    # Close files
    img_sve_csv.close()
    sve_csv.close()

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


def benchmark(img_pth, gt_pth, mdl_pth, sve_pth, img_ext="png", gt_ext="png"):
    """

    :param img_pth: Path to directory containing RGB images.
    :param gt_pth: Path to directory ground truth images.
    :param mdl_pth: Path to tensorflow model file.
    :param sve_pth: Path to save benchmark results.
    :param img_ext: Extension for image files.
    :param gt_ext: Extension for ground truth files.
    :return: None.
    """
    # Derived benchmark results paths
    # Heatmap
    hmp_pth = sve_pth.joinpath("heatmap/")
    hmp_pth.mkdir(parents=True, exist_ok=True)
    # Grayscale
    gsc_pth = sve_pth.joinpath("grayscale/")
    gsc_pth.mkdir(parents=True, exist_ok=True)
    # Binary thresholds
    bin_pth = sve_pth.joinpath("binary/")
    bin_pth.mkdir(parents=True, exist_ok=True)
    # Confusion images
    cnf_pth = sve_pth.joinpath("confusion/")
    cnf_pth.mkdir(parents=True, exist_ok=True)
    # Overlay paths
    hmp_ovl_pth = sve_pth.joinpath("heatmap_overlay/")
    hmp_ovl_pth.mkdir(parents=True, exist_ok=True)
    gsc_ovl_pth = sve_pth.joinpath("grayscale_overlay/")
    gsc_ovl_pth.mkdir(parents=True, exist_ok=True)
    cnf_ovl_pth = sve_pth.joinpath("confusion_overlay/")
    cnf_ovl_pth.mkdir(parents=True, exist_ok=True)

    # Open tensorflow neural network model incl. architecture.
    model = tf.keras.models.load_model(mdl_pth)

    # Initiate all benchmark processes.
    # Predict heatmap, grayscale, and binary thresholds
    predict_heatmap(img_pth, img_ext, model, sve_pth=hmp_pth)
    predict_grayscale(img_pth, img_ext, model, sve_pth=gsc_pth)

    # Overlay benchmark images
    print("Heatmap overlay... ", end="")
    overlay_masks(img_pth, hmp_pth, hmp_ovl_pth, img_ext, img_ext)
    print("[done]")
    print("Grayscale overlay... ", end="")
    overlay_masks(img_pth, gsc_pth, gsc_ovl_pth, img_ext, img_ext)
    print("[done]")

    # Calculate confusion images and metrics
    predict_binary(img_pth, img_ext, model, sve_pth=bin_pth)
    best_thd = calculate_confusion_metrics(bin_pth, gt_pth, gt_ext, img_ext, sve_pth)
    print("Confusion images... ", end="")
    confusion_images(best_thd, gt_pth, bin_pth, gt_ext, img_ext, cnf_pth)
    print("[done]")
    tf.keras.backend.clear_session()

    # Overlay
    print("Confusion overlay... ", end="")
    overlay_masks(img_pth, cnf_pth, cnf_ovl_pth, img_ext, img_ext)
    print("[done]")


def main():
    """
    Entry point for benchmark.

    :return: None.
    """
    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Path to RGB image dir and file extension to collect files.
    img_pth = ""
    img_ext = "png"
    img_pth = pathlib.Path(img_pth)

    # Path to ground truth dir and file extension to collect files.
    gt_pth = ""
    gt_ext = "png"
    gt_pth = pathlib.Path(gt_pth)

    # Collect Grid search directories
    gses_pth = ""
    gses_pth = pathlib.Path(gses_pth)
    gs_pths = sorted(list(gses_pth.iterdir()))

    # Iterate over grid searches
    for gs_pth in gs_pths:
        # There is only one sub folder named by creation date of experiment
        gs_pth = next(gs_pth.iterdir())

        # Path to model to benchmark
        mdl_pth = gs_pth.joinpath("checkpoints/end_model.h5")

        # Path to save benchmark results
        bmk_pth = gs_pth.joinpath("prd")
        bmk_pth.mkdir(parents=True, exist_ok=True)

        print(mdl_pth)

        # Start benchmark
        benchmark(img_pth, gt_pth, mdl_pth, bmk_pth, img_ext, gt_ext)


if __name__ == "__main__":
    main()
