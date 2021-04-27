import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib.pyplot import get_cmap
import os
import tqdm
import glob
import pathlib
from itertools import chain


def predict_image(
    img_pth,
    model,
    save_pth,
    vis_type="grayscale",
    thd=0.5,
) -> None:
    """
    Predict image and save to disc.

    :param img_pth: Path to RGB image.
    :param model: Prepared model to predict image.
    :param save_pth: Path to save prediction map.
    :param vis_type: Kind to visualize prediction ("grayscale", "heatmap", "binary").
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


def main():
    """
    Entry point for training.

    :return:
    """
    # Paths to files and model
    img_pth = "/data/hofstetter_data/bosch_testset/labeled/"
    img_type = "jpeg"
    save_pth = "/data/hofstetter_data/bosch_testset/prediction/binary/"
    model_pth = "model.h5"

    # Prediction options
    test_thresholds = True  # Prdict multiple thresholds
    vis_type = "heatmap"  # Select wether "heatmap", "grayscale", "binary"
    pgs = True  # Progress bar of prediction
    thd = 0.5  # Threshold for binary prediction

    model = tf.keras.models.load_model(model_pth)

    if test_thresholds:
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
            )
    else:
        predict_images(
            img_pth, model, save_pth, img_type, vis_type=vis_type, pgs=pgs, thd=thd
        )


if __name__ == "__main__":
    main()
