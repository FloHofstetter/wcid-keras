import tensorflow as tf
import numpy as np
from tensorflow.keras.activations import sigmoid
from PIL import Image
from matplotlib.pyplot import get_cmap
import os
import tqdm
import glob


def predict_image(img_pth, model, save_pth, vis_type="grayscale", thd=0.5) -> None:
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

    # Calculate logits
    prd_t = sigmoid(prd_t)

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
        msg = (f'Unknown visualisation type. Expected "grayscale" or "heatmap" '
               + f'or "binary", got "{vis_type}".')
        raise ValueError(msg)

    # Save image with same name under save path
    file_name = os.path.basename(img_pth)
    save_pth = os.path.join(save_pth, file_name)
    prd_img = Image.fromarray(out_image_arr)
    prd_img = prd_img.resize(original_size)
    prd_img.save(save_pth)


def predict_images(imgs_pth, model, save_pth, vis_type="grayscale", thd=0.5, pgs=False) -> None:
    """
    Predict multiple images.

    :param imgs_pth:
    :param model: Prepared model to predict image.
    :param save_pth: Path to save prediction map.
    :param vis_type: Kind to visualize prediction ("grayscale", "heatmap", "binary").
    :param thd: Threshold for binary prediction.
    :param pgs: Progress bar.
    :return: None
    """
    imgs_pth = os.path.join(imgs_pth, "*png")
    img_pths = glob.glob(imgs_pth)
    for img_pth in tqdm.tqdm(img_pths, disable=(not pgs)):
        predict_image(img_pth, model, save_pth, vis_type=vis_type, thd=thd)


def main():
    """
    Entry point for training.

    :return:
    """
    img_pth = ""
    save_pth = ""
    model_pth = ""

    model = tf.keras.models.load_model(model_pth)
    predict_images(img_pth, model, save_pth, vis_type="binary", thd=0.75, pgs=True)


if __name__ == "__main__":
    main()
