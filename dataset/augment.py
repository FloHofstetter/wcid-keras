import pathlib
import random

import cv2
import numpy as np


def augment_images(
    fg_arr: np.ndarray,
    msk_arr: np.ndarray,
    bg_dir_pth: pathlib.Path,
    bg_ext: str,
):
    """
    Blend in random background image.
    Separation of foreground and background is defined
    by the mask.

    :param fg_arr: Foreground image as numpy array.
    :param msk_arr: Mask image as numpy array.
    :param bg_dir_pth: Path to background image directory.
    :param bg_ext: Background image file extension.
    :return: Augmented image as numpy array.
    """
    # Collect available background images.
    bg_pths = list(bg_dir_pth.glob(f"*.{bg_ext}"))

    # Choose random background and open
    # and resize to foreground.
    bg_pth = random.choice(bg_pths)
    bg_arr = cv2.imread(str(bg_pth))
    fg_size = tuple(reversed(fg_arr.shape[:2]))
    bg_arr = cv2.resize(bg_arr, fg_size)

    # Invert mask and spread contrast
    msk_arr = 255 - msk_arr * 255

    # Calculate distance to label and min max scale
    # distance.
    dist_arr, _ = cv2.distanceTransformWithLabels(msk_arr, cv2.DIST_L2, 5)
    dist_arr = (dist_arr - dist_arr.min()) / (dist_arr.max() - dist_arr.min())
    dist_arr = np.stack([dist_arr] * 3, axis=2)

    # Normalize foreground and background.
    bg_arr = bg_arr / 255.0
    fg_arr = fg_arr / 255.0

    # Blend foreground and background.
    dist_arr = 1 - (1 - dist_arr) ** 3
    blended = (1 - dist_arr) * fg_arr + dist_arr * bg_arr

    # Back to uint8 Range.
    blended = (blended * 255).astype(np.uint8)

    return blended


def main():
    # Path to images.
    fg_pth = pathlib.Path("")
    bg_pth = pathlib.Path("")
    bg_ext = "jpg"
    msk_pth = pathlib.Path("")

    # Open images.
    fg_arr = cv2.imread(str(fg_pth))
    msk_arr = cv2.imread(str(msk_pth), cv2.IMREAD_GRAYSCALE)

    # Augment images.
    blended = augment_images(fg_arr, msk_arr, bg_pth, bg_ext)

    # Save to disk.
    cv2.imwrite("blended.png", blended)


if __name__ == "__main__":
    main()
