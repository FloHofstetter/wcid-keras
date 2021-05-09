from typing import Union, List, Tuple
import pathlib
import random

import cv2
import numpy as np
from PIL import Image


def augment_images(
    fg_arr: np.ndarray,
    msk_arr: np.ndarray,
    bg_dir_pth: Union[pathlib.Path, str],
    bg_ext: str,
    p: float = 1.0,
) -> np.ndarray:
    """
    Blend in random background image.
    Separation of foreground and background is defined
    by the mask.
    Colored image information is expected and provided in
    RGBC order.

    :param fg_arr: Foreground image as numpy array.
    :param msk_arr: Mask image as numpy array.
    :param bg_dir_pth: Path to background image directory.
    :param bg_ext: Background image file extension.
    :param p: Probability to apply augmentation.
    :return: Augmented image as numpy array.
    """
    # Random number to decide if augmentation is applied.
    rand_nr: float = random.random()

    # Apply augmentation.
    if rand_nr < p:
        # Get path if string provided.
        bg_dir_pth = pathlib.Path(bg_dir_pth)
        # Convert RGB input to cv2 native color order
        fg_arr: np.ndarray = cv2.cvtColor(fg_arr, cv2.COLOR_RGB2BGR)

        # Collect available background images.
        bg_pths: List = list(bg_dir_pth.glob(f"*.{bg_ext}"))

        # Choose random background and open
        # and resize to foreground.
        bg_pth: Union[pathlib.Path, str]
        bg_pth = random.choice(bg_pths)
        bg_arr: np.ndarray = cv2.imread(str(bg_pth))
        fg_size: Tuple = tuple(reversed(fg_arr.shape[:2]))
        bg_arr = cv2.resize(bg_arr, fg_size)

        # Invert mask and spread contrast
        msk_arr = 255 - msk_arr * 255

        # Calculate distance to label and min max scale
        # distance.
        dist_arr: np.ndarray
        dist_arr, _ = cv2.distanceTransformWithLabels(msk_arr, cv2.DIST_L2, 5)
        # TODO: Clean solution for zero division case.
        if (dist_arr.max() - dist_arr.min()) > 0:
            dist_arr = (dist_arr - dist_arr.min()) / (dist_arr.max() - dist_arr.min())
        else:
            print("Dist_arr < 0!")
        dist_arr = np.stack([dist_arr] * 3, axis=2)

        # Normalize foreground and background.
        bg_arr = bg_arr / 255.0
        fg_arr = fg_arr / 255.0

        # Blend foreground and background.
        dist_arr = 1 - (1 - dist_arr) ** 3
        blended: np.ndarray = (1 - dist_arr) * fg_arr + dist_arr * bg_arr

        # Back to uint8 range.
        blended = (blended * 255).astype(np.uint8)

        # Convert cv2 native color order to RGB
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    # Do not apply augmentation
    else:
        blended = fg_arr

    return blended


def main() -> None:
    """
    Program entry point.

    :return: None
    :rtype: None
    """
    # Path to images.
    fg_pth: pathlib.Path = pathlib.Path("")
    bg_pth: pathlib.Path = pathlib.Path("")
    bg_ext: str = "jpg"
    msk_pth: pathlib.Path = pathlib.Path("")

    # Open images.
    fg_img: Image.Image = Image.open(fg_pth)
    msk_img: Image.Image = Image.open(msk_pth)
    fg_arr: np.ndarray = np.asarray(fg_img)
    msk_arr: np.ndarray = np.asarray(msk_img)

    # Augment images.
    blended: np.ndarray = augment_images(fg_arr, msk_arr, bg_pth, bg_ext)

    # Show blended image.
    blended_img: Image.Image = Image.fromarray(blended)
    blended_img.show()


if __name__ == "__main__":
    main()
