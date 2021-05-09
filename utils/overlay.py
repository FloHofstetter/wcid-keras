from PIL import Image
from typing import Iterable, Union
import itertools
import concurrent.futures
import os
import pathlib


def overlay_mask(
    image_pth: pathlib.Path,
    mask_pth: pathlib.Path,
    save_pth: pathlib.Path,
) -> Image.Image:
    """
    Blend two images and save to given path.
    Create save path if not exists.

    :param image_pth: Image to overlay.
    :param mask_pth: Make to overlay.
    :param save_pth: Save path for overplayed image.
    :return: Image overplayed with mask.
    """
    # Create save path if not exists.
    pathlib.Path(save_pth).mkdir(parents=True, exist_ok=True)

    # Blend images.
    image: Image.Image = Image.open(image_pth).convert("RGB")
    mask: Image.Image = Image.open(mask_pth).convert("RGB")
    overplayed: Image.Image = Image.blend(image, mask, alpha=0.5)

    # Save blended image with same filename.
    image_name: str = os.path.basename(image_pth)
    save_name: str = os.path.join(save_pth, image_name)
    overplayed.save(save_name, quality=100)

    return overplayed


def overlay_masks(
    imgs_pth: Union[pathlib.Path, str],
    msks_pth: Union[pathlib.Path, str],
    sve_pth: Union[pathlib.Path, str],
    img_ext: str = "png",
    msk_ext="png",
) -> None:
    """
    Blend images with same name from two directors.
    Files are collected by file extension.
    Save blended images with same name.

    :param imgs_pth: Path to a directory with images.
    :param msks_pth: Path to a directory with masks.
    :param sve_pth: Path to a directory to save blends.
    :param img_ext: File extension to look for in images folder.
    :param msk_ext: File extension to look for in masks folder.
    :return: None.
    """
    # Collect Images
    imgs_pth = pathlib.Path(imgs_pth)
    images_paths = imgs_pth.glob(f"{'*' if img_ext is None else '*.' + img_ext}")
    images_paths = sorted(list(images_paths))

    # Collect Masks
    msks_pth = pathlib.Path(msks_pth)
    masks_paths = msks_pth.glob(f"{'*' if msk_ext is None else '*.' + msk_ext}")
    masks_paths = sorted(list(masks_paths))

    # Sanity check
    if len(images_paths) < 1:
        msg = f"Expected at least one mask, got none. "
        msg += f'Did you use the correct file extension? Got "{img_ext}".'
        raise ValueError(msg)
    if len(masks_paths) < 1:
        msg = f"Expected at least one mask, got none. "
        msg += f'Did you use the correct file extension? Got "{msk_ext}".'
        raise ValueError(msg)
    if len(images_paths) != len(masks_paths):
        msg = "Expected same amount of images and masks, got "
        msg += f"{len(images_paths)} images and {len(masks_paths)}."
        raise ValueError(msg)

    # Execute simultaneously
    arguments: Iterable
    arguments = (images_paths, masks_paths, itertools.repeat(sve_pth))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(overlay_mask, *arguments)


def main():
    image: str = "/data/hofstetter_data/full_dataset_with_nlb/img/val/"
    mask: str = "/data/chumak_data/tf_to_plaidml/none/heatmap/"
    save: str = "/data/chumak_data/tf_to_plaidml/none/heatmap_overlay/"
    img_ext: str = "png"
    msk_ext: str = "png"
    overlay_masks(image, mask, save, img_ext=img_ext, msk_ext=msk_ext)


if __name__ == "__main__":
    main()
