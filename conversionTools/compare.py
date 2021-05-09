from PIL import Image, ImageChops, ImageStat
from typing import Iterable, Union
import itertools
import concurrent.futures
import os
import pathlib
import argparse


def compare_image(
    image1: pathlib.Path,
    image2: pathlib.Path,
    save_pth: pathlib.Path,
):
    img1: Image.Image = Image.open(image1).convert("RGB")
    img2: Image.Image = Image.open(image2).convert("RGB")

    # if image sizes differ somethings wrong
    if img1.size != img2.size:
        msg = "Expected images to be the same size"
        msg += f"{image1} has the size: {img1.size}"
        msg += f"{image2} has the size: {img2.size}"
        raise ValueError(msg)

    difference = ImageChops.difference(img1, img2)

    # get variance
    stat = ImageStat.Stat(difference)
    variance = stat.var

    if max(variance) >= 1:
        # Create save path if it does not exist.
        pathlib.Path(save_pth).mkdir(parents=True, exist_ok=True)
        image_name: str = os.path.basename(image1)
        save_name: str = os.path.join(save_pth, image_name)
        print("Found a difference, check: {}".format(image_name, save_name))
        difference.save(save_name, quality=100)


def compare_images(
    image1_folder: Union[pathlib.Path, str],
    image2_folder: Union[pathlib.Path, str],
    save_path: Union[pathlib.Path, str],
    image_extention: str = "png",
) -> None:
    """
    Compares 2 Images and saves the difference, if variance >1

    :param image1_folder: Path to a directory with images.
    :param image2_folder: Path to a directory with masks.
    :param save_path: Path to a directory to save non similar images.
    :param image_extention: File extension to look for in images folder.
    :return: None.
    """
    # Collect Images
    image1_folder = pathlib.Path(image1_folder)
    image1_paths = image1_folder.glob(f"{'*.' + image_extention}")
    image1_paths = sorted(list(image1_paths))

    # Collect Masks
    image2_folder = pathlib.Path(image2_folder)
    image2_paths = image2_folder.glob(f"{'*.' + image_extention}")
    image2_paths = sorted(list(image2_paths))

    # Sanity check
    if len(image1_paths) < 1:
        msg = f"Found no Images at: {image1_folder}"
        msg += f'Did you use the correct file extension? Got "{image_extention}".'
        raise ValueError(msg)
    if len(image2_paths) < 1:
        msg = f"Found no Images at: {image1_folder}"
        msg += f'Did you use the correct file extension? Got "{image_extention}".'
        raise ValueError(msg)
    if len(image1_paths) != len(image2_paths):
        msg = "Expected same amount of images"
        msg += f"{len(image1_paths)} images in folder 1 and {len(image2_paths)} images in folder 2."
        raise ValueError(msg)

    # Execute simultaneously
    arguments: Iterable
    arguments = (image1_paths, image2_paths, itertools.repeat(save_path))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(compare_image, *arguments)


def parse_args(parser: argparse.ArgumentParser):
    """
    Args

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "image_path1",
        type=pathlib.Path,
        help="Path to a heatmap folder",
    )
    parser.add_argument(
        "image_path2",
        type=pathlib.Path,
        help="Path to another heatmap folder",
    )
    parser.add_argument(
        "save_dir",
        type=pathlib.Path,
        help="Output path for non similar images",
    )
    parser.add_argument(
        "-ft",
        "--file_type",
        type=str,
        help="image filetype (must be the same)",
        default="png",
        required=False,
    )

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    image1 = args.image_path1
    image2 = args.image_path2
    save = args.save_dir
    img_ext = args.file_type

    compare_images(image1, image2, save, image_extention=img_ext)
    print("Finished.")


if __name__ == "__main__":
    main()
