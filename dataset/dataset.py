from tensorflow import keras
import numpy as np
from PIL import Image
from typing import Tuple, List
import glob
import os
import random
import albumentations as A


class RailDataset(keras.utils.Sequence):
    """
    Class representation of the Nordlandbahn track data.
    """
    def __init__(
            self,
            imgs_pth: str,
            msks_pth: str,
            res: Tuple[int, int] = (8, 8),
            img_ftype: str = "png",
            msk_ftype: str = "png",
            batch_size: int = 1,
            transforms: bool = True
    ) -> None:
        """
        Set up the parameters of the Dataset.

        :param imgs_pth: Path to images folder.
        :param msks_pth: Path to masks folder.
        :param img_ftype: File extension for images.
        :param msk_ftype: File extension for masks.
        :param res: Resolution for training (width, height)
        :param batch_size: Count of images returned per batch.
        :param transforms: Activate image augmentations
        :return: None.
        """
        self.imgs_pth: str = imgs_pth
        self.msks_pth: str = msks_pth
        self.res: Tuple[int, int] = res
        self.img_ftype: str = img_ftype
        self.msk_ftype: str = msk_ftype
        self.batch_size = batch_size
        self.transforms = transforms

        # Collect image paths
        imgs_pth: str = os.path.join(imgs_pth, f"*.{img_ftype}")
        img_pths: List[str] = glob.glob(imgs_pth)
        # Sort to get for every image corresponding mask
        img_pths = sorted(img_pths)

        # Collect mask paths
        msks_pth: str = os.path.join(msks_pth, f"*.{msk_ftype}")
        msk_pths: List[str] = glob.glob(msks_pth)
        # Sort to get for every mask corresponding image
        msk_pths = sorted(msk_pths)

        # Shuffle lists in the same way
        shuffled_list: List = list(zip(img_pths, msk_pths))
        random.shuffle(shuffled_list)
        self.img_pths, self.msk_pths = zip(*shuffled_list)

        # Sanity checks
        if res[0] < 1 or res[1] < 1:
            err = f"Resolution mus be grater than 1, got {res}"
            raise ValueError(err)

        if not len(self.img_pths) == len(self.msk_pths):
            err = (
                    f"Amount of images and masks must be the same,"
                    + f"got {len(self.img_pths)} and"
                    + f" {len(self.msk_pths)} masks"
            )
            raise ValueError(err)

    def __len__(self):
        """
        Get length of the amount of batches in the dataset.

        :return: Overall batches in the dataset.
        """
        return len(self.img_pths) // self.batch_size

    def __getitem__(self, batch_idx):
        """
        Get next batch from the dataset.

        :param batch_idx: Index of the iterator.
        :return: Image and mask batch.
        """
        img_list = []
        msk_list = []
        # For every image in the batch
        for file_idx in range(batch_idx, batch_idx + self.batch_size):
            # Open images and masks
            image_img = Image.open(self.img_pths[file_idx])
            mask_img = Image.open(self.msk_pths[file_idx])

            # Resize images and masks
            image_img = image_img.resize(self.res)
            mask_img = mask_img.resize(self.res)

            # Convert images and mask to array
            image_arr = np.asarray(image_img).copy()
            mask_arr = np.asarray(mask_img).copy()

            # List transformations
            transform = A.Compose(
                [
                    A.RandomResizedCrop(height=self.res[1], width=self.res[0], p=0.9, scale=(0.8, 1.0,)),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Rotate(limit=(-25., 25.,), p=0.9),
                    A.MotionBlur(always_apply=False, p=0.1, blur_limit=(14, 20))
                ]
            )
            # Augment Images
            if self.transforms:
                augmentations = transform(image=image_arr, mask=mask_arr)
                image_arr = augmentations["image"]
                mask_arr = augmentations["mask"]

            # Make sure only 2 classes
            highest_class = np.max(mask_arr)
            lowest_class = np.min(mask_arr)
            if highest_class > 1 or lowest_class < 0:
                classes = np.unique(mask_arr)
                err = f"Expected two classes [0 1], got {len(classes)}: {classes}."
                raise ValueError(err)

            # Expand Mask dimension
            mask_arr = np.expand_dims(mask_arr, axis=0)

            # Cast datatype and normalize Image
            image_arr = image_arr.astype(np.float32)
            mask_arr = mask_arr.astype(np.float32)

            # Scale images
            image_arr /= 255.0

            # Height width channels to channel height width
            img_trans = image_arr.transpose((0, 1, 2))
            mask_arr = mask_arr.transpose((1, 2, 0))

            # List of arrays
            img_list.append(img_trans)
            msk_list.append(mask_arr)

        img_batch = np.stack(img_list)
        msk_batch = np.stack(msk_list)

        return img_batch, msk_batch

    def on_epoch_end(self):
        """
        Behaviour on end of epoch.

        :return:
        """
        # Shuffle dataset again
        shuffled_list: List = list(zip(self.img_pths, self.msk_pths))
        random.shuffle(shuffled_list)
        self.img_pths, self.msk_pths = zip(*shuffled_list)
