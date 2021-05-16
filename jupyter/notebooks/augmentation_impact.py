# %% [markdown]
"""
# Research on augmentation impact
This notebook researches the impact of augmentation on the available rail dataset.
"""

# %% [markdown]
"""
## Import librarys and setup notebook
"""
# %%
import itertools

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from PIL import Image

# %%
IPython.display.set_matplotlib_formats("pdf", "svg")

# Set up absolute paths.
# Path to grid search root.
SRC_PTH = ""
SRC_PTH = pathlib.Path(SRC_PTH)
# Path to save figures.
SVE_PTH = ""
pathlib.Path(SVE_PTH)

# %% [markdown]
"""
## Functions to manipulate data and constatns
"""
# Augmentation probabilities.
AUG_PRBS: float = [prb / 10 for prb in range(0, 11)]


def get_filename(pth):
    """
    Convert image path to file name.

    :param path: Absolute path to Image.
    :return: Image name with file extension.
    """
    pth = pathlib.Path(pth)
    pth = pth.name
    pth = str(pth)
    return pth


def get_worst(prb: float, df: pd.DataFrame, aug_pth: pathlib.Path) -> tuple[list[str], list[float], float]:
    """
    Get worst images for given probability
    
    :param prob: Probability of augmentation.
    :df: Dataframe with IoU probability metrics.
    :aug_pth: Path to augmentation images.
    :return: Tuple with list of image names, list of image IoUs
             and given probability of augmentation.
    """
    
    df = df[["img", f"p_{prb:2.1f}"]]
    df = df.sort_values(by=f"p_{prb:2.1f}").head(10)
    img_abs_pth: list[str] = list(df["img"])
    img_names: list[pathlib.Path] = [aug_pth.joinpath(f"p_{prb:0.1f}/prd/confusion_overlay/{img_name}") for img_name in img_abs_pth]
    img_iou: list[float] = list(df[f"p_{prb:2.1f}"])
    return img_names, img_iou, prb

# %% [markdown]
"""
## Horizontal flip augmentation
"""

# %%
# Load data
hflip_pth = SRC_PTH.joinpath("01_h_flip")
hflip_csv_pth = hflip_pth.joinpath("h_flip.csv")
hflip_df = pd.read_csv(hflip_csv_pth)
hflip_df["img"] = hflip_df["img"].map(get_filename)

img_names, img_ious, img_prbs = [], [], []
for probability in AUG_PRBS:
    name, iou, prb = get_worst(probability, hflip_df, hflip_pth)
    for name, iou, p in zip(name, iou, itertools.repeat(prb)):
        img_names.append(name)
        img_ious.append(iou)
        img_prbs.append(p)

fig, axs = plt.subplots(ncols=10, nrows=11, figsize=[40, 30])
for ax, img_name, img_iou, img_prb in zip(axs.flat, img_names, img_ious, img_prbs):
    ax.axis('off')    
    ax.set_title(f"IoU: {img_iou: 0.2f}\n horizontal flip probability: {img_prb: 0.1f}")
    img = Image.open(img_name)
    ax.imshow(img)
plt.tight_layout()
plt.show()
save_pth = SVE_PTH.joinpath("horizontal.jpg")
fig.savefig(save_pth, bbox_inches="tight", dpi=300)

# %% [markdown]
"""
## Background augmentation
"""

# %%
# Load data
background_pth = SRC_PTH.joinpath("02_background")
background_csv_pth = background_pth.joinpath("background.csv")
background_df = pd.read_csv(background_csv_pth)
background_df["img"] = background_df["img"].map(get_filename)

img_names, img_ious, img_prbs = [], [], []
for probability in AUG_PRBS:
    name, iou, prb = get_worst(probability, background_df, background_pth)
    for name, iou, p in zip(name, iou, itertools.repeat(prb)):
        img_names.append(name)
        img_ious.append(iou)
        img_prbs.append(p)

fig, axs = plt.subplots(ncols=10, nrows=11, figsize=[40, 30])
for ax, img_name, img_iou, img_prb in zip(axs.flat, img_names, img_ious, img_prbs):
    ax.axis('off')    
    ax.set_title(f"IoU: {img_iou: 0.2f}\n background augmentation probability: {img_prb: 0.1f}")
    img = Image.open(img_name)
    ax.imshow(img)
plt.tight_layout()
plt.show()
save_pth = SVE_PTH.joinpath("background.jpg")
fig.savefig(save_pth, bbox_inches="tight", dpi=300)

# %% [markdown]
"""
## Contrast and Brithness augmentation
"""

# %%
# Load data
contrast_pth = SRC_PTH.joinpath("03_contrast")
contrast_csv_pth = contrast_pth.joinpath("contrast.csv")
contrast_df = pd.read_csv(contrast_csv_pth)
contrast_df["img"] = contrast_df["img"].map(get_filename)

img_names, img_ious, img_prbs = [], [], []
for probability in AUG_PRBS:
    name, iou, prb = get_worst(probability, contrast_df, contrast_pth)
    for name, iou, p in zip(name, iou, itertools.repeat(prb)):
        img_names.append(name)
        img_ious.append(iou)
        img_prbs.append(p)

fig, axs = plt.subplots(ncols=10, nrows=11, figsize=[40, 30])
for ax, img_name, img_iou, img_prb in zip(axs.flat, img_names, img_ious, img_prbs):
    ax.axis('off')    
    ax.set_title(f"IoU: {img_iou: 0.2f}\n contr. brigh. aug. prob.: {img_prb: 0.1f}")
    img = Image.open(img_name)
    ax.imshow(img)
plt.tight_layout()
plt.show()
save_pth = SVE_PTH.joinpath("contrast.jpg")
fig.savefig(save_pth, bbox_inches="tight", dpi=300)

# %% [markdown]
"""
## Motion blur augmentation
"""

# %%
# Load data
motion_pth = SRC_PTH.joinpath("04_motion")
motion_csv_pth = motion_pth.joinpath("motion.csv")
motion_df = pd.read_csv(motion_csv_pth)
motion_df["img"] = motion_df["img"].map(get_filename)

img_names, img_ious, img_prbs = [], [], []
for probability in AUG_PRBS:
    name, iou, prb = get_worst(probability, motion_df, motion_pth)
    for name, iou, p in zip(name, iou, itertools.repeat(prb)):
        img_names.append(name)
        img_ious.append(iou)
        img_prbs.append(p)

fig, axs = plt.subplots(ncols=10, nrows=11, figsize=[40, 30])
for ax, img_name, img_iou, img_prb in zip(axs.flat, img_names, img_ious, img_prbs):
    ax.axis('off')    
    ax.set_title(f"IoU: {img_iou: 0.2f}\n motion blurr augmentation probability: {img_prb: 0.1f}")
    img = Image.open(img_name)
    ax.imshow(img)
plt.tight_layout()
plt.show()
save_pth = SVE_PTH.joinpath("motion.jpg")
fig.savefig(save_pth, bbox_inches="tight", dpi=300)

# %% [markdown]
"""
## Rotation augmentation
"""

# %%
# Load data
rotation_pth = SRC_PTH.joinpath("05_rotate")
rotation_csv_pth = rotation_pth.joinpath("rotate.csv")
rotation_df = pd.read_csv(rotation_csv_pth)
rotation_df["img"] = rotation_df["img"].map(get_filename)

img_names, img_ious, img_prbs = [], [], []
for probability in AUG_PRBS:
    name, iou, prb = get_worst(probability, rotation_df, rotation_pth)
    for name, iou, p in zip(name, iou, itertools.repeat(prb)):
        img_names.append(name)
        img_ious.append(iou)
        img_prbs.append(p)

fig, axs = plt.subplots(ncols=10, nrows=11, figsize=[40, 30])
for ax, img_name, img_iou, img_prb in zip(axs.flat, img_names, img_ious, img_prbs):
    ax.axis('off')    
    ax.set_title(f"IoU: {img_iou: 0.2f}\n rotation augmentation probability: {img_prb: 0.1f}")
    img = Image.open(img_name)
    ax.imshow(img)
plt.tight_layout()
plt.show()
save_pth = SVE_PTH.joinpath("rotation.jpg")
fig.savefig(save_pth, bbox_inches="tight", dpi=300)
