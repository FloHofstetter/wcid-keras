# %% [markdown]
"""
# Research on augmentation impact
This notebook shows the worst pictures predicted by the classifier.
"""

# %% [markdown]
"""
## Import library's and setup notebook
"""
# %%
import os
import pathlib
from typing import AnyStr, Union

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# %%
IPython.display.set_matplotlib_formats("pdf", "svg")

# Set up absolute paths.
# Path to grid search root.
SRC_PTH = ""
src_pth = pathlib.Path(SRC_PTH)
# Path to save figures.
SVE_PTH = ""
sve_pth = pathlib.Path(SVE_PTH)

# %% [markdown]
"""
## Functions to manipulate data and constants
"""


def get_filename(pth: os.PathLike[str]) -> str:
    """
    Convert image path to file name.

    :param pth: Absolute path to Image.
    :return: Image name with file extension.
    """
    file_pth = pathlib.Path
    file_pth = pathlib.Path(pth)

    return file_pth.name


def get_worst(
    df: pd.DataFrame,
    benchmark_pth: pathlib.Path,
) -> tuple[list[pathlib.Path], list[float]]:
    """
    Get worst images for a given benchmark.

    :df: Dataframe with IoU probability metrics.
    :return: Tuple with list of image names, list of image IoUs
             and given probability of augmentation.
    """

    df = df[["img", "iou"]]
    df = df.sort_values(by=f"iou").head(25)
    img_abs_pth: list[str] = list(df["img"])
    img_names: list[pathlib.Path]
    img_names = [
        benchmark_pth.joinpath(f"confusion_overlay/{img_name}")
        for img_name in img_abs_pth
    ]
    img_iou: list[float] = list(df[f"iou"])
    return img_names, img_iou


# %% [markdown]
"""
## Worst predicted images
"""

# %%
# Load data
benchmark_csv_pth = src_pth.joinpath("image_metrics.csv")
benchmark_df = pd.read_csv(benchmark_csv_pth)
benchmark_df["img"] = benchmark_df["img"].map(get_filename)

img_names: list[pathlib.Path] = []
img_ious: list[float] = []

names, ious = get_worst(benchmark_df, src_pth)
for name, iou in zip(names, ious):
    img_names.append(name)
    img_ious.append(iou)

fig: plt.Figure
axs: np.ndarray
fig, axs = plt.subplots(ncols=5, nrows=5, figsize=[20, 20])
fig.suptitle('All augmentations', fontsize=16)

ax: plt.Axes
img_name: str
img_iou: float
for ax, img_name, img_iou in tqdm(zip(axs.flat, img_names, img_ious)):
    ax.axis("off")
    ax.set_title(f"IoU: {img_iou: 0.2f}")
    img = Image.open(img_name)
    ax.imshow(img)
plt.tight_layout()
plt.show()
save_pth = sve_pth.joinpath("worst.jpg")
fig.savefig(save_pth, bbox_inches="tight", dpi=300)
