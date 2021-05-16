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
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import pandas as pd
import numpy as np

# %%
IPython.display.set_matplotlib_formats("pdf", "svg")

# Set up absolute paths.
SRC_PTH = "/home/flo/ml-server/data/hofstetter_data/full_dataset_with_nlb/prd/mark_320x160/2021-05-09T00:50/grid/"
SVE_PTH = ""
SRC_PTH = pathlib.Path(SRC_PTH)
# pathlib.Path(SVE_PTH)

# %% [markdown]
"""
## Horizontal flip augmentation
"""

# %%
# Load data
hflip_csv_pth = SRC_PTH.joinpath("01_h_flip/h_flip.csv")
hflip_df = pd.read_csv(hflip_csv_pth)
hflip_df.head()

# %%
