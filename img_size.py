#!/usr/bin/env python
# Gets the average img size of the dataset

"""
Get information about images in a folder.
"""

from os import listdir
from os.path import isfile, join
import os
import pandas as pd
from PIL import Image


def print_data(data):
    """
    Parameters
    ----------
    data : dict
    """
    for k, v in data.items():
        print("%s:\t%s" % (k, v))
    print("Min width: %i" % data["min_width"])
    print("Max width: %i" % data["max_width"])
    print("Min height: %i" % data["min_height"])
    print("Max height: %i" % data["max_height"])


def main(path):
    """
    Parameters
    ----------
    path : str

        Path where to look for image files.
    """
    data_directory = "dataset"
    train_img_paths = pd.read_csv(
        os.path.join(data_directory, "MURA-v1.1/train_image_paths.csv"), names=["path"]
    )
    test_img_paths = pd.read_csv(
        os.path.join(data_directory, "MURA-v1.1/test_image_paths.csv"), names=["path"]
    )
    onlyfiles = [path for path in test_img_paths["path"]]
    onlyfiles.extend([path for path in train_img_paths["path"]])

    # Filter files by extension
    # onlyfiles = [f for f in onlyfiles if f.endswith(".jpg")]

    data = {}
    data["images_count"] = len(onlyfiles)
    data["min_width"] = 10**100  # No image will be bigger than that
    data["max_width"] = 0
    data["min_height"] = 10**100  # No image will be bigger than that
    data["max_height"] = 0

    total_width = 0
    total_height = 0

    for filename in onlyfiles:
        im = Image.open(os.path.join(data_directory, filename))
        width, height = im.size
        total_width += width
        total_height += height
        data["min_width"] = min(width, data["min_width"])
        data["max_width"] = max(width, data["max_width"])
        data["min_height"] = min(height, data["min_height"])
        data["max_height"] = max(height, data["max_height"])

    print_data(data)
    print("average width: ", total_width / data["images_count"])
    print("average height: ", total_height / data["images_count"])


if __name__ == "__main__":
    main(path=".")
