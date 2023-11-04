"""Rename all images in data/royalnavy based on label and store in data/royalnavy-1."""

import os
import os.path
import glob
import shutil
import json


original_dir = "data/royalnavy"
output_dir = "data/royalnavy-1"
labels_json = "data/meta/saved-labels.json"


def clean_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def do(original_dir, output_dir, labels_json):
    clean_directory(output_dir)
    labels = json.load(open(labels_json, "r"))
    for image_o in glob.iglob(f"{original_dir}/*"):
        if not os.path.isfile(image_o):
            continue
        image = os.path.basename(image_o)
        lbl = labels.get(image, "unknown")
        new_image = f"{lbl}__{image}"
        shutil.copy(image_o, f"{output_dir}/{new_image}")
        print(f"{image} -> {new_image}")

if __name__ == "__main__":
    do(original_dir, output_dir, labels_json)