"""Gathers all images and preprocesses them to make sure they are BW and max dimension = 720px."""

import numpy as np
import glob
import shutil
import os.path
from PIL import Image, ImageOps


# Ignore decompression attacks because our files are large
Image.MAX_IMAGE_PIXELS = None
image_target_size = 720
input_directories = ["data/portraits", "data/royalnavy-1", "data/royalnavy-2", "data/navalhistory", "data/uboatnet"]
output_directory = "data/processed-royalnavy"


def clean_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def resize_to(img, targ_sz, use_min=False):
    w, h = img.size
    min_sz = (min if use_min else max)(w, h)
    ratio = targ_sz / min_sz
    w, h = int(np.ceil(w*ratio)), int(np.ceil(h*ratio))
    if w == targ_sz-1 or w == targ_sz+1: w = targ_sz
    if h == targ_sz-1 or h == targ_sz+1: h = targ_sz
    return w, h


def resize_image(
        src, dest, max_size=None, n_channels=3, 
        greyscale=True,
        ext=None, img_format=None, resample=Image.Resampling.BILINEAR, resume=False
    ):
    if resume and os.path.exists(dest): return
    img = Image.open(src)
    if ext is not None: dest = f"{dest}{ext}"
    if greyscale: img = ImageOps.grayscale(img)
    if max_size is not None:
        new_sz = resize_to(img, max_size)
        img = img.resize(new_sz, resample=resample)
    if n_channels == 3: img = img.convert("RGB")
    img.save(dest, img_format)
    

def do_directory(input_directory, prefix):
    for entry in sorted(glob.iglob(f"{input_directory}/*")):
        if os.path.isdir(entry):
            do_directory(entry, prefix)
        elif os.path.isfile(entry):
            base_name = os.path.basename(entry)
            dest = f"{output_directory}/{prefix}__{base_name}"
            try:
                resize_image(entry, dest, max_size=image_target_size, ext=".jpg", resume=True)
            except:
                continue
            print(f"{entry} -> {dest}")


def do():
    clean_directory(output_directory)
    for input_directory in input_directories:
        do_directory(input_directory, os.path.basename(input_directory))


if __name__ == "__main__":
    do()