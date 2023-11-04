"""Gathers all images and preprocesses them further, use SAM to remove water and air and pad to 720x720."""

import numpy as np
import glob
import shutil
import os.path
from PIL import Image, ImageOps
from segment_anything import sam_model_registry, SamPredictor


image_target_size = 720
input_directories = ["data/processed-external"]
output_directory = "data/cleaned-external"
sam_checkpoint = "c:/users/seppe/downloads/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def clean_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

def pad_image_to_square(image, size=720, fill_color=(255, 255, 255)):
    height, width, channels = image.shape
    pad_width = size - width
    pad_height = size - height
    top_pad = pad_height // 2
    left_pad = pad_width // 2
    padded_image = np.full((size, size, channels), fill_color, dtype=np.uint8)
    padded_image[top_pad:top_pad+height, left_pad:left_pad+width, :] = image
    return padded_image

def whiten_image(image, fill_color=(255, 255, 255)):
    predictor.set_image(image)
    points = [
        (image.shape[1] // 4, 0), 
        (image.shape[1] // 2, 0), 
        (image.shape[1] // 4 * 3, 0),
        (image.shape[1] // 4, image.shape[0] - 1), 
        (image.shape[1] // 2, image.shape[0] - 1), 
        (image.shape[1] // 4 * 3, image.shape[0] - 1), 
    ]
    image_masked = image
    for x, y in points:
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            image_masked[mask] = (255,255,255)
    if image.shape[1] == 720: # width is 720 -- remove dangling lines
        image_masked[0:5,:,:] = (255,255,255)
        image_masked[image.shape[0]-5:image.shape[0],:,:] = (255,255,255)
    return image_masked

def do_directory(input_directory):
    for entry in sorted(glob.iglob(f"{input_directory}/*.jpg")):
        if os.path.isfile(entry):
            base_name = os.path.basename(entry)
            dest = f"{output_directory}/{base_name}"
            print(f"{entry} -> {dest}")
            if not os.path.exists(dest):
                pil_image = Image.open(entry)
                image = np.array(pil_image)
                whitened = whiten_image(image)
                padded = pad_image_to_square(whitened)
                Image.fromarray(padded).save(dest)


def do():
    clean_directory(output_directory)
    for input_directory in input_directories:
        do_directory(input_directory)


if __name__ == "__main__":
    do()