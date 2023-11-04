"""Create train/valid/all splits with labeled subfolders."""

import os
import os.path
import shutil
import numpy as np


in_dir = "data/processed-all"
out_dir = "data_prep/all-nowhite"
   

CLASSES = {
    'survey': 'skip',
    'landing': 'skip',
    'repair': 'skip',
    'transport': 'skip',
    'depot': 'skip', 
    'troop': 'skip', 
    'support': 'skip', 
    'tanker': 'skip',  
    'patrol': 'skip',  
    'minelayer': 'skip',
    'trawler': 'skip', 
    'sloop': 'skip',
    'tanker': 'skip', 
    'monitors': 'skip',
    'escort': 'skip',
    'unknown': 'skip',
    
    'battleship': 'battleship', 
    'destroyer': 'destroyer', 
    'frigate': 'frigate', 
    'cruiser': 'cruiser', 
    'corvette': 'corvette', 
    'carrier': 'carrier', 
    'minesweeper': 'minesweeper',
    'submarine': 'submarine', 
    'crusier': 'cruiser', 
    
    'battlehship': 'battleship',
    'battleships': 'battleship',
    'desroyer': 'destroyer', 
    'destoyer': 'destroyer', 
    'dstroyer': 'destroyer',
    'destroter': 'destroyer', 
    'deastroyer': 'destroyer', 
    'desrtoyer': 'destroyer', 
    'destroyers': 'destroyer',
    'battlsehip': 'battleship', 
}

def clean_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def get_image_file_names(directory):
    files = []
    for r, d, f in os.walk(directory): files.extend(f)
    if 'desktop.ini' in files: files.remove('desktop.ini')
    return files


def get_image_file_name_labels(file_name):
    fnl = file_name.lower()
    if fnl.startswith("navalhistory"): return [c for c in CLASSES if c in fnl.split("__")[1].replace("_", " ")]
    if fnl.startswith("uboatnet"): return [c for c in CLASSES if c in fnl.split("__")[4].replace("_", " ")]
    if fnl.startswith("portraits"): return [c for c in CLASSES if c in fnl.replace("_", " ")]
    if fnl.startswith("royalnavy-1"): return [c for c in CLASSES if c in fnl.split("__")[1].replace("_", " ")]
    if fnl.startswith("royalnavy-2"): return [c for c in CLASSES if c in fnl.replace("_", " ")]

def get_files_without_label(file_names):
    no_label = []
    for file_name in file_names:
        labels = get_image_file_name_labels(file_name)
        if not labels: no_label.append((file_name, labels))
    return no_label

def get_files_with_multiple_labels(file_names):
    mu_label = []
    for file_name in file_names:
        labels = get_image_file_name_labels(file_name)
        if len(labels) > 1: mu_label.append((file_name, labels))
    return mu_label

def get_files_with_labels(file_names):
    data = []
    for file_name in file_names:
        labels = get_image_file_name_labels(file_name)
        classes = set(CLASSES[l] if CLASSES.get(l) else l for l in labels)
        if "skip" in classes: classes.remove("skip")
        if not classes: classes.add("other")
        if len(classes) > 1 and "other" in classes: classes.remove("other")
        assert len(classes) == 1, f"{file_name} {classes}"
        data.append((file_name, next(iter(classes))))
    return data

def move_files_to_staging_dir(
    files_with_labels, source_directory, target_directory, 
    valid_part = 0.2, skip_other=True
):
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.makedirs(target_directory, exist_ok=True)
    target_directory_train = f"{target_directory}/train"
    target_directory_valid = f"{target_directory}/valid"
    target_directory_all = f"{target_directory}/all"
    for file, label in files_with_labels:
        if skip_other and label == "other": continue

        target_directory_with_label = target_directory_train if np.random.random() > valid_part else target_directory_valid
        target_directory_with_label += f"/{label}"
        if not os.path.exists(target_directory_with_label): os.makedirs(target_directory_with_label, exist_ok=True)
        shutil.copy(f"{source_directory}/{file}", f"{target_directory_with_label}/{file}")

        target_directory_with_label = f"{target_directory_all}/{label}"
        if not os.path.exists(target_directory_with_label): os.makedirs(target_directory_with_label, exist_ok=True)
        shutil.copy(f"{source_directory}/{file}", f"{target_directory_with_label}/{file}")


if __name__ == "__main__":
    image_files = get_image_file_names(in_dir)
    files_labels = get_files_with_labels(image_files)
    print(len(image_files))
    move_files_to_staging_dir(files_labels, in_dir, out_dir)