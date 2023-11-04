from pathlib import Path
import json
import shutil
import numpy as np


def extract_from_filename(filename):
    details = filename.split("__")
    return details[3]

def do(directory, train_size=0.7, label_file=None):
    if label_file is None: label_file = f"{directory}-labels.json"
    labels = json.load(open(label_file, "r"))
    train_dir = Path(f"{directory}/train")
    valid_dir = Path(f"{directory}/valid")
    train_dir.mkdir(exist_ok=True, parents=True)
    valid_dir.mkdir(exist_ok=True, parents=True)
    for img, label in labels.items():
        if np.random.random() <= train_size:
            dest_dir = train_dir / label
        else:
            dest_dir = valid_dir / label
        dest_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f"{directory}/{img}", dest_dir)


if __name__ == "__main__":
    do("original-processed")
    do("uboatnet-processed")