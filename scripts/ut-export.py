import glob
import os.path
import json
import csv



def do(img_dir, label_file=None, out=None):
    if label_file is None: label_file = f"{img_dir}-labels.json"
    if out is None: out = f"{label_file}.csv"
    labels = json.load(open(label_file, "r"))
    with open(out, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["filename", "label"])
        for img in glob.iglob(f"{img_dir}/*"):
            label = labels.get(os.path.basename(img), "NA")
            csvwriter.writerow([os.path.basename(img), label])
    


if __name__ == "__main__":
    do("original-processed", "original-processed-labels.json")