import glob
import os.path
import json
import csv



def do(csv_file):
    labels = {}
    with open(csv_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            if row[0] == "filename": continue
            bart = row[4]
            filename = row[2]
            seppe = row[3]
            label = bart if bart else (filename if filename else seppe)
            if not label: continue
            labels[row[0].strip()] = label.strip()
    print(list(set(labels.values())))
    with open("saved-labels.json", "w") as jf:
        json.dump(labels, jf)


if __name__ == "__main__":
    do("labels - original.csv")