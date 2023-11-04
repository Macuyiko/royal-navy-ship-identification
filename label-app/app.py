from flask import Flask, request, render_template, jsonify, g, flash, redirect, url_for
from flask_jsglue import JSGlue
from time import time
import os.path
import glob
import base64
import json
from oslo_concurrency import lockutils
from oslo_concurrency import processutils


app = Flask(__name__)
jsglue = JSGlue(app)

app.config['SECRET_KEY'] = 'ijreiowhjriohwetoihwetoinweoihgwehgowehtgwuehfevbgnewg'
image_dir = "original-processed"
images = sorted(list(map(os.path.basename, glob.iglob(f"{image_dir}/*.jpg"))))
classes = ['survey vessel', 'repair ship', 'tanker', 'destroyer', 'frigate', 'minelayer', 'survey ship', 'landing ship', 'cruiser', 'depot ship', 'aircraft transport ship', 'submarine depot ship', 'corvette', 'motor launch ship', 'Iowa class battleship', 'patrol boat', 'trawler', 'amphibious assault ship', 'tank landing ship', 'carrier', 'minesweeper', 'submarine', 'sloop', 'troop ship', 'support ship']
classes = sorted(classes) + ['other']
#classes = ["destroyer", "frigate", "cruiser", "corvette", "submarine", "carrier", "other"]
label_file = "saved-labels.json"


@lockutils.synchronized('not_thread_process_safe')
def save_label_to_file(file_name, label=None, just_get=False):
    if not os.path.exists(label_file):
        labels = {}
    else:
        with open(label_file, "r") as fn:
            labels = json.load(fn)
    if not just_get:
        labels[file_name] = label
        with open(label_file, "w") as fn:
            json.dump(labels, fn)
    return labels.get(file_name, None)


@app.route('/', methods=['GET', 'POST'])
def main():
    labels = {image: save_label_to_file(image, label=None, just_get=True) for image in images}
    return render_template('index.html', total=len(images), classes=classes, images=images, labels=labels)


@app.route('/save_label/<int(signed=True):ind>/<label>', methods=['POST'])
def save_label(ind, label):
    if ind < 0 or ind >= len(images):
        return jsonify({"result": "error"})
    label = str(label)
    if label == "null":
        label = None
    saved = save_label_to_file(images[ind], label)
    return jsonify({"result": "success", "label": saved})


@app.route('/get_image/<int(signed=True):ind>', methods=['GET'])
def get_image(ind):
    if ind >= len(images): ind = 0
    if ind < 0: ind = len(images) - 1
    with open(f"{image_dir}/{images[ind]}", "rb") as image2string:
        data = base64.b64encode(image2string.read()).decode('ascii')
    label = save_label_to_file(images[ind], label=None, just_get=True)
    return jsonify({
        "filename": os.path.basename(images[ind]),
        "index": ind,
        "data": data,
        "label": label
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=3982)
