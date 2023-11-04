import glob
import os.path
import json
from bs4 import BeautifulSoup
import requests
import re


base_url = "https://uboat.net/"
valid_labels = {'submarine', 'cruiser', 'frigate', 'destroyer', 'carrier', 'corvette'}

def extract_from_uboat(filename):
    def valid_piece(piece):
        found = re.search(r"([a-z]+)", piece)
        found = found.group(1) if found else ""
        return found, len(found) > 2
    keyword = os.path.basename(filename).lower()\
        .replace(".jpg", "").replace("._", " ").replace("hnlms", "").replace("hms", "")\
        .split()
    keyword = ' '.join(valid_piece(k)[0].strip() for k in keyword if valid_piece(k)[1])
    soup = BeautifulSoup(requests.get(f"{base_url}allies/warships/search.php", params={"keyword": keyword}).text, 'lxml')
    table = soup.find("table", attrs={"class": "table_subtle width640"})
    if not table: return None
    print()
    print(filename, keyword)
    for tr in table.find_all("tr")[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        print(tds)
        if keyword in tds[1].lower(): 
            label = tds[3]
            print("FOUND", filename, keyword, tds[1], label)
            return label
    return None
        
                      

def extract_from_filename(filename):
    details = filename.split("__")
    return details[3]

def do(directory, extract_func=extract_from_filename, out=None):
    if out is None: out = f"{directory}-labels.json"
    labels = {}
    for img in glob.iglob(f"{directory}/*"):
        base = os.path.basename(img)
        label = extract_func(base)
        if not label: continue
        for lbl in valid_labels:
            if lbl in label.lower():
                labels[base] = lbl
    json.dump(labels, open(out, "w"))


if __name__ == "__main__":
    do("original-processed", extract_from_uboat)
    do("uboatnet-processed", extract_from_filename)