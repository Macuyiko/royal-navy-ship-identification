# from: https://uboat.net/allies/

from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import requests

base_url = "https://uboat.net/"
url = base_url + "allies/warships/ship/{}.html"

save_path = Path("./data/uboatnet/")
meta_path = Path("./data/meta/uboatnet/")

def do(start=0, end=25001):
    for i in tqdm(range(start, end)):
        soup = BeautifulSoup(requests.get(url.format(i)).text, 'lxml')
        try:
            info_table = soup.find("div", attrs={"id": "content"}).find("table", attrs={"class": "table_subtle width550"})
            image = soup.find("p", attrs={"class": "caption"}).find("img")
        except:
            continue
        if not info_table: continue
        if not image: continue
        header_ = soup.find("h1", attrs={"class": "warship_header"}).get_text(strip=True)
        navy_ = info_table.find("td", text=r"Navy").find_next_sibling('td').get_text(strip=True)
        type_ = info_table.find("td", text=r"Type").find_next_sibling('td').get_text(strip=True)
        class_ = info_table.find("td", text=r"Class").find_next_sibling('td').get_text(strip=True)
        srcname = image.get("src")
        filename = f"{i}__{header_}__{navy_}__{type_}__{class_}.jpg".replace("/", "")
        with open(save_path / filename, 'wb') as f:
            f.write(requests.get(f"{base_url}{srcname}").content)
        with open(meta_path / filename.replace(".jpg", ".html"), 'w', encoding="utf-8") as f:
            f.write(soup.find(attrs={"id": "content"}).prettify())

if __name__ == "__main__":
    save_path.mkdir(exist_ok=True, parents=True)
    meta_path.mkdir(exist_ok=True, parents=True)
    do()