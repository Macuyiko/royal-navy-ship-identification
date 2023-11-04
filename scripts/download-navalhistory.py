# from: https://uboat.net/allies/

from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import requests
import re

base_url = "https://www.naval-history.net/"
url = f"{base_url}xGM-aContents.htm"

save_path = Path("./data/navalhistory/")
meta_path = Path("./data/meta/navalhistory/")


def strip_nl(txt):
    return " ".join(line.strip() for line in txt.split("\n"))


def get_valid_filename(name):
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    return s


def do():
    soup = BeautifulSoup(requests.get(url).text, 'lxml')
    main_table = soup.find(string=re.compile(r"TYPE\s+and\s+CLASS")).find_parent("table")
    sub_tables = main_table.find_all("table", attrs={"width": "100%"})
    type_header = None
    sub_header = None
    sub_header_prefix = ""
    for sub_table in tqdm(sub_tables):
        for tr in tqdm(sub_table.find_all("tr")):
            ships = []
            tds = tr.find_all("td")
            if not tds: continue
            b = tds[0].find("b")
            if len(tds) == 1 and b:
                if b.find("a"):
                    sub_header_prefix = strip_nl(b.get_text(strip=True))
                    sub_header_prefix += "-"
                else: 
                    type_header = strip_nl(b.get_text(strip=True))
                    sub_header = None
                    sub_header_prefix = ""
            elif len(tds) == 2:
                sub_header = strip_nl(tds[0].get_text(strip=True))
                ships = tds[1].find_all("a")
                if not ships: continue
            for ship in tqdm(ships):
                ship_name = strip_nl(ship.get_text(strip=True))
                ship_href = f"{base_url}{ship.get('href')}"
                ship_html = BeautifulSoup(requests.get(ship_href).text, 'lxml')
                try:
                    ship_src = ship_html.find("td", attrs={"bgcolor": "#CCCCCC"}).find("img").get("src")
                except AttributeError:
                    continue
                filename = get_valid_filename(f"{type_header}__{sub_header_prefix}{sub_header}__{ship_name}.jpg".replace("/", ""))
                with open(save_path / filename, 'wb') as f:
                    f.write(requests.get(f"{base_url}{ship_src}").content)
                with open(meta_path / filename.replace(".jpg", ".html"), 'w', encoding="utf-8") as f:
                    f.write(ship_html.prettify())


if __name__ == "__main__":
    save_path.mkdir(exist_ok=True, parents=True)
    meta_path.mkdir(exist_ok=True, parents=True)
    do()