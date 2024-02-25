import requests

URL = "https://datasets-server.huggingface.co/rows"
DATASET = "dataset=AyoubChLin%2FCNN_News_Articles_2011-2022"
CONFIG = "default"
SPLIT = "train"
LENGTH = 1

def get_article(db_index: int) -> str:
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": db_index,
        "length": LENGTH
    }

    response = requests.get(URL, params=params)

    if response.status_code == 200:
        data = response.json()
        return data["rows"][0]["row"]["text"]
    else:
        return None