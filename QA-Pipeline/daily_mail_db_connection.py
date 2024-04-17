import requests
import sys

URL = "https://datasets-server.huggingface.co/rows"
DATASET = "cnn_dailymail"
CONFIG = "1.0.0"
SPLIT = "train"
LENGTH = 1
MAX_RETRIES = 10

def get_article(db_index: int) -> str:
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": db_index,
        "length": LENGTH
    }

    for attempt in range(MAX_RETRIES):
        response = requests.get(URL, params=params)

        if response.status_code == 200:
            data = response.json()
            return data["rows"][0]["row"]["article"]
        else:
            print(f"Error grabbing article with db index {db_index}: Attempt {attempt+1}/{MAX_RETRIES}")

    print(f"Thread w/ db index {db_index} failed to retrieve context")
    sys.exit(1)

#print(get_article(2))