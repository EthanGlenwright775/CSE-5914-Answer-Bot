# Connection to the CC News dataset through the Hugging Face API

import requests
import sys

URL = "https://datasets-server.huggingface.co/rows"
DATASET = "multi_news"
CONFIG = "default"
SPLIT = "train"
LENGTH = 1
MAX_RETRIES = 10
    
def cc_get_article(db_index):
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
            return [row['row']['document'] for row in data['rows']][0]
        else:
            print(f"Error grabbing article with db index {db_index}: Attempt {attempt+1}/{MAX_RETRIES}")

    print(f"Thread w/ db index {db_index} failed to retrieve context")
    sys.exit(1)