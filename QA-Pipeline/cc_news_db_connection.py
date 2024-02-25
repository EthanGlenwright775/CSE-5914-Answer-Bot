import requests

URL = "https://datasets-server.huggingface.co/rows"

def get_article_summaries(db_index_start, num_articles) -> list[str]:
    params = {
        "dataset": "multi_news",
        "config": "default",
        "split": "train",
        "offset": db_index_start,
        "length": num_articles
    }

    response = requests.get(URL, params=params)

    if response.status_code == 200:
        data = response.json()
        summaries = [row['row']['summary'] for row in data['rows']]
        return summaries
    else:
        return None
    
def get_article_documents(db_index_start, num_articles) -> list[str]:
    params = {
        "dataset": "multi_news",
        "config": "default",
        "split": "train",
        "offset": db_index_start,
        "length": num_articles
    }

    response = requests.get(URL, params=params)

    if response.status_code == 200:
        data = response.json()
        document = [row['row']['document'] for row in data['rows']]
        return document
    else:
        return None