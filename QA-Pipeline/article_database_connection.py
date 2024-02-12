import requests

def get_article_summaries(db_index_start, num_articles):

    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "multi_news",
        "config": "default",
        "split": "train",
        "offset": db_index_start,
        "length": num_articles
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        summaries = [row['row']['summary'] for row in data['rows']]
        return summaries
    else:
        return None
    