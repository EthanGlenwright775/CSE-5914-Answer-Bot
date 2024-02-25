import os
import json
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def qa_database_storage(context_pairs: dict[str, list[dict[str, str]]], context_id: int):
    id = context_id
    json_data = []
    # one article at a time -> loop will run just once
    for context, qa_list in context_pairs.items(): 
        json_element = {
            "id": id,
            "context": context, 
            "qa_pairs": qa_list
        }
        json_data.append(json_element)
    path = os.path.join("output", f"qa_pipeline_{TIMESTAMP}.json")
    with open(path, "a") as json_file:
        json.dump(json_data, json_file, indent=4)