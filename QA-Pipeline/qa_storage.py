import os
import json
from datetime import datetime

def qa_database_storage(context_pairs: dict[str, dict[str, str]]):
    id = 0
    json_data = []
    for context, qa_pairs in context_pairs.items(): 
        qa_list = []
        for question, answer in qa_pairs.items():
            qa_list.append({"question": question, "answer": answer})
        json_element = {"id": id,
                        "context": context, 
                        "qa_pairs": qa_list
        }
        json_data.append(json_element)
        id += 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join("output", f"qa_pipeline_{timestamp}.json")
    with open(path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)