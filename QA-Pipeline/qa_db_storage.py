import os
import json
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PATH = os.path.join("output", f"qa_pipeline_{TIMESTAMP}.json")

def qa_database_storage(context_pairs: dict[str, any], context_id: int):
    id = context_id
    # one article at a time -> loop will run just once
    with open(PATH, "a") as json_file:
        if json_file.tell() == 0:
            json_file.write("[\n")
        else:
            json_file.write(",\n")
        context = context_pairs.pop("context")
        qa_pairs = context_pairs.pop("qa_pairs")
        json_file.write(json.dumps({"id": id, "context": context, "qa_pairs": qa_pairs}, indent=4))

def eof_qa_database():
    with open(PATH, "a") as json_file:
        json_file.write("\n]")