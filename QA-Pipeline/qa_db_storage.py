import os
import json
import threading

JSON_PATH = os.path.join("Output", "pipeline_output.json")
TRAINING_PATH = os.path.join("QA-Output", "training.csv")
VALIDATION_PATH = os.path.join("QA-Output", "validation.csv")
TESTING_PATH = os.path.join("QA-Output", "testing.csv")

output_json_lock = threading.Lock()
training_csv_lock = threading.Lock()
validation_csv_lock = threading.Lock()
testing_csv_lock = threading.Lock()

count_lock = threading.Lock()

training_count = 0
validation_count = 0
testing_count = 0
SPLIT = 8 # 80 - 10 - 10 (training - validation - testing) split of qa pairs

# done before threading
def pre_storage():
    with open(JSON_PATH, "w") as file:
        file.write("[\n")
    with open(TRAINING_PATH, "w") as file:
        file.write("context, target\n")
    with open(VALIDATION_PATH, "w") as file:
        file.write("context, target\n")
    with open(TESTING_PATH, "w") as file:
        file.write("context, target\n")

# each thread will store its items in json and csv
def qa_database_storage(context_pairs: dict[str, dict[str, str]], context_id: int):
    json_storage(context_pairs, context_id)
    csv_storage(context_pairs)

def json_storage(context_pairs: dict[str, dict[str, str]], context_id: int):
    id = context_id
    context = context_pairs.get("context")
    qa_pairs = context_pairs.get("qa_pairs")
    with output_json_lock:
        with open(JSON_PATH, "a") as json_file:
            json_file.write(",\n")
            json_file.write(json.dumps({"id": id, "context": context, "qa_pairs": qa_pairs}, indent=4))

def csv_storage(context_pairs: dict[str, dict[str, str]]):
    context = context_pairs.get("context")
    for pair in context_pairs.get("qa_pairs"):
        question = pair.get("question")
        answer = pair.get("answer")
        with count_lock:
            if training_count < validation_count and training_count < testing_count:
                path = TRAINING_PATH
                lock = training_csv_lock
                training_count += 1
            elif validation_count < training_count and validation_count < testing_count:
                path = VALIDATION_PATH
                lock = validation_csv_lock
                validation_count += SPLIT
            else:
                path = TESTING_PATH
                lock = testing_csv_lock
                testing_count += SPLIT
        with lock:
            with open(path, "a") as csv_file:
                csv_file.write(f"{context}; {question}, {answer}")

# done after threading
def post_storage():
    with open(JSON_PATH, "a") as json_file:
        json_file.write("\n]")