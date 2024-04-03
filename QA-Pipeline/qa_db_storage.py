import os
import json
import threading

JSON_PATH = os.path.join("QA-Output", "pipeline_output.json")
TRAINING_PATH = os.path.join("QA-Output", "training.tsv")
VALIDATION_PATH = os.path.join("QA-Output", "validation.tsv")
TESTING_PATH = os.path.join("QA-Output", "testing.tsv")

output_json_lock = threading.Lock()
training_csv_lock = threading.Lock()
validation_csv_lock = threading.Lock()
testing_csv_lock = threading.Lock()

count_lock = threading.Lock()

training_count = 0
validation_count = 0
testing_count = 0
RATIO = 8 # 80 - 10 - 10 (training - validation - testing) split of qa pairs

first_json = True

# done before threading
def pre_storage():
    with open(JSON_PATH, "w") as file:
        file.write("[\n")
    with open(TRAINING_PATH, "w") as file:
        file.write("context\ttarget\n")
    with open(VALIDATION_PATH, "w") as file:
        file.write("context\ttarget\n")
    with open(TESTING_PATH, "w") as file:
        file.write("context\ttarget\n")

# each thread will store its items in json and csv
def qa_database_storage(context_pairs: dict[str, any], context_id: int):
    json_storage(context_pairs, context_id)
    csv_storage(context_pairs)

def json_storage(context_pairs: dict[str, any], context_id: int):
    global first_json
    id = context_id
    context = context_pairs.get("context")
    qa_pairs = context_pairs.get("qa_pairs")
    with output_json_lock:
        with open(JSON_PATH, "a") as json_file:
            if first_json:
                first_json = False
            else:
                json_file.write(",\n")
            json_file.write(json.dumps({"id": id, "context": context, "qa_pairs": qa_pairs}, indent=4))

def csv_storage(context_pairs: dict[str, any]):
    global training_count
    global validation_count
    global testing_count
    article = context_pairs.get("context")
    qa_pairs = context_pairs.get("qa_pairs")
    with count_lock:
        if testing_count < (training_count + validation_count + testing_count) / RATIO:
            path = TESTING_PATH
            lock = testing_csv_lock
            testing_count += len(qa_pairs)
            print(f"Testing Count: {testing_count}")
        elif validation_count < (training_count + validation_count + testing_count) / RATIO:
            path = VALIDATION_PATH
            lock = validation_csv_lock
            validation_count += len(qa_pairs)
            print(f"Validation Count: {validation_count}")
        else:
            path = TRAINING_PATH
            lock = training_csv_lock
            training_count += len(qa_pairs)
            print(f"Training Count: {training_count}")
    with lock:
        with open(path, "a") as csv_file:
            for qa_pair in qa_pairs:
                question = qa_pair.get("question")
                answer = qa_pair.get("answer")
                csv_file.write(f"Use the following article to answer the question: Article: {article} Question: {question}\t{answer}\n")

# done after threading
def post_storage():
    with open(JSON_PATH, "a") as json_file:
        json_file.write("\n]")