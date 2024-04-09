import os
import json
import threading
import pandas as pd

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
RATIO = 10 # 80 - 10 - 10 (training - validation - testing) split of qa pairs

first_json = True

# done before threading
def pre_storage():
    global training_count
    global validation_count
    global testing_count
    json_exists = os.path.isfile(JSON_PATH)
    tsv_exists = os.path.isfile(TRAINING_PATH) and os.path.isfile(VALIDATION_PATH) and os.path.isfile(TESTING_PATH)
    if not (tsv_exists and json_exists):
        with open(JSON_PATH, "w") as file:
            file.write("[\n")
        with open(TRAINING_PATH, "w") as file:
            file.write("context\ttarget\n")
        with open(VALIDATION_PATH, "w") as file:
            file.write("context\ttarget\n")
        with open(TESTING_PATH, "w") as file:
            file.write("context\ttarget\n")
    else:
        print("---STARTING COUNTS---")
        df = pd.read_csv(TRAINING_PATH, sep='\t')
        training_count = len(df)
        print(f"Training Count: {training_count}")
        df = pd.read_csv(VALIDATION_PATH, sep='\t')
        validation_count = len(df)
        print(f"Validation Count: {validation_count}")
        df = pd.read_csv(TESTING_PATH, sep='\t')
        testing_count = len(df)
        print(f"Testing Count: {testing_count}")
        print("---------------------")
    #else:
    #    with open(JSON_PATH, "r") as file:
    #        data = file.read().strip()
    #        if data.endswith(']'):
    #            data = data[:-1]
    #    with open(JSON_PATH, "w") as file:
    #        file.write(data)

# each thread will store its items in json and csv
def qa_database_storage(context_pairs, context_id: int):
    #json_storage(context_pairs, context_id)
    csv_storage(context_pairs)

def json_storage(context_pairs, context_id: int):
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

def csv_storage(context_pairs):
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
            #print(f"Testing Count: {testing_count}")
        elif validation_count < (training_count + validation_count + testing_count) / RATIO:
            path = VALIDATION_PATH
            lock = validation_csv_lock
            validation_count += len(qa_pairs)
            #print(f"Validation Count: {validation_count}")
        else:
            path = TRAINING_PATH
            lock = training_csv_lock
            training_count += len(qa_pairs)
            #print(f"Training Count: {training_count}")
    with lock:
        data = []
        for qa_pair in qa_pairs:
            question = qa_pair.get("question")
            answer = qa_pair.get("answer")
            data.append({
                'context': f"Use the following article to answer the question: Article: {article} Question: {question}",
                'target': answer
            })
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, sep='\t', header=False, mode='a')
        
# done after threading
def post_storage():
    with open(JSON_PATH, "a") as json_file:
        json_file.write("]")
    global training_count
    global validation_count
    global testing_count
    print("---FINISHING COUNTS---")
    print(f"Training Count: {training_count}")
    print(f"Validation Count: {validation_count}")
    print(f"Testing Count: {testing_count}")
    print("---------------------")