import torch
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_eval = AutoTokenizer.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
model_eval = AutoModelForSequenceClassification.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
model_eval = model_eval.to(device)

q_eval_threshold = 1
q_count = 0
rejected_count = 0

q_count_lock = threading.Lock()
rejected_count_lock = threading.Lock()

def set_q_eval_threshold(threshold: int):
    q_eval_threshold = threshold
    print(f"Question evaluator threshold set to {q_eval_threshold}")

def eval_qa_pair(q: str, a: str):
    global q_eval_threshold
    encoded_input = tokenizer_eval(text=q, text_pair=a, truncation=True, return_tensors="pt").to(device)
    output = model_eval(**encoded_input)
    output = output[0][0][1].item()

    if output > q_eval_threshold:
        return True
    else:
        # print("q: " + q)
        # print("a: " + a)
        # print("score: " + output)
        return False

def count_questions():
    global q_count
    with q_count_lock:
        q_count += 1

def count_rejected_questions():
    global rejected_count
    with rejected_count_lock:
        rejected_count += 1

def print_q_eval_stats():
    global q_count
    global rejected_count
    print(f"Total # of questions: {q_count}")
    print(f"Total # of rejected questions: {rejected_count}")
    rejection_rate = (rejected_count / q_count) * 100
    print(f"Question rejection rate: {rejection_rate} %")

