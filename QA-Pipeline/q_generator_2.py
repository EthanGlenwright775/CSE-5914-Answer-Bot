import torch
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
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

# Uses the second question generator that takes in ONLY context and outputs a QA pair
def generate_question_plusanswer(context: str) -> list[dict[str, str]]:
    qa_pair_list = []

    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    question_answer_split = question_answer.split(tokenizer.sep_token)
    if len(question_answer_split) != 2: return []
    question, answer = question_answer_split

    count_questions()
    if len(answer) > 0 and len(question) > 0 and eval_qa_pair(question,answer):
        qa_pair_list.append({"question": question, "answer": answer})
    else:
        count_rejected_questions()
    return qa_pair_list

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