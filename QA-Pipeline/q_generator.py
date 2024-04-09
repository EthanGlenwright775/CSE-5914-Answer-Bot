import torch
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
model = model.to(device)
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

def __generate__(answer: str, context: str):
        concat = "<answer> " + answer + " <context> " + context
        concat_tokenized = tokenizer(concat, 
            padding="max_length", 
            truncation=True,
            max_length=512, 
            return_tensors='pt').to(device)
        question_tokenized = model.generate(concat_tokenized['input_ids'], attention_mask = concat_tokenized['attention_mask'], max_length=512, num_beams=5)
        question = tokenizer.batch_decode(question_tokenized, skip_special_tokens=True)
        return question

#def generate_questions_multicontext(answer_list: list[str], context_list: list[str]) -> list[dict[str, str]]:
#
#    if not (isinstance(answer_list, list) and isinstance(context_list, list)): raise RuntimeError("Both Parameters must be lists to use multicontext.")
#    if (len(answer_list) != len(context_list)): raise RuntimeError("Answer list length must match Context list length to use multicontext.")
#
#    qa_pair_list = []
#
#    for index in range(len(answer_list)):
#        question = __generate__(answer_list[index], context_list[index])
#        question = question[0]
#        question = question[:question.find("?") + 1]
#        qa_pair_list.append({"question": question, "answer": answer_list[index]})
#
#    return qa_pair_list

def generate_questions_monocontext(answer_list, context):

    if not isinstance(answer_list, list): raise RuntimeError("Answers must be a List.")
    if not isinstance(context, str): raise RuntimeError("Context must be a single String.")

    qa_pair_list = []
    
    for index in range(len(answer_list)):
        answer = answer_list[index]
        question = __generate__(answer, context)
        question = question[0]
        question = question[:question.find("?") + 1]
        #print(f"Context: {context}\nQuestion: {question}\nAnswer: {answer}")
        count_questions()
        if eval_qa_pair(question,answer):
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