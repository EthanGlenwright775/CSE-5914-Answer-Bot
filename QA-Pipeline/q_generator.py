import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
model = model.to(device)

def __generate__(answer: str, context: str) -> list[dict[str, str]]:
        concat = "<answer> " + answer + " <context> " + context
        concat_tokenized = tokenizer(concat, 
            padding="max_length", 
            truncation=True,
            max_length=512, 
            return_tensors='pt').to(device)
        question_tokenized = model.generate(concat_tokenized['input_ids'], attention_mask = concat_tokenized['attention_mask'], max_length=256, num_beams=5)
        question = tokenizer.batch_decode(question_tokenized, skip_special_tokens=True)
        return question


def generate_questions_multicontext(answer_list: list[str], context_list: list[str]) -> list[dict[str, str]]:

    if not (isinstance(answer_list, list) and isinstance(context_list, list)): raise RuntimeError("Both Parameters must be lists to use multicontext.")
    if (len(answer_list) != len(context_list)): raise RuntimeError("Answer list length must match Context list length to use multicontext.")

    qa_pair_list = []

    for index in range(len(answer_list)):
        question = __generate__(answer_list[index], context_list[index])
        question = question[0]
        question = question[:question.find("?") + 1]
        qa_pair_list.append({"question": question, "answer": answer_list[index]})

    return qa_pair_list

def generate_questions_monocontext(answer_list: list[str], context: str) -> list[dict[str, str]]:

    if not isinstance(answer_list, list): raise RuntimeError("Answers must be a List.")
    if not isinstance(context, str): raise RuntimeError("Context must be a single String.")

    qa_pair_list = []

    for index in range(len(answer_list)):
        question = __generate__(answer_list[index], context)
        question = question[0]
        question = question[:question.find("?") + 1]
        qa_pair_list.append({"question": question, "answer": answer_list[index]})

    return qa_pair_list