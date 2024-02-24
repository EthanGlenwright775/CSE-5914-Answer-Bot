from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")

def __generate__(answer, context):
        concat = "<answer> " + answer + " <context> " + context
        concat_tokenized = tokenizer(concat, 
                                        padding="max_length", 
                                        truncation=True,
                                        max_length=256, 
                                        return_tensors='pt')
        question_tokenized = model.generate(concat_tokenized['input_ids'], attention_mask = concat_tokenized['attention_mask'], max_length=256, num_beams=5)
        question = tokenizer.batch_decode(question_tokenized, skip_special_tokens=True)
        return question


def generate_questions_multicontext(answer_list, context_list):

    if not (isinstance(answer_list, list) and isinstance(context_list, list)): raise RuntimeError("Both Parameters must be lists to use multicontext.")
    if (len(answer_list) != len(context_list)): raise RuntimeError("Answer list length must match Context list length to use multicontext.")

    question_list = []

    for index in range(len(answer_list)):
        question = __generate__(answer_list[index], context_list[index])
        question_list.append(question)

    return question_list

def generate_questions_monocontext(answer_list, context):

    if not isinstance(answer_list, list): raise RuntimeError("Answers must be a List.")
    if not isinstance(context, str): raise RuntimeError("Context must be a single String.")

    question_list = []

    for index in range(len(answer_list)):
        question = __generate__(answer_list[index], context)
        question_list.append(question)

    return question_list