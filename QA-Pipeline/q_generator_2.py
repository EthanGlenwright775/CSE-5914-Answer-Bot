from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

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

    qa_pair_list.append({"question": question, "answer": answer})
    return qa_pair_list