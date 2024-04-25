from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Summarization model
sum_tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")
sum_model = sum_model.to(device)

q_gen_2_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
q_gen_2_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
q_gen_2_model = q_gen_2_model.to(device)

def qa_pair_summarize(self, article):
        context_q_a_pairs = []
        chunks = self.chunk_text(article, 400)
        for chunk in chunks:
            summary = self.summarize(chunk)
            context_q_a_pairs.append(self.question_gen_summaries(summary))
        return context_q_a_pairs

def summarize(text: str) -> str:
    tokens = sum_tokenizer(text, 
            padding="max_length", 
            truncation=True,
            max_length=512, 
            return_tensors='pt'
            ).to(device)
    summary_tokens = sum_model.generate(tokens['input_ids'], attention_mask = tokens['attention_mask'], max_length=512, num_beams=5)
    summary = sum_tokenizer.batch_decode(summary_tokens, skip_special_tokens=True)
    return summary[0]

def question_gen_summaries(summary: str):
    context_qa_pair_list = []
    inputs = q_gen_2_tokenizer(summary, return_tensors="pt").to(device)
    outputs = q_gen_2_model.generate(**inputs)
    question_answer = q_gen_2_tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(q_gen_2_tokenizer.pad_token, "").replace(q_gen_2_tokenizer.eos_token, "")
    question_answer_split = question_answer.split(q_gen_2_tokenizer.sep_token)
    if len(question_answer_split) != 2: return []
    question, answer = question_answer_split
    context_qa_pair_list.append({"context": summary, "question": question, "answer": answer})
    return context_qa_pair_list

print(question_gen_summaries(summarize('This is the body of the text. Please summarize this body.')))