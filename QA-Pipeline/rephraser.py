import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("unikei/t5-base-split-and-rephrase")
model = T5ForConditionalGeneration.from_pretrained("unikei/t5-base-split-and-rephrase")
model = model.to(device)

def rephrase(text: str) -> str:
    text_tokenized = tokenizer(text, 
                                 padding="max_length", 
                                 truncation=True,
                                 max_length=256, 
                                 return_tensors='pt').to(device)
    rephrased_tokenized = model.generate(text_tokenized['input_ids'], attention_mask = text_tokenized['attention_mask'], max_length=256, num_beams=5)
    rephrased = tokenizer.batch_decode(rephrased_tokenized, skip_special_tokens=True)
    return rephrased[0]