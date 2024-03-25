import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# XLMRobertaTokenizerFast
# model_id = "xlm-roberta-large-finetuned-conll03-english"
# t = XLMRobertaTokenizerFast.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("unikei/t5-base-split-and-rephrase")
model = T5ForConditionalGeneration.from_pretrained("unikei/t5-base-split-and-rephrase")
model = model.to(device)

def rephrase(text: str) -> str:
    text_tokenized = tokenizer(text=text, 
                                 padding="max_length", 
                                 truncation=True,
                                 max_length=256, 
                                 return_tensors='pt').to(device)

    # Implementation using tokenizer from XMLRoberta model

    # sample_text = "This string is to be split up into chunks of 8 tokens with an overlap of 2 tokens per chunk"
    # encoded_overflow = t(sample_text, max_length=8, truncation=True, return_overflowing_tokens=True, stride=2).input_ids
    #
    # print([len(x) for x in encoded_overflow])
    # print(*t.batch_decode(encoded_overflow), sep="\n[NEW CHUNK]\n")

    rephrased_tokenized = model.generate(text_tokenized['input_ids'], attention_mask = text_tokenized['attention_mask'], max_length=256, num_beams=5)
    rephrased = tokenizer.batch_decode(rephrased_tokenized, skip_special_tokens=True)
    return rephrased[0]