from transformers import pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def summarize(text: str, length: int) -> str:
    
    return summarizer(text, max_length=length, do_sample=False)[0].get('summary_text')