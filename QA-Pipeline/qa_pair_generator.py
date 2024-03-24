from rephraser import rephrase
from q_generator import generate_questions_monocontext
import nltk
import re

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize


# Implement a sliding window technique for the context

def chunk_text(text: str, window_size: int, overlap: int = 10) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), window_size - overlap):
        chunks.append(' '.join(words[i:i + window_size]))
    return chunks


def generate_qa_pairs(text: str) -> dict[str, any]:
    text_pairs = {}
    qa_pairs = []

    text = re.sub(r'[\t\n]', '', text)
    chunks = chunk_text(text, 200, 10)
    for i in range(len(chunks)):
        phrases = sent_tokenize((rephrase(chunks[i])))
        if i == len(chunks) - 1:
            qa_pairs += generate_questions_monocontext(phrases, chunks[i-1] + chunks[i])
        else:
            qa_pairs += generate_questions_monocontext(phrases, chunks[i] + chunks[i+1])

    text_pairs.update({"context": text, "qa_pairs": qa_pairs})

    # phrases = sent_tokenize(rephrase(text))
    # qa_pairs = generate_questions_monocontext(phrases, text)
    # text_pairs = {"context": text, "qa_pairs": qa_pairs}

    return text_pairs

