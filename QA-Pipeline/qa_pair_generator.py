from rephraser import rephrase
from q_generator import generate_questions_monocontext
import nltk
import re

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
CHUNK_SIZE = 200

# Implement a sliding window technique for the context

def chunk_text(text: str) -> list[str]:
    startingSentence = 0
    wordsInChunk = 0
    chunks = []
    sentences = sent_tokenize(text)
    for i in range(0,  len(sentences)):
        words = sentences[i].split()
        if wordsInChunk < CHUNK_SIZE:
            wordsInChunk += len(words)
        else:
            wordsInChunk = 0
            chunks.append(" ".join(sentences[startingSentence : i]))
            startingSentence = i
    chunks.append(" ".join(sentences[startingSentence:]))
    #for chunk in chunks:
    #    print(f"CHUNK: {chunk}")

    return chunks

def generate_qa_pairs(text: str) -> dict[str, any]:
    qa_pairs = []

    text = re.sub(r'[\t\n]', '', text)
    chunks = chunk_text(text)
    for i in range(len(chunks)):
        phrases = sent_tokenize(rephrase(chunks[i]))
        if i == len(chunks) - 1:
            qa_pairs += generate_questions_monocontext(phrases, " ".join([chunks[i - 1], chunks[i]]))
        else:
            qa_pairs += generate_questions_monocontext(phrases, " ".join([chunks[i], chunks[i+1]]))

    text_pairs = {"context": text, "qa_pairs": qa_pairs}

    return text_pairs