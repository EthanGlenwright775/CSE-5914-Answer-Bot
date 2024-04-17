from rephraser import rephrase
from q_generator import generate_questions_monocontext
from summarizer import condense
import nltk
import re

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
CHUNK_SIZE = 200

# Implement a sliding window technique for the context

def chunk_text(text: str):
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

def generate_qa_pairs(text: str):
    conext_q_a_pairs = []

    text = re.sub(r'[\t\n]', '', text)
    chunks = chunk_text(text)
    for i in range(len(chunks)):
        qa_pairs = []
        answers = sent_tokenize(rephrase(chunks[i]))
        if i == len(chunks) - 1:
            context = " ".join([chunks[i - 1], chunks[i]])
            qa_pairs += generate_questions_monocontext(answers, context)
        else:
            context = " ".join([chunks[i], chunks[i+1]])
            qa_pairs += generate_questions_monocontext(answers, context)
        for qa_pair in qa_pairs:
            conext_q_a_pairs.append({"context": context, "question": qa_pair.get("question"), "answer": qa_pair.get("answer")})
        
    #context = condense(text)
    #answers = sent_tokenize(text)
    #qa_pairs = generate_questions_monocontext(answers, context)
    #for pair in conext_q_a_pairs:
    #    print(f"Context: {pair.get("context")}\nQuestion: {pair.get("question")}\nAnswer: {pair.get("answer")}\n")

    return conext_q_a_pairs