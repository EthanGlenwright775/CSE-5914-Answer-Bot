from rephraser import rephrase
from q_generator_2 import generate_question_plusanswer
from summarizer import summarize
import nltk
import re

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
CHUNK_SIZE = 40
STEP = 1

# Implement a sliding window technique for the context

def chunk_text(text: str) -> list[str]:
    chunks = []
    sentences = sent_tokenize(text)

    chunk_index = 0
    loop = True
    while loop:
        chunk_size = 0
        sentence_index = 0
        chunk = ""
        while chunk_size < CHUNK_SIZE:
            chunk += sentences[chunk_index + sentence_index]
            chunk_size += len(word_tokenize(sentences[chunk_index + sentence_index]))
            sentence_index += 1
            if (chunk_index + sentence_index) >= len(sentences):
                loop = False
                break
        chunks.append(chunk)
        chunk_index += STEP
        

    return chunks

def prune_duplicates(list):
    match_index = 0
    while match_index < len(list)-1:
        search_index = match_index + 1
        while search_index < len(list):
            if list[match_index].get('question') == list[search_index].get('question'):
                del list[search_index]
            search_index += 1
        match_index += 1
    return list

def generate_qa_pairs(text: str) -> dict[str, any]:
    qa_pairs = []

    text = re.sub(r'[\t\n]', '', text)
    chunks = chunk_text(text)

    # QA generation off SOURCE
    for chunk in chunks:
        qa_pairs += generate_question_plusanswer(chunk)
    # QA generation off REPHRASED
    for chunk in chunks:
        rephrased_chunk = rephrase(chunk)
        qa_pairs += generate_question_plusanswer(rephrased_chunk)
    # QA generation off SUMMARIZED
    for chunk in chunks:
        summarized_chunk = summarize(chunk, len(word_tokenize(chunk)))
        qa_pairs += generate_question_plusanswer(summarized_chunk)

    qa_pairs = prune_duplicates(qa_pairs)
        
    text_pairs = {"context": text, "qa_pairs": qa_pairs}

    return text_pairs