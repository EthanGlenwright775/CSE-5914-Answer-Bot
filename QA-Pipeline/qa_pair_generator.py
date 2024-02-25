from rephraser import rephrase
from q_generator import generate_questions_monocontext
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def generate_qa_pairs(texts: list[str]) -> dict[str, dict[str, str]]:
    text_pairs = {}
    for text in texts:
        phrases = sent_tokenize(rephrase(text))
        qa_pairs = generate_questions_monocontext(phrases, text)
        text_pairs[text] = qa_pairs
        # Print statements for debugging
        #for question, answer in qa_pairs.items():
        #    print("Question: " + question)
        #    print("Answer: " + answer)
    return text_pairs