from rephraser import rephrase
from q_generator import generate_questions_monocontext
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def generate_qa_pairs(text: str) -> dict[str, any]:
    text_pairs = {}
    phrases = sent_tokenize(rephrase(text))
    qa_pairs = generate_questions_monocontext(phrases, text)
    text_pairs = {"context": text, "qa_pairs": qa_pairs}
    return text_pairs