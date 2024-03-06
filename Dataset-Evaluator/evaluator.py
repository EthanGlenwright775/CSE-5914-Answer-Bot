import argparse
import ijson
from lexical_diversity import lex_div as ld

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def generate_tokens(f_in):
    tokens = []
    with open(f_in, "rb") as file_in:
        for question in ijson.items(file_in, "item.qa_pairs.item.question"):
            tokens = tokens + ld.tokenize(question)
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="File path to dataset json")
    parser.add_argument("-o", type=str, nargs='?', help="Optional file path to save program output", default="Dataset-Evaluator/eval_results_"+TIMESTAMP+".txt")
    parser.add_argument("-s", type=int, nargs='?', help="Optional window size for advanced TTR calculations, default is 100 tokens", default=100)
    args = parser.parse_args()

    f_in = args.f
    f_out = args.o
    win_size = args.s

    tokens = generate_tokens(f_in)

    print("MTLD:      ", ld.mtld(tokens))
    print("Simple TTR:", ld.ttr(tokens))
    print("MSTTR " + str(win_size) +": ", ld.msttr(tokens, window_length=100))
    print("MATTR " + str(win_size) +": ", ld.mattr(tokens, window_length=100))

    with open(f_out, "w") as file_out:
        file_out.write("MTLD:       " + str(ld.mtld(tokens)) + "\n")
        file_out.write("Simple TTR: " + str(ld.ttr(tokens)) + "\n")
        file_out.write("MSTTR " + str(win_size) +":  " + str(ld.msttr(tokens, window_length=100)) + "\n")
        file_out.write("MATTR " + str(win_size) +":  " + str(ld.mattr(tokens, window_length=100)) + "\n")

if __name__ == "__main__":
    main()