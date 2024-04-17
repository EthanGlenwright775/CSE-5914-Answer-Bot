import argparse
from lexical_diversity import lex_div as ld
import pandas as pd

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FIN1 = "./QA-Output/testing.tsv"
FIN2 = "./QA-Output/training.tsv"
FIN3 = "./QA-Output/validation.tsv"

def __read_file__(file):
    df = pd.read_csv(file, sep='\t')
    return df['context'].tolist()

def __get_questions__(contexts):
    questions = []
    for context in contexts:
        question = context.split("Question: ")
        question = question[1].split(" Article:")
        questions.append(question[0])
    return questions


def __generate_stats__(text, f_out, win_size):
    tokens = []
    for item in text:
        tokens = tokens + ld.tokenize(item)

    mtld = ld.mtld(tokens)
    ttr = ld.ttr(tokens)
    msttr = ld.msttr(tokens, window_length=win_size)
    mattr = ld.mattr(tokens, window_length=win_size)

    print("Dataset metrics")
    print("MTLD:      ", mtld)
    print("Simple TTR:", ttr)
    print("MSTTR " + str(win_size) +": ", msttr)
    print("MATTR " + str(win_size) +": ", mattr)

    with open(f_out, "w") as file_out:
        file_out.write("Dataset metrics\n")
        file_out.write("MTLD:       " + str(mtld) + "\n")
        file_out.write("Simple TTR: " + str(ttr) + "\n")
        file_out.write("MSTTR " + str(win_size) +":  " + str(msttr) + "\n")
        file_out.write("MATTR " + str(win_size) +":  " + str(mattr) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, nargs='?', help="Optional file path to save program output", default="Dataset-Evaluator/eval_results_"+TIMESTAMP+".txt")
    parser.add_argument("-s", type=int, nargs='?', help="Optional window size for advanced TTR calculations, default is 100 tokens", default=100)
    args = parser.parse_args()

    f_out = args.o
    win_size = args.s

    contexts = []
    contexts.extend(__read_file__(FIN1))
    contexts.extend(__read_file__(FIN2))
    contexts.extend(__read_file__(FIN3))

    questions = __get_questions__(contexts)

    __generate_stats__(questions, f_out, win_size)


if __name__ == "__main__":
    main()