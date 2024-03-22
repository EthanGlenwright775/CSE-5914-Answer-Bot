import argparse
import ijson
from lexical_diversity import lex_div as ld

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def __generate_question_stats__(f_in, f_out, win_size):
    tokens = []
    with open(f_in, "rb") as file_in:
        for question in ijson.items(file_in, "item.qa_pairs.item.question"):
            tokens = tokens + ld.tokenize(question)

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
    
    return [mtld, ttr, msttr, mattr]

def __generate_source_stats__(f_in, f_out, win_size):
    mtld = []
    ttr = []
    msttr = []
    mattr = []
    with open(f_in, "rb") as file_in:
        for context in ijson.items(file_in, "item.context"):
            tokens = ld.tokenize(context)
            mtld.append(ld.mtld(tokens))
            ttr.append(ld.ttr(tokens))
            msttr.append(ld.msttr(tokens))
            mattr.append(ld.mattr(tokens))
    
    avg_mtld = sum(mtld) / len(mtld)
    avg_ttr = sum(ttr) / len(ttr)
    avg_msttr = sum(msttr) / len(msttr)
    avg_mattr = sum(mattr) / len(mattr)

    print("Source metrics")
    print("MTLD:      ", str(avg_mtld))
    print("Simple TTR:", str(avg_ttr))
    print("MSTTR " + str(win_size) +": ", avg_msttr)
    print("MATTR " + str(win_size) +": ", avg_mattr)

    with open(f_out, "a") as file_out:
        file_out.write("Source metrics\n")
        file_out.write("MTLD:       " + str(avg_mtld) + "\n")
        file_out.write("Simple TTR: " + str(avg_ttr) + "\n")
        file_out.write("MSTTR " + str(win_size) +":  " + str(avg_msttr) + "\n")
        file_out.write("MATTR " + str(win_size) +":  " + str(avg_mattr) + "\n")
    
    return [avg_mtld, avg_ttr, avg_msttr, avg_mattr]

def __generate_difference_stats__(f_out, win_size, source_stats, question_stats):
    difference = []
    for i in range(4): difference.append(source_stats[i]-question_stats[i]) 

    print("Difference")
    print("MTLD:      ", str(difference[0]))
    print("Simple TTR:", str(difference[1]))
    print("MSTTR " + str(win_size) +": ", difference[2])
    print("MATTR " + str(win_size) +": ", difference[3])

    with open(f_out, "a") as file_out:
        file_out.write("Difference\n")
        file_out.write("MTLD:       " + str(difference[0]) + "\n")
        file_out.write("Simple TTR: " + str(difference[1]) + "\n")
        file_out.write("MSTTR " + str(win_size) +":  " + str(difference[2]) + "\n")
        file_out.write("MATTR " + str(win_size) +":  " + str(difference[3]) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="File path to dataset json")
    parser.add_argument("-o", type=str, nargs='?', help="Optional file path to save program output", default="Dataset-Evaluator/eval_results_"+TIMESTAMP+".txt")
    parser.add_argument("-s", type=int, nargs='?', help="Optional window size for advanced TTR calculations, default is 100 tokens", default=100)
    args = parser.parse_args()

    f_in = args.f
    f_out = args.o
    win_size = args.s

    source_stats = __generate_source_stats__(f_in, f_out, win_size)
    question_stats = __generate_question_stats__(f_in, f_out, win_size)
    __generate_difference_stats__(f_out, win_size, source_stats, question_stats)


if __name__ == "__main__":
    main()