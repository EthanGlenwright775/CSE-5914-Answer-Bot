import argparse
import threading
import os
import time
import torch
import pandas as pd
import re
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from cnn_news_db_connection import cnn_get_article
from daily_mail_db_connection import daily_get_article
from cc_news_db_connection import cc_get_article

# HuggingFace pre-trained models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Question generator model used for generating from rephrased text
tokenizer_q_gen = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model_q_gen = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
model_q_gen = model_q_gen.to(device)

# Question evaluator model
tokenizer_eval = AutoTokenizer.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
model_eval = AutoModelForSequenceClassification.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
model_eval = model_eval.to(device)

# Rephrasing model
tokenizer_rephrase = T5Tokenizer.from_pretrained("unikei/t5-base-split-and-rephrase")
model_rephrase = T5ForConditionalGeneration.from_pretrained("unikei/t5-base-split-and-rephrase")
model_rephrase = model_rephrase.to(device)

# Summarization model
sum_tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")
sum_model = sum_model.to(device)

# Question generator model used for generating from summaries
q_gen_2_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
q_gen_2_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
q_gen_2_model = q_gen_2_model.to(device)

# Locks to enable multi-threaded implementation of pipeline

# Protects article db index - each thread grabs a different article
article_db_lock = threading.Lock()

# Protects total accepted question count
q_count_lock = threading.Lock()

# Protects total rejected question count
rejected_count_lock = threading.Lock()

# Protects average BERT score for question answer pairs
average_q_score_lock = threading.Lock()

# Protects ouput JSON file
output_json_lock = threading.Lock()

# Protects ouput training data file
training_csv_lock = threading.Lock()

# Protects output validation data file
validation_csv_lock = threading.Lock()

# Protects output testing data file
testing_csv_lock = threading.Lock()

# Protects training, validation, and testing example counts
example_count_lock = threading.Lock()

# Width of pipeline output headers
HEADER_WIDTH = 80

# Chunk sizes
REPHRASE_CHUNK_SIZE = 175
SUMMARIZE_CHUNK_SIZE = 400

# 80 - 10 - 10 (training - validation - testing) split of qa pairs
DATA_RATIO = 10 

class QA_Pipeline:
    def __init__(self, args):
        self.print_header("CONFIGURATION")
        self.print_pipeline_options(args)
        self.article_index = args['article_index']
        self.article_count = args['article_count']
        self.thread_count = args['thread_count']
        self.q_eval_threshold = args['q_eval_threshold']

        self.output_directory = args['output_directory']
        # create the ouput directory if it does not exist already
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.training_file = os.path.join(self.output_directory, args['training_file'])
        self.validation_file = os.path.join(self.output_directory, args['validation_file'])
        self.testing_file = os.path.join(self.output_directory, args['testing_file'])

        self.article_db = args['article_db']
        # default article db is cnn news
        self.get_article = cnn_get_article
        if self.article_db == 'daily_mail':
            self.get_article = daily_get_article
        elif self.article_db == 'cc_news':
            self.get_article = cc_get_article

        self.qa_gen_method = args['qa_gen_method']
        # default pair generation method is chunk and rephrase
        self.generate_qa_pairs = self.qa_pair_chunk_rephrase
        if self.qa_gen_method == 'summarize':
            self.generate_qa_pairs = self.qa_pair_summarize

        self.complete_articles = 0
        self.accepted_q_count = 0
        self.rejected_q_count = 0
        self.average_q_score = 0
        self.training_count = 0
        self.validation_count = 0
        self.testing_count = 0

        # Check for CUDA
        if torch.cuda.is_available():
            print("CUDA is available")
        else:
            print("CUDA is not available")
            if torch.cuda.device_count() == 0:
                print("No CUDA devices found.")
            else:
                print("CUDA device(s) found, but PyTorch is not compiled with CUDA support.")

    def print_pipeline_options(self, args):
        for arg, value in args.items():
            print(f"{arg} = {value}")

    def print_header(self, title):
        half = (HEADER_WIDTH - len(title)) // 2
        for i in range(half):
            print('=', end='')
        print(f" {title} ", end='')
        for i in range(half):
            print('=', end='')
        print()

    def run(self):
        # Ready output files
        self.pre_storage()

        # Print running pipeline
        self.print_header("PROCESSING")

        # Keep track of time at start of work
        start_time = time.time()

        # Create and start threads -> 1 thread takes 1 article through pipeline
        pipeline_threads = []
        for i in range(self.thread_count):
            pipeline_threads.append(threading.Thread(target=self.qa_pipeline_thread_task))
            pipeline_threads[i].start()

        # Wait for all threads to rejoin main thread
        for i in range(self.thread_count):
            pipeline_threads[i].join()

        # Finish output files
        self.post_storage()

        # Print finishing pipeline
        self.print_header("RESULTS OF CURRENT RUN")

        # Print Q Eval Stats
        self.print_q_eval_stats()

        # Print total time
        total_runtime = round(time.time() - start_time, 2)
        print(f"Pipeline runtime (s): {total_runtime}")
        print(f"Accepted questions per second: {round(self.accepted_q_count / total_runtime, 2)}")

    def qa_pipeline_thread_task(self):
        done = False
        while(1):
            with article_db_lock:
                if self.complete_articles < self.article_count:
                    thread_db_index = self.article_index + self.complete_articles
                    self.complete_articles += 1
                else:
                    done = True
            if done:
                return

            # Get article from article db
            article = self.get_article(thread_db_index)
            print(f"THREAD w/ db_index {thread_db_index} has context")

            # Generate context-QA pairs from article
            context_qa_pairs = self.generate_qa_pairs(article)
            print(f"THREAD w/ db_index {thread_db_index} has qa_pairs")

            # Store context-QA pairs in QA db
            self.csv_storage(context_qa_pairs)
            print(f"THREAD w/ db_index {thread_db_index} has stored its context and qa_pairs in QA-Output")
    
    def qa_pair_chunk_rephrase(self, text):
        conext_q_a_pairs = []
        text = re.sub(r'[\t\n]', '', text)
        chunks = self.chunk_text(text, REPHRASE_CHUNK_SIZE)
        for i in range(len(chunks)):
            qa_pairs = []
            answers = sent_tokenize(self.rephrase(chunks[i]))
            if i == len(chunks) - 1:
                context = " ".join([chunks[i - 1], chunks[i]])
                qa_pairs += self.question_gen_phrases(answers, context)
            else:
                context = " ".join([chunks[i], chunks[i+1]])
                qa_pairs += self.question_gen_phrases(answers, context)
            for qa_pair in qa_pairs:
                conext_q_a_pairs.append({"context": context, "question": qa_pair.get("question"), "answer": qa_pair.get("answer")})
        return conext_q_a_pairs
    
    def rephrase(self, text: str) -> str:
        text_tokenized = tokenizer_rephrase(text=text, 
                                     padding="max_length", 
                                     truncation=True,
                                     max_length=256, 
                                     return_tensors='pt').to(device)
        rephrased_tokenized = model_rephrase.generate(text_tokenized['input_ids'], attention_mask = text_tokenized['attention_mask'], max_length=256, num_beams=5)
        rephrased = tokenizer_rephrase.batch_decode(rephrased_tokenized, skip_special_tokens=True)
        return rephrased[0]
    
    def chunk_text(self, text: str, chunk_size):
        startingSentence = 0
        wordsInChunk = 0
        chunks = []
        sentences = sent_tokenize(text)
        for i in range(0,  len(sentences)):
            words = sentences[i].split()
            wordsInSentence = len(words)
            if wordsInChunk + wordsInSentence < chunk_size:
                wordsInChunk += wordsInSentence
            else:
                wordsInChunk = 0
                chunks.append(" ".join(sentences[startingSentence : i]))
                startingSentence = i
        chunks.append(" ".join(sentences[startingSentence:]))
        #for chunk in chunks:
        #    print(f"CHUNK: {chunk}")
        return chunks
    
    def question_gen_phrases(self, answer_list, context):
        qa_pair_list = []
        for index in range(len(answer_list)):
            answer = answer_list[index]
            concat = "<answer> " + answer + " <context> " + context
            concat_tokenized = tokenizer_q_gen(concat, 
                padding="max_length", 
                truncation=True,
                max_length=512, 
                return_tensors='pt').to(device)
            question_tokenized = model_q_gen.generate(concat_tokenized['input_ids'], attention_mask = concat_tokenized['attention_mask'], max_length=512, num_beams=5)
            question = tokenizer_q_gen.batch_decode(question_tokenized, skip_special_tokens=True)
            question = question[0]
            question = question[:question.find("?") + 1]
            #print(f"Context: {context}\nQuestion: {question}\nAnswer: {answer}")
            if self.eval_qa_pair(question,answer):
                self.count_accepted_questions()
                qa_pair_list.append({"question": question, "answer": answer})
            else:
                self.count_rejected_questions()

        return qa_pair_list
    
    def qa_pair_summarize(self, article):
        context_q_a_pairs = []
        chunks = self.chunk_text(article, SUMMARIZE_CHUNK_SIZE)
        for chunk in chunks:
            summary = self.summarize(chunk)
            context_q_a_pairs.append(self.question_gen_summaries(summary))
        return context_q_a_pairs

    def summarize(text: str) -> str:
        tokens = sum_tokenizer(text, 
                padding="max_length", 
                truncation=True,
                max_length=512, 
                return_tensors='pt'
                ).to(device)
        summary_tokens = sum_model.generate(tokens['input_ids'], attention_mask = tokens['attention_mask'], max_length=512, num_beams=5)
        summary = sum_tokenizer.batch_decode(summary_tokens, skip_special_tokens=True)
        return summary[0]
    
    def question_gen_summaries(summary: str):
        context_qa_pair_list = []
        inputs = q_gen_2_tokenizer(summary, return_tensors="pt").to(device)
        outputs = q_gen_2_model.generate(**inputs)
        question_answer = q_gen_2_tokenizer.decode(outputs[0], skip_special_tokens=False)
        question_answer = question_answer.replace(q_gen_2_tokenizer.pad_token, "").replace(q_gen_2_tokenizer.eos_token, "")
        question_answer_split = question_answer.split(q_gen_2_tokenizer.sep_token)
        if len(question_answer_split) != 2: return []
        question, answer = question_answer_split

        context_qa_pair_list.append({"context": summary, "question": question, "answer": answer})
        return context_qa_pair_list
    
    def eval_qa_pair(self, q: str, a: str):
        encoded_input = tokenizer_eval(text=q, text_pair=a, truncation=True, return_tensors="pt").to(device)
        output = model_eval(**encoded_input)
        output = output[0][0][1].item()
        self.adjust_average_q_score(output)
        if output > self.q_eval_threshold:
            return True
        else:
            # print("q: " + q)
            # print("a: " + a)
            # print("score: " + output)
            return False
    
    def count_accepted_questions(self):
        with q_count_lock:
            self.accepted_q_count += 1

    def count_rejected_questions(self):
        with rejected_count_lock:
            self.rejected_q_count += 1

    def adjust_average_q_score(self, score):
        with average_q_score_lock:
            self.average_q_score += score

    def print_q_eval_stats(self):
        print(f"Total # of questions: {self.accepted_q_count + self.rejected_q_count}")
        print(f"Total # of accepted questions: {self.accepted_q_count}")
        print(f"Total # of rejected questions: {self.rejected_q_count}")
        rejection_rate = (self.rejected_q_count / (self.accepted_q_count + self.rejected_q_count)) * 100
        rejection_rate = round(rejection_rate, 2)
        print(f"Question rejection rate: {rejection_rate}%")
        average_bert = self.average_q_score / (self.accepted_q_count + self.rejected_q_count)
        average_bert = round(average_bert, 2)
        print(f"Average QA pair BERT score: {average_bert}")

    # done before threading
    def pre_storage(self):
        tsv_exists = os.path.isfile(self.training_file) and os.path.isfile(self.validation_file) and os.path.isfile(self.testing_file)
        if not (tsv_exists):
            with open(self.training_file, "w") as file:
                file.write("context\ttarget\n")
            with open(self.validation_file, "w") as file:
                file.write("context\ttarget\n")
            with open(self.testing_file, "w") as file:
                file.write("context\ttarget\n")
        else:
            self.print_header("STARTING DATA")
            df = pd.read_csv(self.training_file, sep='\t')
            self.training_count = len(df)
            df = pd.read_csv(self.validation_file, sep='\t')
            self.validation_count = len(df)
            df = pd.read_csv(self.testing_file, sep='\t')
            self.testing_count = len(df)
            total_count = self.training_count + self.validation_count + self.testing_count
            print(f"Total Count: {total_count}")
            print(f"Training: {round(self.training_count / total_count * 100, 2)}%")
            print(f"Validation: {round(self.validation_count / total_count * 100, 2)}%")
            print(f"Testing: {round(self.testing_count / total_count * 100, 2)}%")

    def csv_storage(self, context_pairs):
        with example_count_lock:
            if self.testing_count < (self.training_count + self.validation_count + self.testing_count) / DATA_RATIO:
                path = self.testing_file
                lock = testing_csv_lock
                self.testing_count += len(context_pairs)
                #print(f"Testing Count: {self.testing_count}")
            elif self.validation_count < (self.training_count + self.validation_count + self.testing_count) / DATA_RATIO:
                path = self.validation_file
                lock = validation_csv_lock
                self.validation_count += len(context_pairs)
                #print(f"Validation Count: {self.validation_count}")
            else:
                path = self.training_file
                lock = training_csv_lock
                self.training_count += len(context_pairs)
                #print(f"Training Count: {self.training_count}")
        with lock:
            data = []
            for context_q_a_pair in context_pairs:
                context = context_q_a_pair.get("context")
                question = context_q_a_pair.get("question")
                answer = context_q_a_pair.get("answer")
                data.append({
                    'context': f"Use the Article to answer the Question: {question} Article: {context}",
                    'target': answer
                })
            df = pd.DataFrame(data)
            df.to_csv(path, index=False, sep='\t', header=False, mode='a')

    # done after threading
    def post_storage(self):
        self.print_header("ENDING DATA")
        total_count = self.training_count + self.validation_count + self.testing_count
        print(f"Total Count: {total_count}")
        print(f"Training: {round(self.training_count / total_count * 100, 2)}%")
        print(f"Validation: {round(self.validation_count / total_count * 100, 2)}%")
        print(f"Testing: {round(self.testing_count / total_count * 100, 2)}%")

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--article_index', type=int, default=0,
                        help='index from which to pull articles from article db')
    parser.add_argument('--article_count', type=int, default=4,
                        help='number of articles to pull from article db')
    parser.add_argument('--thread_count', type=int, default=4,
                        help='number of cpu threads to use for pipeline')
    parser.add_argument('--q_eval_threshold', type=int, default=1,
                        help='minimum BERT score for acceptance of qa-pairs into the pipeline outupt')
    parser.add_argument('--output_directory', type=str, default='QA-Output/',
                        help='file directory in which to place pipeline output files')
    parser.add_argument('--training_file', type=str, default='training.tsv',
                        help='file name for training data')
    parser.add_argument('--validation_file', type=str, default='validation.tsv',
                        help='file name for validation data')
    parser.add_argument('--testing_file', type=str, default='testing.tsv',
                        help='file name for testing data')
    parser.add_argument('--article_db', type=str, default='cnn_news',
                        help='database from which to pull articles')
    parser.add_argument('--qa_gen_method', type=str, default='chunk_rephrase',
                        help='method by which to generate qa_pairs')
    args = parser.parse_args()
    return vars(args)

def main():
    args = getParameters()
    pipeline = QA_Pipeline(args)
    pipeline.run()

if __name__ == '__main__':
	main()