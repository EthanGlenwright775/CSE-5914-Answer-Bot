import sys
import threading
import time
from cnn_news_db_connection import get_article
from qa_pair_generator import generate_qa_pairs
from qa_db_storage import qa_database_storage, eof_qa_database

num_threads = 10
lock = threading.Lock()
db_index_start = 0
starting_articles = 10
taken_articles = 0

def qa_pipeline_thread_task():
    global db_index_start
    global starting_articles
    global taken_articles
    while(1):
        lock.acquire()
        try:
            if taken_articles == starting_articles:
                return
            else:
                db_index = db_index_start + taken_articles
                taken_articles += 1
        finally:
            lock.release()

        # Get article from article db
        article = get_article(db_index)
        print(f"THREAD w/ db_index {db_index} has context")

        # Generate context-QA pairs from article
        context_qa_pairs = generate_qa_pairs(article)
        print(f"THREAD w/ db_index {db_index} has context and qa_pairs")

        # Store context-QA pairs in QA db
        qa_database_storage(context_qa_pairs, db_index - db_index_start)
        print(f"THREAD w/ db_index {db_index} has stored its context and qa_pairs in the DB")

def main():

    # Command line args
    if len(sys.argv) == 4:
        global db_index_start
        db_index_start = int(sys.argv[1])
        global starting_articles 
        starting_articles = int(sys.argv[2])
        global num_threads 
        num_threads = int(sys.argv[3])

    # Keep track of time
    start_time = time.time()

    # Create and start threads -> 1 thread takes 1 article through pipeline
    pipeline_threads = []
    for i in range(num_threads):
        pipeline_threads.append(threading.Thread(target=qa_pipeline_thread_task))
        pipeline_threads[i].start()

    # Wait for all threads to rejoin main thread
    for i in range(num_threads):
        pipeline_threads[i].join()
    
    # Write EOF for QA database file
    eof_qa_database()

    # Print total time
    total_runtime = time.time() - start_time
    print(f"QA_PIPELINE completed all tasks in {total_runtime}")

if __name__ == "__main__":
    main()