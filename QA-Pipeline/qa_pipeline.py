import sys
from article_database_connection import get_article_summaries
from qa_pair_generator import get_qa_pairs

def main():

    # Command line args
    if len(sys.argv) < 3:
        print("Usage: python qa_pipeline.py <db_index_start> <num_articles>")
        sys.exit(1)

    db_index_start = int(sys.argv[1])
    num_articles = int(sys.argv[2])

    # Get articles from database connection
    articles = get_article_summaries(db_index_start, num_articles)

    # Generate QA pairs from summaries
    qa_pairs = get_qa_pairs(articles)

if __name__ == "__main__":
    main()