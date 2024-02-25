import sys
from article_database_connection import get_article_documents
from qa_pair_generator import generate_qa_pairs
from qa_storage import qa_database_storage

def main():

    # Command line args
    if len(sys.argv) < 3:
        print("Usage: python qa_pipeline.py <db_index_start> <num_articles>")
        sys.exit(1)

    db_index_start = int(sys.argv[1])
    num_articles = int(sys.argv[2])

    # Get articles from database connection
    articles = get_article_documents(db_index_start, num_articles)

    # Print statements for debugging
    #for article in articles:
    #    print(article)
    
    # Generate QA pairs from articles
    context_qa_pairs = generate_qa_pairs(articles)

    # Store context with pairs in json db
    qa_database_storage(context_qa_pairs)

if __name__ == "__main__":
    main()