from typing import List
from question_generator import questiongenerator

def get_qa_pairs(summaries: List[str]):
    qg = questiongenerator.QuestionGenerator()
    for summary in summaries:
        print(qg.generate(summary, num_questions=10))