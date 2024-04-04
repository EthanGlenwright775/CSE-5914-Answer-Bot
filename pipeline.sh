#!/bin/bash
# pull command line arguments from config
source pipeline_config.txt
# run qa pipeline
python ./QA-Pipeline/qa_pipeline.py "$db_index_start" "$article_count" "$thread_count" "$q_eval_threshold"
# run data diversity analysis on pipeline output
#python ./QA-Evaluator/evaluator.py -f  ./QA-Output/pipeline_output.json -o ./QA-Output/data_diversity_scores.txt