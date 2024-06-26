# Script to run the qa pipeline with default settings

#!/bin/bash
python3 ./QA-Pipeline/qa_pipeline.py \
    --article_index 0 \
    --article_count 4 \
    --thread_count 4 \
    --q_eval_threshold 1 \
    --output_directory "QA-Output" \
    --training_file "training.tsv" \
    --validation_file "validation.tsv" \
    --testing_file "testing.tsv" \
    --article_db "cnn_news" \
    --qa_gen_method "rephrase"