#!/bin/bash
python3 ./QA-Pipeline/qa_pipeline.py \
    --article_index 804 \
    --article_count 76 \
    --thread_count 4 \
    --q_eval_threshold 1 \
    --output_directory "QA-Output-v1" \
    --training_file "training.tsv" \
    --validation_file "validation.tsv" \
    --testing_file "testing.tsv" \
    --article_db "cnn_news" \
    --qa_gen_method "chunk_rephrase"