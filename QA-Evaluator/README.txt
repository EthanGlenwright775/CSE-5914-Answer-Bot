Calculates lexical diversity metrics for the questions aspect of the synthetic dataset.
Calculates average lexical diversity metrics for the source dataset.
Calculates difference between these two sets of values. (how much more diverse the source is than the generated questions)

USE IN ROOT DIRECTORY TO RUN:
python Dataset-Evaluator/evaluator.py -f  QA-Pipeline\output\qa_pipeline_2024-02-25_06-35-43.json
Change "qa_pipeline_2024-02-25_06-35-43.json" in the second argument to the desired json.

Also supports saving to a custom path:
python Dataset-Evaluator/evaluator.py -f  QA-Pipeline\output\qa_pipeline_2024-02-25_06-35-43.json -o some\new\path.txt

Also supports changing window size for Mean Segmented TTR and Moving Average TTR, but the default window size of 100 is sufficient.