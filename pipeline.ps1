# read config file
$config = Get-Content "pipeline_config.txt"

# create a hashtable to store configs
$configHash = @{}

# parse each line in the config file
foreach ($line in $config) {
    $key, $value = $line -split '=', 2
    $key = $key.Trim()
    $value = $value.Trim()
    $configHash[$key] = $value
}

# access config values
$db_index_start = $configHash["db_index_start"]
$article_count = $configHash["article_count"]
$thread_count = $configHash["thread_count"]
$q_eval_threshold = $configHash["q_eval_threshold"]

# run pipeline and analysis with configs
& python .\QA-Pipeline\qa_pipeline.py $db_index_start $article_count $thread_count $q_eval_threshold
#& python .\QA-Evaluator\evaluator.py -f .\QA-Output\pipeline_output.json -o .\QA-Output\data_diversity_scores.txt