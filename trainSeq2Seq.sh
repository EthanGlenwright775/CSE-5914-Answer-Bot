export HF_DATASETS_CACHE="/research/nfs_fosler_1/models/huggingface_cache"
export CUDA_VISIBLE_DEVICES=1


# Make sure LOGGER_NAME and --model_name are correct
LOGGER_NAME="t5_xl_goldFacts"
python3 lightning_t5_trainer.py \
	--model_directory "/research/nfs_fosler_1/amadh/cosi/modelStore/" \
	--model_path "/research/nfs_fosler_1/amadh/cosi/modelStore/${LOGGER_NAME}/" \
	--train_data "QA-Output/training.tsv" \
	--dev_data "QA-Output/validation.tsv" \
	--test_data "QA-Output/testing.tsv" \
	--evidence_data "data/evidence.tsv" \
	--num_turns 3 \
	--model_name "google/flan-t5-xl" \
	--generation_input ""data/llama_fine_tune_dev_set.json"" \
	--generation_output "data/results/${LOGGER_NAME}_dev_generations.tsv" \
	--data_seed 42 \
	--train_seed 42 \
	--dev_ratio 0.1 \
	--max_epochs 100 \
	--lr 0.001 \
	--batch_size 64 \
	--max_tkn_length 2048 \
	--num_workers 8 \
	--patience 10 \
	--logger_name $LOGGER_NAME \
	--notes "" \
	--gpu_num 0 \
	--run_generation