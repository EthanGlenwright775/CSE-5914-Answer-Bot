# Credit to Amad Hussain for providing this code

# Script for starting up the training for the seq 2 seq model

# Make sure LOGGER_NAME and --model_name are correct
LOGGER_NAME="t5_answer_bot_v1"
python3 lightning_t5_trainer.py \
	--model_directory "QA-Bot/" \
	--model_path "QA-Bot/${LOGGER_NAME}/" \
	--train_data "QA-Output-/training.tsv" \
	--dev_data "QA-Output-/validation.tsv" \
	--test_data "QA-Output/testing.tsv" \
	--model_name "google/flan-t5-large" \
	--data_seed 42 \
	--train_seed 42 \
	--dev_ratio 0.1 \
	--max_epochs 30 \
	--lr 0.001 \
	--batch_size 16 \
	--max_tkn_length 512 \
	--num_workers 8 \
	--patience 10 \
	--logger_name $LOGGER_NAME \
	--notes "" \
	--gpu_num 0 \
	--run_training