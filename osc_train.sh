# Batch process script for use on OSC for training the chat bot

#!/bin/bash
#SBATCH --account=PAS2706
#SBATCH --job-name=train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

module load miniconda3 cuda
source activate training_1
cd /fs/ess/PAS2706/CSE-5914-Answer-Bot
bash trainSeq2Seq.sh