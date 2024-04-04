#!/bin/bash
#SBATCH --account=PAS2706
#SBATCH --job-name=generate_training_data
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=1

module load miniconda3 cuda/11.8.0
source activate local
cd /fs/ess/PAS2706/CSE-5914-Answer-Bot
bash pipeline.sh