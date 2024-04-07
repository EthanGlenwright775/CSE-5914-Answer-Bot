#!/bin/bash
#SBATCH --account=PAS2706
#SBATCH --job-name=generate_training_data
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

module load miniconda3 cuda
source activate local_2
cd /fs/ess/PAS2706/CSE-5914-Answer-Bot
bash pipeline.sh