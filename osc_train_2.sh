#!/bin/bash
#SBATCH --account=PAS2706
#SBATCH --job-name=train_2
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

module load miniconda3 cuda
source activate training_1
cd /fs/ess/PAS2706/CSE-5914-Answer-Bot
bash trainSeq2Seq_2.sh