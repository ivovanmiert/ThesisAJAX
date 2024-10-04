#!/bin/bash
#Set job requirements
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=trainModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A.out 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=i.a.j.van.miert@student.vu.nl

model purge 
module load 2022
module load Anaconda3/2022.05

#Your job starts in the directory where you call sbatch
cd $HOME/overview/
#Activate your environment
source activate football_classifier

srun  python -u $HOME/overview/run_classification/run_classification_balanced_different_cl.py --hidden_size $1 --num_layers $2 --dropout_rate $3 --batch_size $4 --learning_rate $5 --classification_sort $6 --hpe_sort $7
#srun  python -u $HOME/overview/run_classification/run_classification_balanced_different.py
#srun python -u $HOME/overview/run_classification/check_union.py


