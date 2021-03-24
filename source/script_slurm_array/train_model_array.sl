#!/bin/bash
#SBATCH --job-name=train_model   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --mem-per-cpu=8gb           # Memory per processor
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.out    # Standard output and error log
#SBATCH --array=1-75                 # Array range

PARAMS=$(python /home/majacques/projects/CODEX/source/generate_paramString.py $SLURM_ARRAY_TASK_ID)
python /home/majacques/projects/CODEX/source/train_model_lightning.py -d /home/majacques/projects/CODEX/data/erk_p21_EtopEGF_noInsulin.zip -e 300 --logdir /home/majacques/projects/CODEX/logs_pl/array_$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID --ngpu 0 $PARAMS
