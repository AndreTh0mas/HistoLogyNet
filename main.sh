#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name SAM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=4-00:00:00
#SBATCH --error=error
#SBATCH --output=output
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_NTASKS="$SLURM_NTASKS
ulimit -s unlimited
ulimit -c unlimited

source /home/apps/DL/DL-CondaPy3.7/bin/activate
source activate /home/geetanjali.scee.iitmandi/miniconda3/envs/torch
python3 run.py > output.txt

#This is file used for submitting the job to ParamHimalaya, IIT Mandi