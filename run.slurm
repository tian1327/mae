#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=mae_scratch_300eps
#SBATCH --account=132713667044
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --output=slurm_log/%j_%x.log
#SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:a30:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:rtx:1              
##SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu          # Submit job to the gpu queue
#SBATCH --mail-type=END
#SBATCH --mail-user=tian.liu@tamu.edu

## ----------- execute python script
#export MASTER_ADDR=localhost
#export MASTER_PORT=29500
#export WORLD_SIZE=1
#export LOCAL_RANK=0
#export RANK=0

bash run_pretrain_tian.sh


##----------- Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"