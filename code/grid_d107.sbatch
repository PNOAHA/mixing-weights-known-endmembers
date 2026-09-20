#!/bin/bash
#SBATCH --job-name=paperA_grid
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-25
#SBATCH --output=logs/grid_%a.out
#SBATCH --error=logs/grid_%a.err
cd ~/tianyuan/paperA
mkdir -p logs results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NPROC=32
/public/software/Anaconda3/bin/python3 run_grid_d107.py ${SLURM_ARRAY_TASK_ID}
