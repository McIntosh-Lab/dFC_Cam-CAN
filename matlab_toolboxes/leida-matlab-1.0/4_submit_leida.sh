#!/bin/bash
#SBATCH --account=rrg-rmcintos
#SBATCH --time=0-18:00:00            # Time limit (hh:mm:ss)
#SBATCH --ntasks=8                # Number of CPU cores
#SBATCH --mem=90G                 # Memory
#SBATCH --nodes=1                 # Number of nodes

module load matlab

matlab -nodisplay -r "run('LEiDA_Start.m'); exit;"
matlab -nodisplay -r "run('run_all_after_start.m'); exit;"
