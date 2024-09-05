#!/bin/bash
#SBATCH --account=rrg-rmcintos
#SBATCH --time=0-2:00:00            # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                # Number of CPU cores
#SBATCH --mem=4G                 # Memory
#SBATCH --nodes=1                 # Number of nodes

module load matlab

matlab -nodisplay -r "run('PLS_group_analyses.m'); exit;"
