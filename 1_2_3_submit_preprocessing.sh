#!/bin/bash
#SBATCH --account=rrg-rmcintos
#SBATCH --time=0-3:00:00            # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                # Number of CPU cores
#SBATCH --mem=10G                 # Memory
#SBATCH --nodes=1                 # Number of nodes

module load StdEnv/2020 r/4.3.1 matlab/2022b.2
source neudorf_dFC/bin/activate
export R_LIBS=~/.local/R/$EBVERSIONR/
echo "Starting consistency_thresholding.py"
python -u consistency_thresholding.py
echo "Starting import_Cam-CAN_data.py"
python -u import_Cam-CAN_data.py 
echo "Starting prep_data.py"
python -u prep_data.py 
