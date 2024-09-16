# Analysis code for: "Dynamic network features of functional and structural brain networks support visual working memory in aging adults"

This repository contains the code used for analyses in:

Neudorf, J., Shen, K., & McIntosh, A. R. (2024). Dynamic network features of functional and structural brain networks support visual working memory in aging adults. bioRxiv. https://doi.org/10.1101/2024.07.30.605891

## Environment installation instructions:
### Python:
#### Local Bash
```bash
git clone https://github.com/McIntosh-Lab/dFC_Cam-CAN.git
cd dFC_Cam-CAN
conda create -n neudorf_dFC -c conda-forge python=3.10 numpy scipy nibabel nilearn matplotlib pillow pandas seaborn tqdm statsmodels plotnine
conda activate neudorf_dFC
pip install nctpy
python -m pip install matlabengine=9.13.11 #matlab 2022b.2
cd python_packages/brainvistools
python -m pip install .
cd ../PLS_wrapper
python -m pip install .
cd ../PyNeudorf
python -m pip install .
```

#### HPC (Alliance Canada) Bash
```bash
git clone https://github.com/McIntosh-Lab/dFC_Cam-CAN.git
cd dFC_Cam-CAN
module load StdEnv/2020 matlab/2022b.2 python/3.10
python3.10 -m venv neudorf_dFC
source neudorf_dFC/bin/activate
pip install numpy scipy nibabel nilearn matplotlib==3.6.2 pillow pandas seaborn==0.12.1 tqdm statsmodels plotnine==0.12.3 certifi
pip install nctpy
#edit line below by finding matlabroot using `which matlab`, and substitute `matlab` executable with `glnxa64`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/restricted.computecanada.ca/easybuild/software/2020/x86-64-v3/Core/matlab/2022b.2/bin/glnxa64
#set matlabengine version number to match matlab version 2022b.2
python -m pip install matlabengine==9.13.11
cd python_packages/brainvistools
python -m pip install .
cd ../PLS_wrapper
python -m pip install .
cd ../PyNeudorf
python -m pip install . 
```

### R:
#### Local Bash
```bash
Rscript install_packages.r
```

#### HPC (Alliance Canada) Bash
```bash
module load StdEnv/2020 r/4.3.1 gcc/9.3.0 gdal/3.5.1 udunits/2.2.28
mkdir -p ~/.local/R/$EBVERSIONR
export R_LIBS=~/.local/R/$EBVERSIONR/
Rscript install_packages.r
```

### Matlab:
#### Local Bash
Follow instructions here and add `plscmd` folder to startup.m: https://github.com/McIntosh-Lab/PLS

#### HPC (Alliance Canada) Bash
```bash
git clone https://github.com/McIntosh-Lab/PLS
cp -r PLS/plscmd ~/matlab
echo "addpath(genpath('~/matlab/plscmd'))" >> ~/matlab/startup.m
echo "addpath(genpath('${PWD}/python_packages/PLS_wrapper'))" >> ~/matlab/startup.m
```

## Usage
LEiDA results used in paper are in `matlab_toolboxes/leida-matlab-1.0/res_Cam-CAN_TVB_SchaeferTian_218`

PLS in `outputs/leida-matlab/PLS`, `PLS_group_analyses`, and `outputs/nctpy/5/PLS`

NCT outputs in `outputs/nctpy`

For code review:
Get data from `~/projects/def-rmcintos/jneudorf/Cam-CAN/code_review` (behavioural data, SC matrices and rsfMRI timecourses)
Extract data/ folder from dFC_data.tar.gz into main folder.
To reproduce:
1. Figure 4 PLS behavioural correlation (A; `outputs/leida-matlab/PLS/TP_dict_K5_age_1000_its_197_subs_0_to_150_age_range/TP_dict_K5_age_1000_its_197_subs_0_to_150_age_range_lv1_behav_corr.png`) and BSR transition probability matrix (B; `outputs/leida-matlab/PLS/TP_dict_K5_age_1000_its_197_subs_0_to_150_age_range/TP_dict_K5_age_1000_its_197_subs_0_to_150_age_range_lv1_bsr_matrix.png`)
2. Figure 5 young adult behavioural correlation (B; `outputs/nctpy/5/PLS/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_114_subs_0_to_50_age_range/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_114_subs_0_to_50_age_range_lv1_behav_corr.png`) and brain map BSR (C; `outputs/nctpy/5/PLS/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_114_subs_0_to_50_age_range/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_114_subs_0_to_50_age_range_lv1_bsr_cortex.png` and `outputs/nctpy/5/PLS/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_114_subs_0_to_50_age_range/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_114_subs_0_to_50_age_range_lv1_bsr_subcortex.png`) plus older adult behavioural correlation (D; `outputs/nctpy/5/PLS/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_83_subs_50_to_150_age_range/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_83_subs_50_to_150_age_range_lv1_behav_corr.png`) and brain map BSR (E; `outputs/nctpy/5/PLS/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_83_subs_50_to_150_age_range/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_83_subs_50_to_150_age_range_lv1_bsr_cortex.png` and `outputs/nctpy/5/PLS/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_83_subs_50_to_150_age_range/state_transition_5_to_3_node_energy_dict_continous_age_Prcsn_PerceptionTest_1000_its_83_subs_50_to_150_age_range_lv1_bsr_subcortex.png`)
Note that the exact BSRs will be different than in the paper because of resampling.
PLS outputs saved in `outputs/leida-matlab/PLS` and `outputs/nctpy/5/PLS`

Order to run code:

1. `consistency_thresholding.py`
2. `import_Cam-CAN_data.py`
3. `prep_data.py`
4. LEiDA toolbox in `matlab_toolboxes/leida-matlab-1.0/`. Info is in README.md as well, but edit paths at the top of `LEiDA_Start.m` and `run_all_after_start.m`. Then run `LEiDA_Start.m` followed by `run_all_after_start.m`. You can change `n_permutations` and `n_bootstraps` to a lower number for testing to make these run faster.
5. `leida-matlab_analyses.py` will perform PLS analyses on the LEiDA outputs and produce images used in figures 2, 3, and 4.
6. `nctpy_analyses.py` will perform the network control theory simulation and the PLS analysis of the state 5 to 3 transition (figure 5).
7. `PLS_group_analyses/PLS_group_analyses.m` performs secondary mean-centred PLS analyses for LEiDA and NCT analyses.

For HPC (Alliance Canada), use sbatch submission scripts provided:
 - `1_2_3_submit_preprocessing.sh`
 - `matlab_toolboxes/leida-matlab-1.0/4_submit_leida.sh` (refer to step 4. above and edit paths)
 - `5_submit_leida_analyses.sh`
 - `6_submit_nctpy_analyses.sh` (change `RSCRIPT='/usr/bin/Rscript'` at top of `nctpy_analyses.py` to `RSCRIPT='/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/r/4.3.1/bin/Rscript'`
 - `PLS_group_analyses/7_submit_group_analysis.sh`
