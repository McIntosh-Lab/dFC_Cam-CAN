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
module load python/3.10
python3.10 -m venv neudorf_dFC
source neudorf_dFC/bin/activate
pip install numpy scipy nibabel nilearn matplotlib pillow pandas seaborn tqdm statsmodels plotnine
pip install nctpycd python_packages/brainvistools
python -m pip install .
cd ../PLS_wrapper
python -m pip install .
cd ../PyNeudorf
python -m pip install . 
```



### Matlab:
#### LEiDA
Instructions here: https://github.com/PSYMARKER/leida-matlab

#### PLS
Instructions here: https://github.com/McIntosh-Lab/PLS

## Usage
LEiDA results used in paper are in `matlab_toolboxes/leida-matlab-1.0/res_Cam-CAN_TVB_SchaeferTian_218`

PLS in `outputs/leida-matlab/PLS`, `PLS_group_analyses`, and `outputs/nctpy/5/PLS`

NCT outputs in `outputs/nctpy`

For code review:
Get data from `~/projects/def-rmcintos/jneudorf/Cam-CAN/code_review` (behavioural data, SC matrices and rsfMRI timecourses)
Extract data/ folder from dFC_data.tar.gz into main folder.
To reproduce:
1. Figure 4 PLS behavioural correlation (A) and BSR transition probability matrix (B)
2. Figure 5 young adult behavioural correlation (B) and brain map BSR (C) plus older adult behavioural correlation (D) and brain map BSR (E)
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
