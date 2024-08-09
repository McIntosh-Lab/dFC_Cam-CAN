# %%
import numpy as np
import pandas as pd
import re
from pathlib import Path
import pickle
import statsmodels.api as sm

# %%
#or run for 320 and 420
ATLAS = '220'
BEHAV_DIR = Path('data/behav')

QC_PARTICIPANTS_FILE = Path('data/QC_participants_ids.txt')
REMAINDER_PARTICIPANTS_FILE = Path('data/remainder_participants_ids.txt')

SUBJECT_AGES_FILE = BEHAV_DIR.joinpath('subject_ages.tsv')
SUBJECT_SEX_FILE = BEHAV_DIR.joinpath('subject_sex.tsv')
VSTM_FILE = BEHAV_DIR.joinpath('VSTMcolour/release001/summary/VSTMcolour_summary.txt')
CATTELL_FILE = BEHAV_DIR.joinpath('Cattell/release001/summary/Cattell_summary.txt')

BEHAVIOURAL_OUTPUT_DIR = Path('data/behav')
BEHAVIOURAL_OUTPUT_DIR.mkdir(exist_ok=True)
BEHAVIOURAL_OUTPUT_FILE = BEHAVIOURAL_OUTPUT_DIR.joinpath('behavioural_data.csv')

SC_DATA_DIR = Path('data/SC/SC_matrices_consistency_thresholded_0.5')
FC_DATA_DIR = Path('data/rsfMRI')
SC_DICT_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dict.pkl')
FC_DICT_FILE = FC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','FC_dict.pkl')
SC_DIST_DICT_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dist_dict.pkl')
FMRI_TIMESERIES_DICT_FILE = FC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','fmri_timeseries_dict.pkl')

# %%
# Get participants list file for QC and remainder and merge into one dataframe of subject ids as integers
subjects_ids_QC = pd.read_csv(QC_PARTICIPANTS_FILE,header=None)
subjects_ids_QC.columns = ['subject']
subjects_ids_rem = pd.read_csv(REMAINDER_PARTICIPANTS_FILE,header=None)
subjects_ids_rem.columns = ['subject']
subjects_ids = pd.concat([subjects_ids_QC,subjects_ids_rem])
subjects = subjects_ids.subject.to_list()

print('unique subjects shape')
print(subjects_ids.subject.unique().shape)

# %%
# Get subject ages as dataframe
ages = pd.read_csv(SUBJECT_AGES_FILE, sep="\t")
ages.subject = ages.subject.str.strip("sub-CC").astype(int)
print('ages shape')
print(ages.shape)
# %%
# Merge ages and subject ids that we have SC data for
subject_ages_merge = pd.merge(subjects_ids,ages)
print('subject_ages_merge shape')
print(subject_ages_merge.shape)

# %%
# Merge sex and subject ids that we have SC data for
sex = pd.read_csv(SUBJECT_SEX_FILE, sep="\t")
sex.subject = sex.subject.str.strip("sub-CC").astype(int)
print('sex shape')
print(sex.shape)

subject_ages_sex_merge = pd.merge(subject_ages_merge,sex,how='left')
print('subject_ages_sex_merge shape')
print(subject_ages_sex_merge.shape)

# %%
# Get data from VSTM file, reformat subject variable, and resave in BEHAVIOURAL_OUTPUT_DIR
vstm_scores = pd.read_csv(VSTM_FILE, sep="\t", skiprows=8, skipfooter=9, engine='python')
vstm_scores["subject"] = vstm_scores.CCID.str.strip("CC").astype(int)
vstm_scores = vstm_scores.drop('CCID', axis=1)
print('vstm_scores columns')
print(vstm_scores.columns)
print('vstm_scores shape')
print(vstm_scores.shape)
vstm_scores.to_csv(BEHAVIOURAL_OUTPUT_DIR.joinpath('vstm_scores.csv'),index=None)

# %%
# Merge the VSTM data with subjects and ages
behaviour_ages = pd.merge(subject_ages_sex_merge,vstm_scores,how='left')
print('behaviour_ages shape')
print(behaviour_ages.shape)

# %%
# Get cattell data, reformat subject variable, drop unneeded variables and save in BEHAVIOURAL_OUTPUT_DIR 
cattell_scores = pd.read_csv(CATTELL_FILE, sep="\t", skiprows=8, skipfooter=9, engine='python')
cattell_scores["subject"] = cattell_scores.CCID.str.strip("CC").astype(int)
cattell_scores = cattell_scores.drop('CCID', axis=1)
cattell_scores = cattell_scores.drop('RA', axis=1)
cattell_scores['CattellTotal'] = cattell_scores.TotalScore
cattell_scores = cattell_scores.loc[:,['subject','CattellTotal']]

print('cattell_scores shape')
print(cattell_scores.shape)

cattell_scores.to_csv(BEHAVIOURAL_OUTPUT_DIR.joinpath('CattellTotalScores.csv'),index=None)
# %%
# Merge cattell scores with rest of data
behaviour_ages = pd.merge(behaviour_ages,cattell_scores,'left')
print('behaviour_ages shape')
print(behaviour_ages.shape)

# %%
# Write behavioural data file
behaviour_ages.to_csv(BEHAVIOURAL_OUTPUT_FILE,sep=',',index=False)


# %%
# This is for QC and remainder subjects (those with nan values were removed previously during consistency thresholding)
# Get SC matrices and make lists with subjects having no nana values (those that have weights.txt files) and the subset of these with behaivour
SC_dir = SC_DATA_DIR.joinpath(f"TVBSchaeferTian{ATLAS}")
SC_matrix_paths = []
SC_distance_paths = []
SC_subjects_no_nan = []
SC_subjects_no_nan_with_behaviour = []

SC_matrix_paths = [SC_dir.joinpath(str(x),'structural_inputs/weights.txt') for x in subjects if SC_dir.joinpath(str(x), 'structural_inputs','weights.txt').exists()]
SC_distance_paths = [SC_dir.joinpath(str(x),'structural_inputs/tract_lengths.txt') for x in subjects if SC_dir.joinpath(str(x), 'structural_inputs','tract_lengths.txt').exists()]
SC_subjects_no_nan = [str(x) for x in subjects if SC_dir.joinpath(str(x), 'structural_inputs','weights.txt').exists()]
print('number of weights.txt found in SC_matrix_paths')
print(len(SC_matrix_paths))

#making list of tuples where element 0 is subject id and element 1 is numpy adjacency matrix
#re.findall("\\d+",x)[-1] uses regex to extract only numbers from string, as listed groups of numbers (so choose last 1 for subject)
SC_subjects = [re.findall("\\d+", str(x))[-1] for x in SC_matrix_paths]
SC_matrix_list = [np.genfromtxt(x) for x in SC_matrix_paths]
SC_dict = {k:v for (k,v) in zip(SC_subjects,SC_matrix_list)}
SC_distance_list = [np.genfromtxt(x) for x in SC_distance_paths]
SC_distance_dict = {k:v for (k,v) in zip(SC_subjects,SC_distance_list)}

# %%
# Check for subjects with regions having only 1 connection and exclude, then write txt file of sub list
# Also check that density of connectivity is not greater than 3 SDs away from mean
exclude_subjects_list = []
SC_dens_dict = {}
SC_dens_list = []
for s in SC_dict:
    regions_n = SC_dict[s].shape[0]
    for i in range(regions_n):
        SC_region_slice = np.copy(SC_dict[s][i,:])
        connections_n = regions_n - SC_region_slice[np.where(SC_region_slice == 0)].size
        #this flags regions with only 1 or less connections
        if connections_n < 2:
            print(f"{s} has {connections_n} connection(s) in region {i}")
            exclude_subjects_list.append(s)

    total_connections = (regions_n * (regions_n-1)) / 2
    SC_dens_dict[s] = np.where(SC_dict[s]>0)[0].shape[0] / total_connections
    SC_dens_list.append(SC_dens_dict[s])
SC_dens_mean = np.mean(SC_dens_list)
SC_dens_std = np.std(SC_dens_list)
SC_dens_ll = SC_dens_mean - 3*SC_dens_std
SC_dens_ul = SC_dens_mean + 3*SC_dens_std
for k,v in SC_dens_dict.items():
    if (v < SC_dens_ll) or (v > SC_dens_ul):
        exclude_subjects_list.append(k)
print(f"exclusion list: {exclude_subjects_list}") # ['112141']
for x in set(SC_subjects_no_nan).intersection(set(exclude_subjects_list)):
    SC_subjects_no_nan.remove(x)
print(len(SC_subjects_no_nan))

#%%
SC_dict = {k:v for (k,v) in SC_dict.items() if k in SC_subjects_no_nan}
SC_distance_dict = {k:v for (k,v) in SC_distance_dict.items() if k in SC_subjects_no_nan}

#%%
# Security risks with pickle so users should run this script to create pickle themselves
with open(SC_DICT_FILE,'wb') as f:
    pickle.dump(SC_dict, f)

#%%
with open(SC_DIST_DICT_FILE,'wb') as f:
    pickle.dump(SC_distance_dict, f)


print(len(SC_dict))

# %%
# This is for FC of QC subjects rated as good (1 or 2 or 3s that were rerated to 2) and remainder subjects predicted to be good based on ML trained on QC subejcts
# Get FC matrices and fMRI timecourses and make pickles of dictionairies with subject number as key
FC_dir = FC_DATA_DIR.joinpath(f"TVBSchaeferTian{ATLAS}")
FC_matrix_paths = []
FC_subjects_no_nan = []
FC_subjects_no_nan_with_behaviour = []

FC_matrix_paths = [FC_dir.joinpath(str(x),'functional_inputs/rfMRI_0.ica/rfMRI_0.ica_functional_connectivity.txt') for x in subjects if FC_dir.joinpath(str(x), 'functional_inputs/rfMRI_0.ica/rfMRI_0.ica_functional_connectivity.txt').exists()]
fmri_timeseries_paths = [FC_dir.joinpath(str(x),'functional_inputs/rfMRI_0.ica/rfMRI_0.ica_time_series.txt') for x in subjects if FC_dir.joinpath(str(x), 'functional_inputs/rfMRI_0.ica/rfMRI_0.ica_time_series.txt').exists()]
print('number of FC files found in FC_matrix_paths')
print(len(FC_matrix_paths))

#making list of tuples where element 0 is subject id and element 1 is numpy adjacency matrix
#re.findall("\\d+",x)[-1] uses regex to extract only numbers from string, as listed groups of numbers (so choose last 1 for subject)
FC_subjects = [re.findall("\\d+", str(x))[-3] for x in FC_matrix_paths]
print(FC_subjects)
FC_matrix_list = [np.genfromtxt(x) for x in FC_matrix_paths]
ROI_remove = [104,214] # remove globus pallidus
for i,fc in enumerate(FC_matrix_list):
    for ROI in ROI_remove:
        FC_matrix_list[i] = np.delete(FC_matrix_list[i], ROI, axis=0)
        FC_matrix_list[i] = np.delete(FC_matrix_list[i], ROI, axis=1)

FC_dict = {k:v for (k,v) in zip(FC_subjects,FC_matrix_list)}

fmri_timeseries_list = [np.genfromtxt(x).T for x in fmri_timeseries_paths]
for i,fc in enumerate(fmri_timeseries_list):
    for ROI in ROI_remove:
        fmri_timeseries_list[i] = np.delete(fmri_timeseries_list[i], ROI, axis=0)
fmri_timeseries_dict = {k:v for (k,v) in zip(FC_subjects,fmri_timeseries_list)}

# Security risks with pickle so users should run this script to create pickle themselves
with open(FC_DICT_FILE,'wb') as f:
    pickle.dump(FC_dict, f)

print(len(FC_dict))

with open(FMRI_TIMESERIES_DICT_FILE,'wb') as f:
    pickle.dump(fmri_timeseries_dict, f)

print(len(fmri_timeseries_dict))