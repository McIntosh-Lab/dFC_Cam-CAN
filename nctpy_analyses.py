#%%
import numpy as np
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from nctpy.metrics import ave_control
from nctpy.utils import matrix_normalization
from nctpy.energies import get_control_inputs, integrate_u
import matplotlib.image as mpimg

from PyNeudorf.pls import *
from PyNeudorf import visualization

RSCRIPT='/usr/bin/Rscript'

ATLAS = '220'
BEHAVIOURAL_DATA_DIR = Path('data/behav/')
BEHAVIOURAL_DATA_FILE = BEHAVIOURAL_DATA_DIR.joinpath('behavioural_data.csv')
SC_DATA_DIR = Path('data/SC/SC_matrices_consistency_thresholded_0.5')
SC_DICT_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dict.pkl')
SC_DIST_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dist_dict.pkl')
RSFMRI_DATA_DIR = Path('data/rsfMRI/')
RSFMRI_DICT_FILE = RSFMRI_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','fmri_timeseries_dict.pkl')
OUTPUT_DIR = Path('outputs')
LEIDA_K = 5
NCTPY_OUTPUT_DIR = OUTPUT_DIR.joinpath(f'nctpy/{LEIDA_K}')
NCTPY_OUTPUT_DIR.mkdir(exist_ok=True,parents=True)
NCTPY_PLS_DIR = NCTPY_OUTPUT_DIR.joinpath('PLS')
NCTPY_PLS_DIR.mkdir(exist_ok=True)
LEIDA_OUTPUTS_DIR = Path('outputs/leida-matlab')
LEIDA_CENTROID_DIR = LEIDA_OUTPUTS_DIR.joinpath(f'K{LEIDA_K}/')
LEIDA_CENTROID_FILES = [LEIDA_CENTROID_DIR.joinpath(f'{k}_of_{LEIDA_K}_cluster_centroid_roi_values.csv') for k in range(1,LEIDA_K+1)]
PLS_GROUP_ANALYSIS_DATA_DIR = Path('PLS_group_analyses/inputs')
PLS_GROUP_ANALYSIS_DATA_DIR.mkdir(parents=True,exist_ok=True)

ITS = 1000

#%%
with open(SC_DICT_FILE, 'rb') as f:
    SC_dict = pickle.load(f)

SC_subjects = list(SC_dict.keys())
regions_n = SC_dict[SC_subjects[0]].shape[0]

#%%
filter_vars = ['age','CattellTotal','Prcsn_PerceptionTest']
behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')
subjects_Y = behaviour_ages[['subject']+filter_vars].dropna()
subjects = subjects_Y.subject.tolist()
# Also filtering by those with SC data so that we can do analyses with SC as well
subjects = [s for s in subjects if str(s) in SC_subjects]
print(len(subjects))

#%%
with open(RSFMRI_DICT_FILE, 'rb') as f:
    rsfMRI_dict = pickle.load(f)

rsfMRI_list = [v.T for k,v in rsfMRI_dict.items() if int(k) in subjects]
print(len(rsfMRI_list))
FC_subjects = [int(k) for k,v in rsfMRI_dict.items() if int(k) in subjects]

#%%
print('state transition energy analyses')
T = 1 # time horizon
dt = .001 # This is the default defined in `get_control_inputs`
timepoints_n = int(T / dt) + 1
# With rho=1 and S=np.eye(regions_n) implementing optimal control
rho = 1 # mixing parameter for state trajectory constraint (ignored with S=np.eye(regions_n))
S = np.eye(regions_n) # nodes in state trajectory to be constrained (unconstrained here)
B = np.eye(regions_n) # control node matrix. All nodes as controllers with equal weight (1)
state_activations = [np.genfromtxt(f) for f in LEIDA_CENTROID_FILES]
state_activations_bin = [np.where(activation>0, 1, 0) for activation in state_activations]

#%% network control theory calculations
# Checked reliability based on inversion and reconstruction errors below
for system in ['continuous']:
    print(f'system: {system}')
    state_transition_inversion_error_dict = {str(s):np.zeros((LEIDA_K,LEIDA_K)) for s in subjects}
    state_transition_reconstruction_error_dict = {str(s):np.zeros((LEIDA_K,LEIDA_K)) for s in subjects}
    state_transition_control_signals_dict = {str(s):np.zeros((LEIDA_K,LEIDA_K,timepoints_n,regions_n)) for s in subjects}
    state_transition_state_trajectory_dict = {str(s):np.zeros((LEIDA_K,LEIDA_K,timepoints_n,regions_n)) for s in subjects}
    state_transition_node_energy_dict = {str(s):np.zeros((LEIDA_K,LEIDA_K,regions_n)) for s in subjects}
    state_transition_energy_dict = {str(s):np.zeros((LEIDA_K,LEIDA_K)) for s in subjects}

    subject_n = 0
    total_subjects = len(subjects)
    for s in subjects:
        subject_n += 1
        print(f'subject: {s} (#{subject_n} / {total_subjects})')
        SC = SC_dict[str(s)]
        SC_norm = matrix_normalization(A=SC, c=1, system=system)
        for i in range(LEIDA_K):
            for j in range(LEIDA_K):
                if i != j:
                    x0 = state_activations[i]
                    xf = state_activations[j]

                    x, u, n_err = get_control_inputs(A_norm=SC_norm, T=T, B=B, x0=x0, xf=xf, system=system, rho=rho, S=S)
                    state_transition_inversion_error_dict[str(s)][i,j] = n_err[0]
                    state_transition_reconstruction_error_dict[str(s)][i,j] = n_err[1]
                    state_transition_control_signals_dict[str(s)][i,j,:,:] = u
                    state_transition_state_trajectory_dict[str(s)][i,j,:,:] = x

                    node_energy = integrate_u(u)
                    energy = np.sum(node_energy)

                    state_transition_node_energy_dict[str(s)][i,j,:] = node_energy
                    state_transition_energy_dict[str(s)][i,j] = energy

    with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_inversion_error_dict_{system}.pkl'), 'wb') as f:
        pickle.dump(state_transition_inversion_error_dict, f)

    with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_reconstruction_error_dict_{system}.pkl'), 'wb') as f:
        pickle.dump(state_transition_reconstruction_error_dict, f)

    with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_control_signals_dict_{system}.pkl'), 'wb') as f:
        pickle.dump(state_transition_control_signals_dict, f)

    with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_state_trajectory_dict_{system}.pkl'), 'wb') as f:
        pickle.dump(state_transition_state_trajectory_dict, f)

    with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_node_energy_dict_{system}.pkl'), 'wb') as f:
        pickle.dump(state_transition_node_energy_dict, f)

    with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_energy_dict_{system}.pkl'), 'wb') as f:
        pickle.dump(state_transition_energy_dict, f)

#%%
with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_energy_dict_continuous.pkl'), 'rb') as f:
    state_transition_energy_dict = pickle.load(f)

with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_node_energy_dict_continuous.pkl'), 'rb') as f:
    state_transition_node_energy_dict = pickle.load(f)
state_transition_5_to_3_node_energy_dict = {k:v[4,2,:] for k,v in state_transition_node_energy_dict.items()}

#%% Check inversion error and reconstruction error
error_thresh = 1e-8
state_transition_inversion_error_sig = np.array([v>error_thresh for k,v in state_transition_inversion_error_dict.items()])
state_transition_inversion_error_sig_n = np.sum(state_transition_inversion_error_sig)
print(f'inversion error number of significantly large cells: {state_transition_inversion_error_sig_n}')
state_transition_reconstruction_error_sig = np.array([v>error_thresh for k,v in state_transition_reconstruction_error_dict.items()])
state_transition_reconstruction_error_sig_n = np.sum(state_transition_reconstruction_error_sig)
print(f'reconstruction error number of significantly large cells: {state_transition_reconstruction_error_sig_n}')

#%% Check state paths
with open(NCTPY_OUTPUT_DIR.joinpath(f'state_transition_state_trajectory_dict_continuous.pkl'), 'rb') as f:
    state_transition_state_trajectory_dict_continuous = pickle.load(f)

#%%
plt.plot(state_transition_state_trajectory_dict_continuous['110056'][4,2,:,:])
plt.show()

state_activations = [np.genfromtxt(f) for f in LEIDA_CENTROID_FILES]
x0 = state_activations[4]
xf = state_activations[2]

for k,v in state_transition_state_trajectory_dict_continuous.items():
    xf_error = np.sum(v[4,2,-1,:] - xf)
    if abs(xf_error) > 1e-12:
        print(k)

#%% PLS analyses of control energy

behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')

age_ranges = [
    [0,50],
    [50,150]
]

variable_combos = [
    ['age','Prcsn_PerceptionTest'],
]

#Filtering to subjects with all of these variables' data for consistency with other analyses
filter_variables = ['age','CattellTotal','Prcsn_PerceptionTest']

def filter_dict_by_subs(dict,subs):
    return {k:v for k,v in dict.items() if int(k) in subs}

data_dicts = {
    'state_transition_5_to_3_node_energy_dict_continous':filter_dict_by_subs(state_transition_5_to_3_node_energy_dict, FC_subjects),
}

# %% 
for variables in variable_combos:
    for age_range in age_ranges:
        for data_dict_name,data_dict_value in data_dicts.items():
            print(data_dict_name)
            age_filter = (behaviour_ages['age'] > age_range[0]) & (behaviour_ages['age'] <= age_range[1])
            behaviour_ages_filtered = behaviour_ages.loc[age_filter]
            sym_matrix = False
            dir_matrix = False
            if data_dict_name in ['state_transition_energy_dict_continuous','state_transition_energy_dict_continuous_bin']:
                dir_matrix = True
            X, Y = pls_x_y_merge(data_dict_value,behaviour_ages_filtered,variables,filter_variables,sym_matrix=sym_matrix)
            print(Y.shape)
            res = pls.pls_analysis(X,Y.shape[0],1,Y,
                                    num_perm=ITS,
                                    num_boot=ITS,
                                    make_script=False)

            pls_process_results(res,variables,age_range,data_dict_name,ITS,Y.shape[0],NCTPY_PLS_DIR,printing=True,sym_matrix=sym_matrix,dir_matrix=dir_matrix)

#%% Visualization
pls_pvalues = pd.DataFrame()
for variables in variable_combos:
    variables_name = '_'.join(variables)
    for age_range in age_ranges:
        for data_dict_name,data_dict_value in data_dicts.items():
            PLS_result_dir_list = list(NCTPY_PLS_DIR.glob(f'{data_dict_name}_{variables_name}_{ITS}_its_*_subs_{age_range[0]}_to_{age_range[1]}_age_range'))
            PLS_result_dir = PLS_result_dir_list[0]
            PLS_result_dir_str = PLS_result_dir.name
            print('----------------------------------')
            print(f'analysis for {PLS_result_dir_str}')

            if len(PLS_result_dir_list) > 1:
                raise Warning('Glob found multiple matching directories with different subject numbers\n\
                               Analyzing first found directory found by glob')
            
            analysis_description = f'{data_dict_name}_{variables_name}_{age_range[0]}_to_{age_range[1]}_age_range'

            LV_n = len(variables)
            p_val_list = []
            for lv in range(LV_n):
                print(f'LV: {lv+1}')
                if LV_n > 1:
                    p_val = np.genfromtxt(PLS_result_dir.joinpath(f'{PLS_result_dir_str}_perm_p.csv'),skip_header=1,delimiter=',')[lv,0]
                else:
                    p_val = np.genfromtxt(PLS_result_dir.joinpath(f'{PLS_result_dir_str}_perm_p.csv'),skip_header=1,delimiter=',')[lv]
                print(f'p val: {p_val}')
                p_val_list.append(p_val)
                pls_pvalues.loc[lv,analysis_description] = p_val
                img = mpimg.imread(PLS_result_dir.joinpath(f'{PLS_result_dir_str}_lv{lv+1}_behav_corr.png'))
                imgplot = plt.imshow(img)
                plt.show()
                bsr = np.genfromtxt(PLS_result_dir.joinpath(f'{PLS_result_dir_str}_lv{lv+1}_bsr.csv'))
                if data_dict_name in ['ac_dict_continuous','state_transition_5_to_3_node_energy_dict_continous','state_transition_5_to_3_node_energy_dict_continous_bin']:
                    visualization.Schaefer200Cortex(bsr,PLS_result_dir.joinpath(f'{PLS_result_dir_str}_lv{lv+1}_bsr_cortex.png'),2.0,RSCRIPT)
                    visualization.SchaeferTian218_subcortical(bsr,PLS_result_dir.joinpath(f'{PLS_result_dir_str}_lv{lv+1}_bsr_subcortex.png'),2.0)

pls_pvalues = pls_pvalues.reindex(sorted(pls_pvalues), axis=1)
pls_pvalues.transpose().to_csv(NCTPY_PLS_DIR.joinpath('PLS_pvalues.csv'),sep=',')

# %% Sex analyses
sex_filter_male = behaviour_ages['sex'] == 'MALE'
behaviour_ages_male = behaviour_ages.loc[sex_filter_male]
age_filter_female = behaviour_ages['sex'] == 'FEMALE'
behaviour_ages_female = behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['state_transition_5_to_3_node_energy_dict_continous'],behaviour_ages_male,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts['state_transition_5_to_3_node_energy_dict_continous'],behaviour_ages_female,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_5_to_3 = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('LV p values\n',res_5_to_3.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_5_to_3.boot_result.orig_corr)

#%% by age groups
filter_male_YA = (behaviour_ages['sex'] == 'MALE') & (behaviour_ages['age'] < 50)
behaviour_ages_male_YA = behaviour_ages.loc[filter_male_YA]
filter_male_OA = (behaviour_ages['sex'] == 'MALE') & (behaviour_ages['age'] > 50)
behaviour_ages_male_OA = behaviour_ages.loc[filter_male_OA]

filter_female_YA = (behaviour_ages['sex'] == 'FEMALE') & (behaviour_ages['age'] < 50)
behaviour_ages_female_YA = behaviour_ages.loc[filter_female_YA]
filter_female_OA = (behaviour_ages['sex'] == 'FEMALE') & (behaviour_ages['age'] > 50)
behaviour_ages_female_OA = behaviour_ages.loc[filter_female_OA]

X_male_YA, Y_male_YA = pls_x_y_merge(data_dicts['state_transition_5_to_3_node_energy_dict_continous'],behaviour_ages_male_YA,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)
X_female_YA, Y_female_YA = pls_x_y_merge(data_dicts['state_transition_5_to_3_node_energy_dict_continous'],behaviour_ages_female_YA,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)

X_male_OA, Y_male_OA = pls_x_y_merge(data_dicts['state_transition_5_to_3_node_energy_dict_continous'],behaviour_ages_male_OA,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)
X_female_OA, Y_female_OA = pls_x_y_merge(data_dicts['state_transition_5_to_3_node_energy_dict_continous'],behaviour_ages_female_OA,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)

#YA first
Y = np.append(Y_male_YA,Y_female_YA,axis=0)

res_5_to_3_YA = pls.pls_analysis([X_male_YA,X_female_YA],[Y_male_YA.shape[0],Y_female_YA.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('YA')
print('LV p values\n',res_5_to_3_YA.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_5_to_3_YA.boot_result.orig_corr)

#OA now
Y = np.append(Y_male_OA,Y_female_OA,axis=0)

res_5_to_3_OA = pls.pls_analysis([X_male_OA,X_female_OA],[Y_male_OA.shape[0],Y_female_OA.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('OA')
print('LV p values\n',res_5_to_3_OA.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_5_to_3_OA.boot_result.orig_corr)

#%% Age and VSTM plot
from scipy.stats import pearsonr
behaviour_ages_filter = behaviour_ages['subject'].isin(subjects)
behaviour_ages_filtered = behaviour_ages.loc[behaviour_ages_filter]
age_cog_plot = sns.regplot(data=behaviour_ages_filtered,x='age',y='Prcsn_PerceptionTest',ci=None, color=(.224,.604,.694))
age_cog_plot.figure.savefig('outputs/age_by_VSTM_SC.png',dpi=600)
print(pearsonr(behaviour_ages_filtered['age'],behaviour_ages_filtered['Prcsn_PerceptionTest']))

#%% Age and VWM Median Split analysis
behaviour_ages_filtered = behaviour_ages.loc[behaviour_ages['subject'].isin(FC_subjects)]
#median = .82682
VWM_median = behaviour_ages_filtered['Prcsn_PerceptionTest'].median()

OA_filter = behaviour_ages_filtered['age'] >= 50
behaviour_ages_OA = behaviour_ages_filtered.loc[OA_filter]
YA_filter = behaviour_ages_filtered['age'] < 50
behaviour_ages_YA = behaviour_ages_filtered.loc[YA_filter]

OA_low_VWM_filter = behaviour_ages_OA['Prcsn_PerceptionTest'] <= VWM_median
behaviour_ages_OA_low_VWM = behaviour_ages_OA.loc[OA_low_VWM_filter]
OA_high_VWM_filter = behaviour_ages_OA['Prcsn_PerceptionTest'] > VWM_median
behaviour_ages_OA_high_VWM = behaviour_ages_OA.loc[OA_high_VWM_filter]
YA_low_VWM_filter = behaviour_ages_YA['Prcsn_PerceptionTest'] <= VWM_median
behaviour_ages_YA_low_VWM = behaviour_ages_YA.loc[YA_low_VWM_filter]
YA_high_VWM_filter = behaviour_ages_YA['Prcsn_PerceptionTest'] > VWM_median
behaviour_ages_YA_high_VWM = behaviour_ages_YA.loc[YA_high_VWM_filter]


for data_dict_name,data_dict in data_dicts.items():
    X_OA_low_VWM = np.array([data_dict[str(s)] for s in behaviour_ages_OA_low_VWM['subject'].tolist()])
    X_OA_high_VWM = np.array([data_dict[str(s)] for s in behaviour_ages_OA_high_VWM['subject'].tolist()])
    X_YA_low_VWM = np.array([data_dict[str(s)] for s in behaviour_ages_YA_low_VWM['subject'].tolist()])
    X_YA_high_VWM = np.array([data_dict[str(s)] for s in behaviour_ages_YA_high_VWM['subject'].tolist()])

    X_dict = {'OA_low_VWM':X_OA_low_VWM,
              'OA_high_VWM':X_OA_high_VWM,
              'YA_low_VWM':X_YA_low_VWM,
              'YA_high_VWM':X_YA_high_VWM}
    for X_name,X in X_dict.items():
        file_prefix = f'{data_dict_name}_{X_name}_{X.shape[0]}_subs'
        np.savetxt(PLS_GROUP_ANALYSIS_DATA_DIR.joinpath(f'{file_prefix}_X.csv'), X, delimiter=',')

#%% Will not work until after running `PLS_group_analyses/PLS_group_analyses.m`
#NCT_PLS_BSRs = ['PLS_group_analyses/NCT_meancentered_LV1_bsr.csv','PLS_group_analyses/NCT_meancentered_LV2_bsr.csv']
#for lv in range(1,3):
#    bsr = np.genfromtxt(f'PLS_group_analyses/NCT_meancentered_LV{lv}_bsr.csv')
#    visualization.Schaefer200Cortex(bsr,f'PLS_group_analyses/NCT_meancentered_LV{lv}_bsr_cortex.png',2.0)
#    visualization.SchaeferTian218_subcortical(bsr,f'PLS_group_analyses/NCT_meancentered_LV{lv}_bsr_subcortex.png',2.0)
