#%% leida-matlab analyses
import numpy as np
import scipy.io as sio
from scipy import stats
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from plotnine import *
import copy
from PLS_wrapper import pls
from PyNeudorf import graphs
from PyNeudorf.pls import *
from brainvistools import visualization
import math
print('done imports')

BEHAVIOURAL_DATA_DIR = Path('data/behav')
BEHAVIOURAL_DATA_FILE = BEHAVIOURAL_DATA_DIR.joinpath('behavioural_data.csv')
LEIDA_RESULTS_DIR = Path('matlab_toolboxes/leida-matlab-1.0/res_Cam-CAN_TVB_SchaeferTian_218')
LEIDA_CLUSTERS_FILE = LEIDA_RESULTS_DIR.joinpath('LEiDA_Clusters.mat')
LEIDA_OUTPUTS_DIR = Path('outputs/leida-matlab')
LEIDA_OUTPUTS_DIR.mkdir(parents=True,exist_ok=True)
LEIDA_PLS_DIR = LEIDA_OUTPUTS_DIR.joinpath('PLS')
LEIDA_PLS_DIR.mkdir(parents=True,exist_ok=True)
SUBJECTS_FILE = Path('data/subjects.csv')
PLS_GROUP_ANALYSIS_DATA_DIR = Path('PLS_group_analyses/inputs')
PLS_GROUP_ANALYSIS_DATA_DIR.mkdir(parents=True,exist_ok=True)

ITS = 1000

K = 5
K_min = 5 #2
K_max = 5 #20

GLO_colour_hex = "409832"
DMN_colour_hex = "d9717d"
VAT_colour_hex = "a251ac"
FPN_colour_hex = "efb943"
SMV_colour_hex = "789ac0"

K5_colours_hex = [  GLO_colour_hex,
                    DMN_colour_hex,
                    VAT_colour_hex,
                    FPN_colour_hex,
                    SMV_colour_hex
                    ]

K5_colours_rgb = [to_rgb('#' + colour) for colour in K5_colours_hex]

#%% Saves state activation maps as figure
K_list = range(K_min,K_max+1)
clusters_mat = sio.loadmat(LEIDA_CLUSTERS_FILE, simplify_cells=True)
for k in K_list:
    LEiDA_centroids_dir = LEIDA_OUTPUTS_DIR.joinpath(f'K{k}')
    LEiDA_centroids_dir.mkdir(exist_ok=True)
    k_idx = k - 2 #index 1 (index 0 for python) in .mat starts at state 2
    cluster_centroid_roi_values = clusters_mat['Kmeans_results'][k_idx]['C']
    print('k',k)
    print("cluster_centroid_roi_values.shape",cluster_centroid_roi_values.shape)
    cluster_centroid_roi_values[0] *= -1
    cluster_centroid_roi_values_pos_bin = np.where(cluster_centroid_roi_values>0, 1.0, 0.0)
    for state in range(k):
        np.savetxt(LEiDA_centroids_dir.joinpath(f'{state+1}_of_{k}_cluster_centroid_roi_values.csv'),
                   cluster_centroid_roi_values[state],delimiter=',')

        if k == 5:
            visualization.vis_cortex(cluster_centroid_roi_values_pos_bin[state],
                                            LEiDA_centroids_dir.joinpath(f'{state+1}_of_{k}_cluster_centroid_roi_values_pos_bin_cortex.png'),
                                            thresh=0.0,
                                            pos_colour=K5_colours_hex[state],
                                            bg_colour='transparent'
                                            )
            visualization.vis_subcortical(cluster_centroid_roi_values_pos_bin[state],
                                            LEiDA_centroids_dir.joinpath(f'{state+1}_of_{k}_cluster_centroid_roi_values_pos_bin_subcortex.png'),
                                            thresh=0.001,
                                            pos_colours=[(0,0,0),K5_colours_rgb[state]],
                                            save_cmap=True)

#%% Loads Transition Probability matrix for each subject and adds to dictionary
TM_mat = sio.loadmat(LEIDA_RESULTS_DIR.joinpath(f'K{K}/LEiDA_Stats_TransitionMatrix.mat'), simplify_cells=True)
trans_prob_norm = TM_mat['TMnorm'].astype(float)
trans_prob_normlist = [trans_prob_norm[s,:,:] for s in range(trans_prob_norm.shape[0])]
subjects = np.genfromtxt(SUBJECTS_FILE,delimiter=',',dtype=int)
TP_dict = {subjects[i]:v for i,v in enumerate(trans_prob_normlist)}

#%% Setup dictionarys for running PLS analyses
behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')

age_ranges = [
    [0,150],
]
variable_combos = [
    ['age'],
]

#Filtering to subjects with all of these variables' data for consistency with other analyses
filter_variables = ['age','CattellTotal','Prcsn_PerceptionTest']

data_dicts = {f'TP_dict_K{K}':TP_dict}

# %%  PLS analysis of transition probability matrix
for variables in variable_combos:
    for age_range in age_ranges:
        for data_dict_name,data_dict_value in data_dicts.items():
            print(data_dict_name)
            age_filter = (behaviour_ages['age'] > age_range[0]) & (behaviour_ages['age'] <= age_range[1])
            behaviour_ages_filtered = behaviour_ages.loc[age_filter]
            sym_matrix = False
            dir_matrix = True
            X, Y = pls_x_y_merge(data_dict_value,behaviour_ages_filtered,variables,filter_variables,sym_matrix=sym_matrix)
            print(Y.shape)
            res = pls.pls_analysis(X,Y.shape[0],1,Y,
                                    num_perm=ITS,
                                    num_boot=ITS,
                                    make_script=False)

            pls_process_results(res,variables,age_range,data_dict_name,ITS,Y.shape[0],LEIDA_PLS_DIR,printing=True,sym_matrix=sym_matrix,dir_matrix=dir_matrix)

#%% Plot BSR TP matrix
bsrs = np.genfromtxt(LEIDA_PLS_DIR.joinpath(f'TP_dict_K{K}_age_1000_its_197_subs_0_to_150_age_range/TP_dict_K{K}_age_1000_its_197_subs_0_to_150_age_range_lv1_bsr.csv'),delimiter=' ')
bsrs[np.where(np.abs(bsrs) < 2)] = 0
bsrs = bsrs.reshape((K,K))
abs_min_bsrs = np.abs(np.min(bsrs))
cmap = LinearSegmentedColormap.from_list('greyscale', [(.188,.533,.639),(.698,.757,.463),(.886,.761,.133),(.922,.0,.02)],N=100)
ax = sns.heatmap(bsrs, cmap=cmap,vmin=-1*abs_min_bsrs,vmax=abs_min_bsrs)
ax.set_xticklabels(list(range(1,K+1)))
ax.set_yticklabels(list(range(1,K+1)), rotation=0)
#ax.collections[0].colorbar.set_label("BSR")
plt.savefig(f'outputs/leida-matlab/TP_dict_K{K}_age_1000_its_197_subs_0_to_150_age_range_lv1_bsr_matrix.png',dpi=300)

#%% Fractional Occupancy PLS
behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')

age_ranges = [
    [0,150],
]

variable_combos = [
    ['age'],
    ['Prcsn_PerceptionTest'],
    ['age','Prcsn_PerceptionTest'],
]

#Filtering to subjects with all of these variables' data for consistency with other analyses
filter_variables = ['age','CattellTotal','Prcsn_PerceptionTest']

fractional_occupancy = sio.loadmat(LEIDA_RESULTS_DIR.joinpath('LEiDA_Stats_FracOccup.mat'), simplify_cells=True)['P'].astype('float')
print(fractional_occupancy.shape)
fractional_occupancy_K5 = fractional_occupancy[:,4,:5]
fractional_occupancy_K5_dict = {}
for i,s in enumerate(subjects):
    fractional_occupancy_K5_dict[str(s)] = fractional_occupancy_K5[i]

data_dicts = {'fractional_occupancy_K5':fractional_occupancy_K5_dict}

#%% 
for variables in variable_combos:
    variables_name = '_'.join(variables)
    for age_range in age_ranges:
        for data_dict_name,data_dict_value in data_dicts.items():
            print(data_dict_name)
            age_filter = (behaviour_ages['age'] > age_range[0]) & (behaviour_ages['age'] <= age_range[1])
            behaviour_ages_filtered = behaviour_ages.loc[age_filter]
            sym_matrix = False
            dir_matrix = False
            X, Y = pls_x_y_merge(data_dict_value,behaviour_ages_filtered,variables,filter_variables,sym_matrix=sym_matrix)
            print(Y.shape)
            res = pls.pls_analysis(X,Y.shape[0],1,Y,
                                    num_perm=ITS,
                                    num_boot=ITS,
                                    make_script=False)

            pls_process_results(res,variables,age_range,data_dict_name,ITS,Y.shape[0],LEIDA_PLS_DIR,printing=True,sym_matrix=sym_matrix,dir_matrix=dir_matrix)

#%% Sex analysis
sex_filter_male = behaviour_ages['sex'] == 'MALE'
behaviour_ages_male = behaviour_ages.loc[sex_filter_male]
age_filter_female = behaviour_ages['sex'] == 'FEMALE'
behaviour_ages_female = behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['fractional_occupancy_K5'],behaviour_ages_male,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts['fractional_occupancy_K5'],behaviour_ages_female,['age','Prcsn_PerceptionTest'],['age','Prcsn_PerceptionTest'],sym_matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_FO = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('LV p values\n',res_FO.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.orig_corr)
print('LV llcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.llcorr)
print('LV ulcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.ulcorr)

#%%
sex_filter_male = behaviour_ages['sex'] == 'MALE'
behaviour_ages_male = behaviour_ages.loc[sex_filter_male]
age_filter_female = behaviour_ages['sex'] == 'FEMALE'
behaviour_ages_female = behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['fractional_occupancy_K5'],behaviour_ages_male,['age'],['age'],sym_matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts['fractional_occupancy_K5'],behaviour_ages_female,['age'],['age'],sym_matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_FO = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('LV p values\n',res_FO.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.orig_corr)
print('LV llcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.llcorr)
print('LV ulcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.ulcorr)

#%%
sex_filter_male = behaviour_ages['sex'] == 'MALE'
behaviour_ages_male = behaviour_ages.loc[sex_filter_male]
age_filter_female = behaviour_ages['sex'] == 'FEMALE'
behaviour_ages_female = behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts['fractional_occupancy_K5'],behaviour_ages_male,['Prcsn_PerceptionTest'],['Prcsn_PerceptionTest'],sym_matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts['fractional_occupancy_K5'],behaviour_ages_female,['Prcsn_PerceptionTest'],['Prcsn_PerceptionTest'],sym_matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_FO = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('LV p values\n',res_FO.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.orig_corr)
print('LV llcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.llcorr)
print('LV ulcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.ulcorr)

#%%
data_dicts = {f'TP_dict_K{K}':TP_dict}

sex_filter_male = behaviour_ages['sex'] == 'MALE'
behaviour_ages_male = behaviour_ages.loc[sex_filter_male]
age_filter_female = behaviour_ages['sex'] == 'FEMALE'
behaviour_ages_female = behaviour_ages.loc[age_filter_female]

X_male, Y_male = pls_x_y_merge(data_dicts[f'TP_dict_K{K}'],behaviour_ages_male,['age'],['age'],sym_matrix=False)
X_female, Y_female = pls_x_y_merge(data_dicts[f'TP_dict_K{K}'],behaviour_ages_female,['age'],['age'],sym_matrix=False)

Y = np.append(Y_male,Y_female,axis=0)

res_FO = pls.pls_analysis([X_male,X_female],[Y_male.shape[0],Y_female.shape[0]],1,Y,
                       num_perm=ITS,
                       num_boot=ITS,
                       make_script=False)

print('LV p values\n',res_FO.perm_result.sprob)
print('LV orig_corr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.orig_corr)
print('LV llcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.llcorr)
print('LV ulcorr (behavioural correlations). Look at first column for LV1.\n',res_FO.boot_result.ulcorr)

#%% Age and VWM Median Split analysis
behaviour_ages_filtered = behaviour_ages.loc[behaviour_ages['subject'].isin(subjects)]
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

fractional_occupancy_K5_dict = {int(k):v for k,v in fractional_occupancy_K5_dict.items()}
data_dicts = {'fractional_occupancy_K5':fractional_occupancy_K5_dict,
              f'TP_dict_K{K}':TP_dict}

for data_dict_name,data_dict in data_dicts.items():
    if data_dict_name == f'TP_dict_K{K}':
        X_OA_low_VWM = np.array([data_dict[s].flatten() for s in behaviour_ages_OA_low_VWM['subject'].tolist()])
        X_OA_high_VWM = np.array([data_dict[s].flatten() for s in behaviour_ages_OA_high_VWM['subject'].tolist()])
        X_YA_low_VWM = np.array([data_dict[s].flatten() for s in behaviour_ages_YA_low_VWM['subject'].tolist()])
        X_YA_high_VWM = np.array([data_dict[s].flatten() for s in behaviour_ages_YA_high_VWM['subject'].tolist()])
    else:
        X_OA_low_VWM = np.array([data_dict[s] for s in behaviour_ages_OA_low_VWM['subject'].tolist()])
        X_OA_high_VWM = np.array([data_dict[s] for s in behaviour_ages_OA_high_VWM['subject'].tolist()])
        X_YA_low_VWM = np.array([data_dict[s] for s in behaviour_ages_YA_low_VWM['subject'].tolist()])
        X_YA_high_VWM = np.array([data_dict[s] for s in behaviour_ages_YA_high_VWM['subject'].tolist()])


    X_dict = {'OA_low_VWM':X_OA_low_VWM,
              'OA_high_VWM':X_OA_high_VWM,
              'YA_low_VWM':X_YA_low_VWM,
              'YA_high_VWM':X_YA_high_VWM}
    for X_name,X in X_dict.items():
        file_prefix = f'{data_dict_name}_{X_name}_{X.shape[0]}_subs'
        np.savetxt(PLS_GROUP_ANALYSIS_DATA_DIR.joinpath(f'{file_prefix}_X.csv'), X, delimiter=',')

#%% Age and VSTM plot
from scipy.stats import pearsonr
behaviour_ages_filter = behaviour_ages['subject'].isin(subjects.astype(int))
behaviour_ages_filtered = behaviour_ages.loc[behaviour_ages_filter]
age_cog_plot = sns.regplot(data=behaviour_ages_filtered,x='age',y='Prcsn_PerceptionTest',ci=None, color=(.224,.604,.694))
age_cog_plot.figure.savefig('outputs/age_by_VSTM_dFC.png',dpi=600)
print(pearsonr(behaviour_ages_filtered['age'],behaviour_ages_filtered['Prcsn_PerceptionTest']))

#%% demographic info
print('overall age range, mean and SD')
print('range',behaviour_ages_filtered.age.min(),'to',behaviour_ages_filtered.age.max())
print('mean',behaviour_ages_filtered.age.mean())
print('SD',behaviour_ages_filtered.age.std())
print('female:',len(behaviour_ages_filtered.loc[behaviour_ages_filtered['sex'] == 'FEMALE']))
print('male:',len(behaviour_ages_filtered.loc[behaviour_ages_filtered['sex'] == 'MALE']))

YA_filter = behaviour_ages_filtered['age'] < 50
behaviour_ages_filtered_YA = behaviour_ages_filtered.loc[YA_filter]
OA_filter = behaviour_ages_filtered['age'] > 50
behaviour_ages_filtered_OA = behaviour_ages_filtered.loc[OA_filter]

print('young adult age range, mean and SD')
print('range',behaviour_ages_filtered_YA.age.min(),'to',behaviour_ages_filtered_YA.age.max())
print('mean',behaviour_ages_filtered_YA.age.mean())
print('SD',behaviour_ages_filtered_YA.age.std())
print('female:',len(behaviour_ages_filtered_YA.loc[behaviour_ages_filtered_YA['sex'] == 'FEMALE']))
print('male:',len(behaviour_ages_filtered_YA.loc[behaviour_ages_filtered_YA['sex'] == 'MALE']))


print('older adult age range, mean and SD')
print('range',behaviour_ages_filtered_OA.age.min(),'to',behaviour_ages_filtered_OA.age.max())
print('mean',behaviour_ages_filtered_OA.age.mean())
print('SD',behaviour_ages_filtered_OA.age.std())
print('female:',len(behaviour_ages_filtered_OA.loc[behaviour_ages_filtered_OA['sex'] == 'FEMALE']))
print('male:',len(behaviour_ages_filtered_OA.loc[behaviour_ages_filtered_OA['sex'] == 'MALE']))
