# These functions are made for working with PLS_wrapper and specific to my workflow
from plotnine import *
import copy
import math
from PLS_wrapper import pls
from PyNeudorf import graphs
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def pls_x_y_merge(X_dict,Y_df,vars,filter_vars,sym_matrix=True):
    """Purpose: to merge by subject a X dictionary with subject number (string) 
    as keys and data as values with a Y pandas DataFrame containing a `subject` 
    variable with the subject number (int) along with other variables of interest.
    `vars` will select variables of interest and `filter_vars` will apply filtering
    based on missing data in those variables. `sym_matrix` will take the flattened
    upper triangle of the matrix if set to True
    Parameters
    ----------
    X_dict      :   dictionary with subject number (string) as keys
                    and ndarray data as values
    Y_df        :   pandas DataFrame containing a `subject` variable with the subject 
                    number (int) along with other variables of interest
    vars        :   list of strings specifying variables of interest in Y_df (other 
                    than `subject`)
    filter_vars :   list of strings specifying variables to use for apply filtering
                    based on missing data in those variables
    sym_matrix  :   bool. will take the flattened upper triangle of the matrix if 
                    set to True
                            
    Returns
    -------
    X           :   1-d array with data flattened
    Y           :   pandas DataFrame with variables `subject` and those contained
                    in `vars`
    """

    SC_subjects = list(X_dict.keys())
    SC_subjects_int = [int(x) for x in SC_subjects]
    SC_subjects_df = pd.DataFrame({'subject':SC_subjects_int})
    Y_df_merged = pd.merge(SC_subjects_df,Y_df,on=['subject'],how='left')
    subjects_Y = Y_df_merged[['subject']+filter_vars].dropna()
    subjects_Y = subjects_Y[['subject']+vars]
    subjects = subjects_Y.subject.tolist()
    Y_df = subjects_Y[vars]
    Y = np.array(Y_df)
    if sym_matrix:
        X_matrices = [v for (k,v) in X_dict.items() if int(k) in subjects]
        X_trius = np.array([graphs.matrix_to_flat_triu(mat) for mat in X_matrices])
        X = X_trius.copy()
    else:
        X = np.array([v.flatten() for (k,v) in X_dict.items() if int(k) in subjects])
    return X, Y

def pls_process_results(res,vars,age_range,X_name,its,subjects_n,output_root,printing=True,sym_matrix=True,dir_matrix=False):
    """Takes the result from a PLS_wrapper analysis (`res`) and saves a number of
    outputs including csv files and png images.
    Parameters
    ----------
    res         :   Dict2Object with PLS_wrapper analysis results.
    vars        :   list of strings specifying variables of interest in analysis
                    (used for file naming)
    age_range   :   list of upper and lower age limits (floats)
                    (used for file naming)
    X_name      :   string. X variable name (used for file naming)
    its         :   int. number of iterations used in PLS (used for file naming)
    subjects_n  :   int. number of subjects in data (used for file naming)
    output_root :   string or pathlib Path. Where to save outputs.
                    Will first create folder in this path based on above variables
    printing    :   bool. Whether to print some outputs
    sym_matrix  :   bool. Whether data is a symmetric matrix (will reconstruct matrix
                    from flattened upper triu)
    dir_matrix  :   bool. Whether data is a directed symmetric matrix (will reconstruct
                    matrix from flattened data)
    Returns
    -------
    None
    """

    res_cp = copy.deepcopy(res)
    Y_name = '_'.join(vars)
    output_dir = output_root.joinpath(f'{X_name}_{Y_name}_{its}_its_{subjects_n}_subs_{age_range[0]}_to_{age_range[1]}_age_range')
    output_dir.mkdir(exist_ok=True)

    lvs_n = len(vars)
    if lvs_n == 1:
        res_cp.s = res_cp.s[None]
        res_cp.boot_result.llcorr = res_cp.boot_result.llcorr[None]
        res_cp.boot_result.ulcorr = res_cp.boot_result.ulcorr[None]
        res_cp.boot_result.orig_corr = res_cp.boot_result.orig_corr[None]
        res_cp.boot_result.compare_u[None]

    behav_corrs = []
    ll_corrs = []
    ul_corrs = []
    percent_covs = []
    bsrs = []
    behav_corr_plots = []
    for lv in range(lvs_n):
        percent_covs.append(res_cp.s[lv]**2/sum(res_cp.s**2)*100)
        if len(vars) == 1:
            behav_corrs.append(res_cp.boot_result.orig_corr)
            ll_corrs.append(res_cp.boot_result.llcorr)
            ul_corrs.append(res_cp.boot_result.ulcorr)
        else:
            behav_corrs.append(res_cp.boot_result.orig_corr[:,lv])
            ll_corrs.append(res_cp.boot_result.llcorr[:,lv])
            ul_corrs.append(res_cp.boot_result.ulcorr[:,lv])
        bsrs.append(res_cp.boot_result.compare_u[:,lv])
        print(behav_corrs[lv])
        print(ll_corrs[lv])
        print(ul_corrs[lv])
        print(len(vars))

        behav_corr_df = pd.DataFrame({'behav_corr':behav_corrs[lv],
                                    'll_corr':ll_corrs[lv],
                                    'ul_corr':ul_corrs[lv],
                                    'vars':[v.capitalize() for v in vars]})
        behav_corr_plots.append((
                                ggplot(behav_corr_df,aes(x='vars',y='behav_corr',fill='vars'))
                                + geom_bar(stat="identity", position=position_dodge(), show_legend=False)
                                + geom_errorbar(aes(ymin=ll_corrs[lv], ymax=ul_corrs[lv]))
                                + labs(x='Dependent Variables',y='Behavioural Correlation')
                                + theme_classic()
                                ))

        file_prefix = f'{X_name}_{Y_name}_{its}_its_{subjects_n}_subs_{age_range[0]}_to_{age_range[1]}_age_range'
        file_prefix_lv = f'{file_prefix}_lv{lv+1}'
        behav_corr_plots[lv].save(output_dir.joinpath(f'{file_prefix_lv}_behav_corr.png'))
        if sym_matrix:
            bsr_matrix = graphs.flat_to_square_matrix(bsrs[lv])
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr_matrix.csv'),bsr_matrix,delimiter=',')
            bsr_matrix_flat = graphs.matrix_to_flat_triu(bsr_matrix)
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr.csv'),bsr_matrix_flat)
        elif dir_matrix:
            square_dim = int(math.sqrt(bsrs[lv].shape[0]))
            bsr_matrix = np.reshape(bsrs[lv], (square_dim,square_dim))
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr_matrix.csv'),bsr_matrix)
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr.csv'),bsrs[lv])
        else:
            np.savetxt(output_dir.joinpath(f'{file_prefix_lv}_bsr.csv'),bsrs[lv])
        if sym_matrix or dir_matrix:
            bsr_matrix[np.where(np.abs(bsr_matrix) < 2)] = 0
            bsr_max_abs = np.max(np.abs(bsr_matrix))
            plt.figure()
            heatmap_plot = sns.heatmap(bsr_matrix, cmap='viridis', vmin=-1*bsr_max_abs, vmax=bsr_max_abs)
            fig = heatmap_plot.get_figure()
            fig.savefig(output_dir.joinpath(f'{file_prefix_lv}_bsr_matrix.png'))
        behav_corr_df.to_csv(output_dir.joinpath(f'{file_prefix_lv}_behav_corr.csv'),index=False)
    percent_covs_df = pd.DataFrame({'percent_cov':percent_covs,'lv':[x+1 for x in list(range(lvs_n))]})
    percent_covs_df.to_csv(output_dir.joinpath(f'{file_prefix}_percent_cov.csv'),index=False)
    if len(vars) == 1:
        perm_p = res_cp.perm_result.sprob
    else:
        perm_p = res_cp.perm_result.sprob[:,0]
    perm_p_df = pd.DataFrame({'perm_p':perm_p,'lv':[x+1 for x in list(range(lvs_n))]})
    perm_p_df.to_csv(output_dir.joinpath(f'{file_prefix}_perm_p.csv'),index=False)
    pls.save_pls_model(str(output_dir.joinpath(f'{file_prefix}_model.mat')),res)

    if printing:
        print(Y_name)
        for lv in range(lvs_n):
            if len(vars) > 1:
                print(f'Permutation p:\t\t{perm_p[lv]:.6f}')
                print(f'Percent Covariance:\t{percent_covs[lv][0]:.6f}')
                print(behav_corr_plots[lv])
            else:
                print(f'Permutation p:\t\t{perm_p:.6f}')
                print(f'Percent Covariance:\t{percent_covs[0]:.6f}')
                print(behav_corr_plots)