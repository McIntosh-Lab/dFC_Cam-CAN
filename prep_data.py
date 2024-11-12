#%%
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr
from scipy.io import savemat
import matplotlib.pyplot as plt
import pickle
from scipy import signal

ATLAS = '220'
TR = 1.97
BEHAVIOURAL_DATA_DIR = Path('data/behav')
BEHAVIOURAL_DATA_FILE = BEHAVIOURAL_DATA_DIR.joinpath('behavioural_data.csv')
SC_DATA_DIR = Path('data/SC/SC_matrices_consistency_thresholded_0.5')
SC_DICT_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dict.pkl')
SC_DIST_FILE = SC_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','SC_dist_dict.pkl')
RSFMRI_DATA_DIR = Path('data/rsfMRI')
RSFMRI_DICT_FILE = RSFMRI_DATA_DIR.joinpath(f'TVBSchaeferTian{ATLAS}','fmri_timeseries_dict.pkl')
INPUTS_DIR = Path('data')
LEIDA_DATA_DIR = Path('LEiDA_inputs')
LEIDA_DATA_DIR.mkdir(exist_ok=True)

#%%
with open(SC_DICT_FILE, 'rb') as f:
    SC_dict = pickle.load(f)

SC_subjects = list(SC_dict.keys())

#%%
filter_vars = ['age','CattellTotal','Prcsn_PerceptionTest']
behaviour_ages = pd.read_csv(BEHAVIOURAL_DATA_FILE,sep=',')
subjects_Y = behaviour_ages[['subject']+filter_vars].dropna()
subjects = subjects_Y.subject.tolist()
# Also filtering by those with SC data so that we can do analyses with SC as well
subjects = [s for s in subjects if str(s) in SC_subjects]
print(len(subjects))

#%% Get rsfMRI data
with open(RSFMRI_DICT_FILE, 'rb') as f:
    rsfMRI_dict = pickle.load(f)

#%% Manipulate to time*region matrix with time on y-axis/rows (for HMM-MAR Matlab)
rsfMRI_list = [v.T for k,v in rsfMRI_dict.items() if int(k) in subjects]
print(len(rsfMRI_list))

def bandpass_filter_rois(rois_timeseries, samp_interval, cutoff_high, cutoff_low, axis=1):
    samp_freq = 1 / samp_interval
    w = [cutoff_low, cutoff_high]
    sos = signal.butter(5,w,'bandpass',fs=samp_freq, output='sos')
    output = signal.sosfiltfilt(sos, rois_timeseries, axis)
    return output

print(rsfMRI_list[0].shape)

rsfMRI_filtered_list = [bandpass_filter_rois(ts, TR, .1, .01, axis=0) for ts in rsfMRI_list]

timeseries = rsfMRI_list[0][:,0]
output = rsfMRI_filtered_list[0][:,0]
plt.figure(0)
plt.plot(timeseries - np.mean(timeseries), label='unfiltered')
plt.plot(output, label='filtered')

rsfMRI_raw_means = [np.mean(ts,axis=0) for ts in rsfMRI_list]
rsfMRI_filtered_raw_units_list = [ts + rsfMRI_raw_means[i] for i,ts in enumerate(rsfMRI_filtered_list)]
plt.figure(1)
plt.plot(rsfMRI_list[0][:,0], label='unfiltered')
plt.plot(rsfMRI_filtered_raw_units_list[0][:,0], label='filtered')
rsfMRI_filtered_zscore_list = [(ts - np.mean(ts,axis=0)) / np.std(ts,axis=0) for ts in rsfMRI_filtered_list]

#%% Frequency power spectrum
def power_spectrum(ts, sample_interval):
    ps = np.abs(np.fft.rfft(ts))**2
    freqs = np.fft.rfftfreq(ts.size, sample_interval)
    idx = np.argsort(freqs)

    plt.plot(freqs[idx], ps[idx])

power_spectrum(timeseries - np.mean(timeseries), TR)
power_spectrum(output, TR)

#%%Save subject order
rsfMRI_subs = np.array([s for s in subjects if str(s) in list(rsfMRI_dict.keys())],dtype=int)
np.savetxt(INPUTS_DIR.joinpath('subjects.csv'),rsfMRI_subs, fmt="%d")

#%% Demographics
rsfMRI_subs_pd = pd.DataFrame({'subject':rsfMRI_subs.tolist()})
behaviour_ages_rsfMRI = pd.merge(behaviour_ages,rsfMRI_subs_pd,on='subject',how='right')
behavoiur_ages_rsfMRI_YA = behaviour_ages_rsfMRI.loc[behaviour_ages_rsfMRI.age < 50]
rsfMRI_YA_min_age = behavoiur_ages_rsfMRI_YA.age.min()
rsfMRI_YA_max_age = behavoiur_ages_rsfMRI_YA.age.max()
rsfMRI_YA_mean_age = behavoiur_ages_rsfMRI_YA.age.mean()
rsfMRI_YA_std_age = behavoiur_ages_rsfMRI_YA.age.std()
rsfMRI_YA_N = behavoiur_ages_rsfMRI_YA.shape[0]
rsfMRI_YA_female_N = behavoiur_ages_rsfMRI_YA.loc[behavoiur_ages_rsfMRI_YA.sex=='FEMALE'].shape[0]
rsfMRI_YA_male_N = behavoiur_ages_rsfMRI_YA.loc[behavoiur_ages_rsfMRI_YA.sex=='MALE'].shape[0]

behavoiur_ages_rsfMRI_OA = behaviour_ages_rsfMRI.loc[behaviour_ages_rsfMRI.age >= 50]
rsfMRI_OA_min_age = behavoiur_ages_rsfMRI_OA.age.min()
rsfMRI_OA_max_age = behavoiur_ages_rsfMRI_OA.age.max()
rsfMRI_OA_mean_age = behavoiur_ages_rsfMRI_OA.age.mean()
rsfMRI_OA_std_age = behavoiur_ages_rsfMRI_OA.age.std()
rsfMRI_OA_N = behavoiur_ages_rsfMRI_OA.shape[0]
rsfMRI_OA_female_N = behavoiur_ages_rsfMRI_OA.loc[behavoiur_ages_rsfMRI_OA.sex=='FEMALE'].shape[0]
rsfMRI_OA_male_N = behavoiur_ages_rsfMRI_OA.loc[behavoiur_ages_rsfMRI_OA.sex=='MALE'].shape[0]

print(f'Total age mean: {behaviour_ages_rsfMRI.age.mean()} and SD: {behaviour_ages_rsfMRI.age.std()}')
print(f'Younger adult ages ranged from {rsfMRI_YA_min_age} to {rsfMRI_YA_max_age} (mean = {rsfMRI_YA_mean_age}, SD = {rsfMRI_YA_std_age}, N = {rsfMRI_YA_N}, {rsfMRI_YA_female_N} female, {rsfMRI_YA_male_N} male)')
print(f'Older adult ages ranged from {rsfMRI_OA_min_age} to {rsfMRI_OA_max_age} (mean = {rsfMRI_OA_mean_age}, SD = {rsfMRI_OA_std_age}, N = {rsfMRI_OA_N}, {rsfMRI_OA_female_N} female, {rsfMRI_OA_male_N} male)')

#%%Create dicts for help in saving
rsfMRI_LEiDA_dict = {'sub-'+str(rsfMRI_subs[i]):v.T for i,v in enumerate(rsfMRI_filtered_zscore_list)}
rsfMRI_not_zscored_raw_units_dict = {'sub-'+str(rsfMRI_subs[i]):v.T for i,v in enumerate(rsfMRI_filtered_raw_units_list)}

#%%
metadata_df = behaviour_ages[behaviour_ages['subject'].isin(rsfMRI_subs)][['subject','age']]
metadata_df['subject_id'] = 'sub-' + metadata_df['subject'].astype(str)
metadata_df.loc[metadata_df['age']<50,['condition']] = 'young'
metadata_df.loc[metadata_df['age']>=50,['condition']] = 'old'

metadata_df = metadata_df[['subject_id','condition']]

metadata_df.to_csv(LEIDA_DATA_DIR.joinpath('metadata.csv'),sep=',',index=False)

#%% Save data for leida-matlab
LEIDA_DATA_DIR.joinpath('matlab/').mkdir(exist_ok=True)
for k,v in rsfMRI_LEiDA_dict.items():
    condition = metadata_df.loc[metadata_df['subject_id'] == k,['condition']].iat[0,0]
    save_name = f'{k}_{condition}.txt'
    np.savetxt(LEIDA_DATA_DIR.joinpath(f'matlab/{save_name}'),v,delimiter='\t')