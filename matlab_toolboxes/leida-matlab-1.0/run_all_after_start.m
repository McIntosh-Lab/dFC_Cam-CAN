%%
%%%%%%% EDIT PATHS BELOW %%%%%%% 
LEiDA_directory = '/path/to/directory/dFC/matlab_toolboxes/leida-matlab-1.0/'; %update for user file system
subjects_file = '/path/to/directory/Cam-CAN/dFC/data/subjects.csv'; %update for user file system
%%%%%%% EDIT PATHS ABOVE %%%%%%%

run_name = 'Cam-CAN_TVB_SchaeferTian_218';
Parcellation = 'TVBSchaeferTian218';
n_permutations = 10000; % can decrease for testing, then increase to 10000
n_bootstraps = 500; % can decrease for testing, then increase to 500





SelectK = 5;
LEiDA_AnalysisK(LEiDA_directory, run_name, SelectK, Parcellation)
LEiDA_AnalysisCentroid(LEiDA_directory, run_name, SelectK, Parcellation)
LEiDA_TransitionsK(LEiDA_directory, run_name, SelectK, n_permutations, n_bootstraps)
LEiDA_StateTime(LEiDA_directory, run_name, SelectK, subjects_file)