%% PLS group analyses of OA and YA in median split groups of VWM
file_prefixes = [   "state_transition_5_to_3_node_energy_dict_continous";
                    "fractional_occupancy_K5"
                    ];

file_prefixes_size = size(file_prefixes);
file_prefixes_n = file_prefixes_size(1);

X_file_strings = ["_YA_low_VWM_47_subs_X";"_YA_high_VWM_67_subs_X";"_OA_low_VWM_52_subs_X";"_OA_high_VWM_31_subs_X"];
X_file_strings_size = size(X_file_strings);
X_file_strings_n = X_file_strings_size(1);
lvs = X_file_strings_n;

results = [];
for i = 1:file_prefixes_n
    disp(file_prefixes(i))
    datamat_lst = [];

    num_subj = [];
    for j = 1:X_file_strings_n
        datamat_lst{j} = readmatrix(strcat(append('inputs/',file_prefixes(i),X_file_strings{j},".csv")));
        datamat_lst_size = size(datamat_lst{j});
        num_subj(1,j) = datamat_lst_size(1);
    end

    %num_subj=size(datamat_lst);

    num_cond=1;
    option.method=1;
    option.num_boot=1000;
    option.num_perm=1000;
    option.meancentering_type=1;

    results{i} = pls_analysis(datamat_lst,num_subj,num_cond,option);

end
%%

for i = 1:file_prefixes_n
    distrib = results{i}.boot_result.distrib;
    usc_se = std(distrib,0,3);
    orig_usc = results{1,i}.boot_result.orig_usc
    results{1,i}.boot_result.usc_se = usc_se;
    results{1,i}.boot_result.usc_se_ll = orig_usc - (1.96*usc_se);
    results{1,i}.boot_result.usc_se_ul = orig_usc + (1.96*usc_se);
end

%%
LV1_bsr = results{1,1}.boot_result.compare_u(:,1);
LV2_bsr = results{1,1}.boot_result.compare_u(:,2);
writematrix(LV1_bsr, 'NCT_meancentered_LV1_bsr.csv')
writematrix(LV2_bsr, 'NCT_meancentered_LV2_bsr.csv')