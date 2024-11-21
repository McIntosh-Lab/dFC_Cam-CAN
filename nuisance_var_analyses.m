%% nuisance variables analysis
%edit below ------------------------------
project_dir = '/PATH/TO/PROJECT/'; %edit
%edit above ------------------------------
pls_dir = [project_dir 'outputs/leida-matlab/PLS/'];
motion_res_file = [pls_dir 'TP_dict_K5_mean_rel_disp_1000_its_197_subs_0_to_150_age_range/TP_dict_K5_mean_rel_disp_1000_its_197_subs_0_to_150_age_range_model.mat'];
motion_res = load(motion_res_file,'res');

res_file = [pls_dir 'TP_dict_K5_age_1000_its_197_subs_0_to_150_age_range/TP_dict_K5_age_1000_its_197_subs_0_to_150_age_range_model.mat'];
res = load(res_file,'res');

x = load([pls_dir 'TP_dict_K5_0_to_150_age_range.mat'],'X');

nperm = 1000;
n_con = 1;
lv_orig = 1;
lv_N = 1;

%% 
x = x.X;
y = motion_res.res.stacked_behavdata(:,1);

orig_result = res.res;
nuisance_result = motion_res.res;

%% Step 3: Compute cosines between original bPLS and secondary bPLS
n=size(y,1);
n=n/n_con;
dot_distribution = zeros(1, nperm); 

for i=1:nperm
    yperm=y(randperm(n*n_con),:);
    if n_con>1
        idx_subj=[1:n*n_con];
        idx_subj=reshape(idx_subj,n,n_con);
        rxPy = [];
    
        for j=1:n_con
            tmp_rxPy = corr(x(idx_subj(:,j),:),yperm(idx_subj(:,j),:));
            rxPy=[rxPy,tmp_rxPy];
        end
    
    else
        rxPy=corr(x,yperm);
    end

    locate_nans = find(isnan(rxPy));
    rxPy(locate_nans)= 0;

    [perm_u, ~, ~] = svd(rxPy, 'econ');
     cosine_val = perm_u(:,lv_N)'* orig_result.u(:,lv_orig);
     dot_distribution(i) = cosine_val;
end

% p-value 
orig_cosine = nuisance_result.u(:,lv_N)'*orig_result.u(:,lv_orig);
p_value = sum(abs(dot_distribution) >= abs(orig_cosine)) / nperm;

disp(['Original Cosine: ', num2str(orig_cosine)]);
disp(['P-value: ', num2str(p_value)]);