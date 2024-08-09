function result = pls_analysis_py(datamat_lst, num_subj_lst, k, opt)
    result_tmp = pls_analysis(datamat_lst, num_subj_lst, k, opt);
    result = rmfield(result_tmp,'field_descrip');
end
