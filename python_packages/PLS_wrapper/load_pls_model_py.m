function result = load_pls_model_py(model_file)
    result_tmp = load(model_file);
    fields = fieldnames(result_tmp);
    result_tmp2 = result_tmp.(fields{1});
    fields2 = fieldnames(result_tmp2);
    if ismember("field_descrip",fields2) == 1
        result = rmfield(result_tmp2,"field_descrip");
    else
        result = result_tmp2;
    end
end
