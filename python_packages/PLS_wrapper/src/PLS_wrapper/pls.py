import matlab.engine
import numpy as np
import os
import random

class Dict2Object:
    """Takes a dictionary and turns it into a class with an attribute for each key.
    """
    def __init__(self, dictionary=None):
        if dictionary is not None:
            for key, value in dictionary.items():
                setattr(self, key, value)

def perm_result_values_conversion(res_dict_value, convert_to):
    """Decides how to convert inner values from dict valued PLS result `perm_result`
    Parameters
    ----------
    res_dict_value      :   dictionary from `other_input`
    convert_to          :   string, choose whether converting to 'python' or
                            'matlab'
    Return
    ------
    new_dict            :   converted dictionary
    """
    assert convert_to in ['python','matlab']

    matlab_double_arrays = [
        'sp',
        'sprob',
        'permsamp'
    ]
    new_dict = {}
    for key, value in res_dict_value.items():
        if key in matlab_double_arrays:
            if convert_to == 'python':
                new_value = np.array(value)
            elif convert_to == 'matlab':
                new_value = matlab.double(value)
        elif key == 'num_perm':
            if convert_to == 'python':
                new_value = int(value)
            elif convert_to == 'matlab':
                new_value = float(value)
        elif key == 'is_perm_splithalf':
            if convert_to == 'python':
                new_value = bool(int(value))
            elif convert_to == 'matlab':
                new_value = float(value)
        else:
            new_value = value
        new_dict[key] = new_value
    return new_dict

def perm_splithalf_values_conversion(res_dict_value, convert_to):
    """Decides how to convert inner values from dict valued PLS result `perm_splithalf`
    Parameters
    ----------
    res_dict_value      :   dictionary from `other_input`
    convert_to          :   string, choose whether converting to 'python' or
                            'matlab'
    Return
    ------
    new_dict            :   converted dictionary
    """
    assert convert_to in ['python','matlab']

    matlab_double_arrays = [
        'orig_ucorr',
        'orig_vcorr',
        'ucorr_prob',
        'vcorr_prob',
        'ucorr_ll',
        'ucorr_ul',
        'vcorr_ll',
        'vcorr_ul'
    ]
    float_to_int = [
        'num_outer_perm',
        'num_split'
    ]

    new_dict = {}
    for key, value in res_dict_value.items():
        if key in matlab_double_arrays:
            if convert_to == 'python':
                new_value = np.array(value)
            elif convert_to == 'matlab':
                new_value = matlab.double(value)
        elif key in float_to_int:
            if convert_to == 'python':
                new_value = int(value)
            elif convert_to == 'matlab':
                new_value = float(value)
        else:
            new_value = value
        new_dict[key] = new_value
    return new_dict

def boot_result_values_conversion(res_dict_value, convert_to):
    """Decides how to convert inner values from dict valued PLS result `boot_result`
    Parameters
    ----------
    res_dict_value      :   dictionary from `other_input`
    convert_to          :   string, choose whether converting to 'python' or
                            'matlab'
    Return
    ------
    new_dict            :   converted dictionary
    """
    assert convert_to in ['python','matlab']

    float_to_int = [
        'num_boot',
        'countnewtotal',
    ]
    float_to_bool = [
        'nonrotated_boot',
    ]
    ndarray_to_float = [
        'clim',
    ]
    matlab_double_arrays_to_float_numpy = [
        'num_LowVariability_behav_boots',
        'ulcorr',
        'llcorr',
        'ulcorr_adj',
        'llcorr_adj',
        'badbeh',
        'prop',
        'distrib',
        'zero_u_se',
    ]
    matlab_double_arrays_to_int_numpy = [
        'bootsamp_4beh',
        'bootsamp',
    ]
    matlab_single_arrays = [
        'orig_corr',
        'compare_u',
        'u_se',
    ]
    ndarray_to_str = [
        'boot_type',
    ]

    new_dict = {}
    for key, value in res_dict_value.items():
        if key in float_to_int:
            if convert_to == 'python':
                new_value = int(value)
            elif convert_to == 'matlab':
                new_value = float(value)
        elif key in float_to_bool:
            if convert_to == 'python':
                new_value = bool(int(value))
            elif convert_to == 'matlab':
                new_value = float(value)
        elif key in ndarray_to_float:
            new_value = float(value)
        elif (key in matlab_double_arrays_to_float_numpy) or (key in matlab_single_arrays):
            if convert_to == 'python':
                new_value = np.array(value)
            elif (convert_to == 'matlab') and (key in matlab_double_arrays_to_float_numpy):
                new_value = matlab.double(value)
            elif (convert_to == 'matlab') and (key in matlab_single_arrays):
                if value.shape==():
                    new_value = matlab.single(float(value))
                else:
                    new_value = matlab.single(value)
        elif key in matlab_double_arrays_to_int_numpy:
            if convert_to == 'python':
                new_value = np.array(value,dtype=np.int64)
            elif convert_to == 'matlab':
                new_value = np.array(value,dtype=np.float64)
                new_value = matlab.double(new_value)
        elif key in ndarray_to_str:
            new_value = str(value)
        else:
            new_value = value
        new_dict[key] = new_value
    return new_dict

def other_input_values_conversion(res_dict_value, convert_to):
    """Decides how to convert inner values from dict valued PLS result `other_input`
    Parameters
    ----------
    res_dict_value      :   dictionary from `other_input`
    convert_to          :   string, choose whether converting to 'python' or
                            'matlab'
    Return
    ------
    new_dict            :   converted dictionary
    """
    assert convert_to in ['python','matlab']

    new_dict = {}
    for key, value in res_dict_value.items():
        if key == 'meancentering_type':
            if convert_to == 'python':
                new_value = bool(value)
            elif convert_to == 'matlab':
                new_value = float(value)
        elif key == 'cormode':
            if convert_to == 'python':
                new_value = bool(int(value))
            elif convert_to == 'matlab':
                new_value = float(value)
        else:
            new_value = value
        new_dict[key] = new_value
    return new_dict

def PLS_result_conversion(res, convert_to):
    """Take a result from pls_analysis_py.m and convert it to an object with only python native types
    Parameters
    ----------
    res                 :   result from behavioural PLS
    convert_to          :   string, choose whether converting to 'python' or
                            'matlab'
    Return
    ------
    new_dict            :   converted dictionary

    """
    assert convert_to in ['python','matlab']

    matlab_single_arrays = [
        'u',
        'v',
        's',
        'lvcorrs',
        'usc',
        'vsc',
        'stacked_behavdata'
    ]
    float_to_int = [
        'num_conditions'
    ]
    res_new = {}

    if convert_to == 'matlab':
        res = res.__dict__.copy()
    for key, value in res.items():
        if key == 'method':
            if convert_to == 'python':
                new_value = int(value)
            if convert_to == 'matlab':
                new_value = float(value)
        if key == 'is_struct':
            if convert_to == 'python':
                new_value = bool(int(value))
            elif convert_to == 'matlab':
                new_value = float(value)
        elif key == 'datamatcorrs_lst':
            if convert_to == 'python':
                new_value = np.array(value[0])
            elif convert_to == 'matlab':
                new_value = [matlab.single(value)]
        elif key in matlab_single_arrays:
            if convert_to == 'python':
                new_value = np.array(value)
            elif convert_to == 'matlab':
                if value.shape==():
                    new_value = matlab.single(float(value))
                else:
                    new_value = matlab.single(value)
        elif key in float_to_int:
            if convert_to == 'python':
                new_value = int(value)
            elif convert_to == 'matlab':
                new_value = float(value)
        elif key == 'num_subj_lst':
            if convert_to == 'python':
                new_value = np.array(value,dtype=np.int64)
            elif convert_to == 'matlab':
                new_value = np.array(value,dtype=np.float64)
                new_value = matlab.double(new_value)
        elif key in 'perm_result':
            if convert_to == 'matlab':
                value = value.__dict__
            new_value = perm_result_values_conversion(value, convert_to=convert_to)
            if convert_to == 'python':
                new_value = Dict2Object(new_value)
        elif key in 'perm_splithalf':
            if convert_to == 'matlab':
                value = value.__dict__
            new_value = perm_splithalf_values_conversion(value, convert_to=convert_to)
            if convert_to == 'python':
                new_value = Dict2Object(new_value)
        elif key in 'boot_result':
            if convert_to == 'matlab':
                value = value.__dict__
            new_value = boot_result_values_conversion(value, convert_to=convert_to)
            if convert_to == 'python':
                new_value = Dict2Object(new_value)
        elif key in 'other_input':
            if convert_to == 'matlab':
                value = value.__dict__
            new_value = other_input_values_conversion(value, convert_to=convert_to)
            if convert_to == 'python':
                new_value = Dict2Object(new_value)
        else:
            new_value = value
        res_new[key] = new_value
    if convert_to == 'python':
        return Dict2Object(res_new)
    elif convert_to == 'matlab':
        return res_new

def pls_analysis(datamat_lst,num_subj_lst,num_cond,stacked_behavdata,
    num_perm=0,
    num_split=0,
    num_boot=0,
    meancentering_type=0,
    cormode=0,
    boot_type='strat',
    clim=95.0,
    make_script=True,
    seed=None
    ):
    """Python wrapper for matlab implementation of pls_analysis.
    Will use matlab python library to call the original matlab script.
    Warning that this creates (and removes when finished) a matlab script 
    called `pls_analysis_py.m` to call `pls_analysis.m` from the matlab PLS 
    package and removes the `field_descrip` variable (character arrays are 
    problematic, and this variable is not necessary), so make sure there isn't 
    a script called `pls_analysis_py.m` in your working directory that you 
    don't want deleted (not likely but worth mentioning). If you want to 
    avoid the need for this you can copy the `pls_analysis_py.m` script from 
    this repository to your PLS directory or a matlab path directory and set 
    `make_script=False`.

    Parameters
    ----------
    datamat_lst         :   list of 2D ndarrays, one for each group. 
                            Rows represent independent variable data. 
                            If a single 2D ndarray is given will convert to 
                            list automatically.
    num_subj_lst        :   list of integers representing sizes of groups. 
                            If a single integer is given will convert to list 
                            automatically.
    num_cond            :   int representing number of conditions.
    stacked_behavdata   :   2D ndarray with columns representing behavioural 
                            data
    num_perm            :   int, default=0. Number of permutations.
    num_split           :   int, default=0. Number of split half permutations.
    num_boot            :   int, default=0. Number of bootstrap permutations.
    meancentering_type  :   int, default=0. Type of meancentering.
                            0. Remove group condition means from conditon means
                            within each group. Tells us how condition effects
                            are modulated by group membership. (Boost condition
                            differences, remove overall group diffrences).
                            1. Remove grand condition means from each group 
                            condition
                            mean. Tells us how conditions are modulated by 
                            group 
                            membership (Boost group differences, remove overall
                            condition diffrences).
                            2. Remove grand mean over all subjects and 
                            conditions.
                            Tells us full spectrum of condition and group 
                            effects.
                            3. Remove all main effects by subtracting condition
                            and group means. This type of analysis will deal
                            with pure group by condition interaction.
    cormode             :   int, default=0. Correlation type to use.
                            0. Pearson correlation
                            2. covariance
                            4. cosine angle
                            6. dot product
    boot_type           :   string, default = 'strat'. Use 'nonstrat' for 
                            nonstratified boot samples.
    clim                :   float, default=95.0. Confidence level between 0.0 
                            and 100.0.
    make_script         :   bool, default=True. Whether to make and delete the
                            pls_analysis_py.m file in the working directory.
                            If you have copied this file to the PLS directory
                            or a matlab path folder you can set this to False.
    seed                :   int, default=None. Seed to initialize rng random
                            number generator in matlab. If none given, will
                            be set to a random seed with python's 
                            random.randitnt(1,2**32)

    Return
    ------
    res_py              :   Object containing same outputs as original matlab 
                            script, with same names, converted to python native
                            types.
                            Matlab single and double arrays are converted to 
                            numpy ndarrays.
                            Floats used as booleans in matlab (0.0 for False, 
                            1.0 for True) are converted to boolean.
                            Integers or floats being used as integers in matlab
                            converted to int in python.
                            Floats as python floats.
    """
    eng = matlab.engine.start_matlab()
    if seed:
        eng.rng(seed)
    else:
        eng.rng(random.randint(1,2**32))

    # Matlab script for calling pls_analysis.m and removing 'field_descrip'
    if make_script:
        temp_script =   'function result = pls_analysis_py(datamat_lst, num_subj_lst, k, opt)\n' \
                        '    result_tmp = pls_analysis(datamat_lst, num_subj_lst, k, opt);\n' \
                        '    result = rmfield(result_tmp,"field_descrip");\n' \
                        'end'

        with open('pls_analysis_py.m','w+') as f:
            f.write(temp_script)

    if type(datamat_lst) is np.ndarray:
        datamat_lst = [datamat_lst.copy()]

    num_subj_lst = matlab.double(num_subj_lst)

    if stacked_behavdata.ndim == 1:
        stacked_behavdata = stacked_behavdata[:,None]
    stacked_behavdata = matlab.double(stacked_behavdata.copy())

    num_cond = matlab.double(num_cond)

    option = {}
    option['method'] = 3
    option['num_perm'] = matlab.double(num_perm)
    option['num_split'] = matlab.double(num_split)
    option['num_boot'] = matlab.double(num_boot)
    option['stacked_behavdata'] = stacked_behavdata
    option['meancentering_type'] = meancentering_type
    option['cormode'] = cormode
    option['boot_type'] = boot_type
    option['clim'] = matlab.double(clim)

    res = eng.pls_analysis_py(datamat_lst,num_subj_lst,num_cond,option)
    res_py = PLS_result_conversion(res, convert_to='python')

    if make_script:
        os.remove('pls_analysis_py.m')

    return res_py

def load_pls_model(model_file, make_script=True):
    """Load saved behavioural PLS model in matlab format (*.mat).
    Warning that this creates (and removes when finished) a matlab script 
    called `load_pls_model_py.m` to load the saved matlab PLS model, so make 
    sure there isn't a script called `load_pls_model_py.m` in your working 
    directory that you don't want deleted (not likely but worth mentioning). 
    If you want to avoid the need for this you can copy the 
    `load_pls_model_py.m` script from this repository to your PLS directory or a 
    matlab path directory and set `make_script=False`.

    Parameters
    ----------
    model_file          :   str path to *.mat matlab model file
    make_script         :   bool, default=True. Whether to make and delete the
                            load_pls_model_py.m file in the working directory.
                            If you have copied this file to the PLS directory
                            or a matlab path folder you can set this to False.

    Return
    ------
    res_py              :   Object containing same outputs as original matlab 
                            script, with same names, converted to python native
                            types.
                            Matlab single and double arrays are converted to 
                            numpy ndarrays.
                            Floats used as booleans in matlab (0.0 for False, 
                            1.0 for True) are converted to boolean.
                            Integers or floats being used as integers in matlab
                            converted to int in python.
                            Floats as python floats.
    """
    eng = matlab.engine.start_matlab()

    # Matlab script for loading the file, accessing the struct inside, then returning
    # (or first removing the "field_dscrip" field if it exists)
    if make_script:
        temp_script =   'function result = load_pls_model_py(model_file)\n' \
                        '    result_tmp = load(model_file);\n' \
                        '    fields = fieldnames(result_tmp);\n' \
                        '    result_tmp2 = result_tmp.(fields{1});\n' \
                        '    fields2 = fieldnames(result_tmp2);\n' \
                        '    if ismember("field_descrip",fields2) == 1\n' \
                        '       result = rmfield(result_tmp2,"field_descrip");\n' \
                        '    else\n' \
                        '       result = result_tmp2;\n' \
                        '    end\n' \
                        'end'

    with open('load_pls_model_py.m','w+') as f:
        f.write(temp_script)

    res = eng.load_pls_model_py(model_file)
    res_py = PLS_result_conversion(res, convert_to='python')

    if make_script:
        os.remove('load_pls_model_py.m')

    return res_py

def save_pls_model(model_file, res_py):
    """Save behavioural PLS model in matlab format (*.mat).
    Parameters
    ----------
    model_file          :   str path to *.mat matlab model file
    res_py              :   model result from `pls_analysis` function

    Return
    ------
    None
    """
    eng = matlab.engine.start_matlab()

    res = PLS_result_conversion(res_py, convert_to='matlab')
    eng.workspace['res'] = res
    eng.save(model_file,'res',nargout=0)
