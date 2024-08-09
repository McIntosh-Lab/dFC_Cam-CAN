# %%
import numpy as np
import os
import shutil

DATA_DIR = 'data/SC/TVBSchaeferTian220_clean'
SUBJECTS_FILE = 'data/all_participants_ids.txt'
OUTPUT_DIR = 'data/SC/SC_matrices_consistency_thresholded_0.5'
REGIONS_N = 220

# %%
def consistency_thresholding(DATA_DIR, OUT_DIR, threshold, subject_list, PARC_NAME, ROI_remove):
    """Script to consistency threshold a group of processed subjects' structural 
     connectivity matrices.
    Arguments
    ----------
    DATA_DIR : 
        path to directory containing weights files for all 
        subjects to be consistency thresholded
    OUT_DIR :
        output path to save thresholded weight.txt and tract_lengths.txt files
        for various atlas parcellations
    threshold : 
        float representing rate at which each structural connection  
        should appear (have a non-zero probability of connection) in the 
        group of subjects. (e.g. threshold=0.4 will eliminate any 
        connections, from all subjects, that dont appear in at least 40% of 
        subjects) 
    subject_list : 
        path to text file containing subject names, one per line, to be
        thresholded
    PARC_NAME :
        name of parcellation. use '' if these zip files were created 
        with an earlier version of the pipeline that did not specify 
        parcellations in tvb_input.zip filenames
    ROI_remove :
        list of ROIs to remove before checking for nan values
    """


    #make output dir
    output_dir=os.path.join(OUT_DIR,PARC_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #import subject names into an array
    subjects = []
    with open(subject_list) as subject_list_file:
        for line in subject_list_file:
            line = line.rstrip('\n')
            subjects.append(line)

    consistency_mask=''
    subcounter=0

    #unzip each subject into output dir
    for subject in subjects:
        print(subject)
        #path to SC
        #changing this for updated file structure
        SC_path=os.path.join(DATA_DIR,subject,'structural_inputs','weights.txt')
        print(SC_path)
        SC=''
        if os.path.exists(SC_path):
            #load SC
            SC=np.loadtxt(SC_path)[:REGIONS_N,:REGIONS_N]
            print(SC.shape)
            
            #remove ROI_remove regions first
            if len(ROI_remove) > 0:
                SC = np.delete(SC, ROI_remove, axis=0)
                SC = np.delete(SC, ROI_remove, axis=1)
            
            if np.any(np.isnan(SC)):
                print('found nan in',SC_path)
            else:
                #binarize SC and add to consistency mask to track how many subs have a connnection for each connection 
                SC=np.where(SC>0, 1, 0)
                if consistency_mask=='':
                    consistency_mask=np.copy(SC)
                else:
                    consistency_mask=consistency_mask+SC
                subcounter = subcounter+1

    #quit if weve encountered no nan-less SC matrices
    print(consistency_mask)
    if consistency_mask == '':
        quit()

    #binarize consistency mask, thresholded by (#subs * threshold %)
    min_sub_count = float(threshold)*subcounter
    if subject == 723395:
        print(consistency_mask, min_sub_count)
    consistency_mask = consistency_mask - min_sub_count
    consistency_mask=np.where(consistency_mask>=0, 1, 0)

    #go into each subject and edit their weights.txt
    for subject in subjects:
        SC_path=os.path.join(DATA_DIR,subject,'structural_inputs','weights.txt')
        SC_thresh_dir=os.path.join(output_dir,subject,'structural_inputs')
        if not os.path.exists(SC_thresh_dir):
            os.makedirs(SC_thresh_dir)
        SC_thresh_path=os.path.join(SC_thresh_dir,'weights.txt')
        TL_path=os.path.join(DATA_DIR,subject,'structural_inputs','tract_lengths.txt')
        TL_thresh_path=os.path.join(SC_thresh_dir,'tract_lengths.txt')

        #load SCs again and apply consistency mask
        SC=''
        if os.path.exists(SC_path):
            SC=np.loadtxt(SC_path)[:REGIONS_N,:REGIONS_N]
            
            #remove ROI_remove regions first
            if len(ROI_remove) > 0:
                SC = np.delete(SC, ROI_remove, axis=0)
                SC = np.delete(SC, ROI_remove, axis=1)
            
            sc_has_nans=np.any(np.isnan(SC))
            if not sc_has_nans:
                #threshold SC and save
                SC=consistency_mask*SC
                np.savetxt(SC_thresh_path, SC)

                #load TL, threshold, and save
                if os.path.exists(TL_path):
                    TL=np.loadtxt(TL_path)[:REGIONS_N,:REGIONS_N]

                    #remove ROI_remove regions first
                    if len(ROI_remove) > 0:
                        TL = np.delete(TL, ROI_remove, axis=0)
                        TL = np.delete(TL, ROI_remove, axis=1)

                    TL=consistency_mask*TL
                    np.savetxt(TL_thresh_path, TL)
                #load rest of struct tvb_input files to have ROIs removed and saved
                cent_path=os.path.join(DATA_DIR,subject,'structural_inputs','centres.txt')
                cort_path=os.path.join(DATA_DIR,subject,'structural_inputs','cortical.txt')
                hemi_path=os.path.join(DATA_DIR,subject,'structural_inputs','hemisphere.txt')
                cent_path_out=os.path.join(SC_thresh_dir,'centres.txt')
                cort_path_out=os.path.join(SC_thresh_dir,'cortical.txt')
                hemi_path_out=os.path.join(SC_thresh_dir,'hemisphere.txt')

                single_row_ROI_remove=[cent_path,cort_path,hemi_path]
                single_row_ROI_remove_out=[cent_path_out,cort_path_out,hemi_path_out]

                for info_file in list(zip(single_row_ROI_remove,single_row_ROI_remove_out)):
                    if os.path.exists(info_file[0]):
                        loaded_input = np.genfromtxt(info_file[0],delimiter='\n', dtype='str')
                        loaded_input = np.delete(loaded_input, ROI_remove)
                        np.savetxt(info_file[1],loaded_input, delimiter='\n', fmt='%s')

            else:
                print('sub has nans - thresh directory deleted',SC_thresh_dir)
                shutil.rmtree(SC_thresh_dir)
        else:
            print('SC not found - thresh directory deleted',SC_thresh_dir)
            if os.path.exists(SC_thresh_dir):
                shutil.rmtree(SC_thresh_dir)

#%%
print('==================')
print('TVBSchaeferTian220')
print('==================')
atlas_220_gp = [104,214]   # Globus pallidus regions to exclude
consistency_thresholding(DATA_DIR,OUTPUT_DIR,0.5,SUBJECTS_FILE,'TVBSchaeferTian220',atlas_220_gp)