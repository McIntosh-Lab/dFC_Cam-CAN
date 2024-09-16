import subprocess
import copy
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import ndimage
from PIL import Image
from nilearn import plotting
from nilearn.image import resample_img
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import importlib.resources

TMP_DIR = Path('/tmp/PyNeudorf')

with importlib.resources.path("PyNeudorf.data","ST218_Labels.xlsx") as label_file:
    LABEL_FILE = label_file
with importlib.resources.path("PyNeudorf.data","MNI152_T1_0.5mm.nii.gz") as atlas_file:
    ATLAS_FILE = atlas_file
with importlib.resources.path("PyNeudorf.data","TVB_SchaeferTian_218_subcort.nii.gz") as subcort_atlas_file:
    SUBCORT_ATLAS_FILE = subcort_atlas_file
with importlib.resources.path("PyNeudorf.data","ggseg_Schaefer200.r") as ggseg_r_file:
    GGSEG_R_FILE = ggseg_r_file

def Schaefer200Cortex(data,output_file,data_thresh,int_fig_thresh=True,labels=LABEL_FILE,tmp_dir=TMP_DIR,rscript='/usr/bin/Rscript'):
    """Use ggseg in R to plot Schaefer 200 parcelation cortex
    Parameters
    ----------
    data            :   1d array. data with same number and order of values as labels
    output_file     :   string. path to output png
    data_thresh     :   float. absolute value at which to threshold data
    int_fig_thresh  :   bool. Default True. whether to round up (ceiling) to integer for color bar
    labels          :   string or pathlib Path. Default to '../../data/ST218_Labels.xlsx' . path to xlsx file with first column called 'region_num',
                        second column called 'region' containing region name, and third
                        column called 'hemi' containing either 'left' or 'right'
    tmp_dir         :   string or pathlib Path. Default to '/tmp/PyNeudorf'. Directory
                        to save data to for use by R.
    Returns
    -------
    None
    """
    tmp_dir.mkdir(exist_ok=True,parents=True)
    data_file = tmp_dir.joinpath('Schaefer200Cortex_data.csv')
    np.savetxt(data_file,data,delimiter=',')

    subprocess.call(f'{rscript} --vanilla {str(GGSEG_R_FILE)} \
                    {labels} \
                    {data_file} \
                    {output_file} \
                    {str(data_thresh)} \
                    {str(int_fig_thresh).upper()}',
                    shell=True)
    
def assign_to_atlas_nifti(data,atlas_file,output_file=None):
    """Saves nifti file after assigning values in data to ST atlas
    Parameters
    ----------
    data        :   1-D ndarray. Data to assign to atlas, with index equal to
                    region number - 1
    atlas_file  :   string. Path to atlas nifti file
    output_file :   string. Path to output nifti file
    regions     :   list. Default=None. Regions to assign data to. If none are
                    given then will assign to all ROIs.
    subcortical :   bool. Default=False. Easy way to assign only to subcortical
                    regions by setting to True (only works for SchaeferTian218).
    Returns
    -------
    output_img  :   Nifti1Image file returned by nibabel.Nifti1Image with atlas regions
                    set to data values
    """
    regions = list(range(len(data)))
    atlas_img = nib.load(atlas_file)
    atlas_affine = atlas_img.affine
    atlas_data = atlas_img.get_fdata()

    output_data = np.zeros_like(atlas_data)

    for r in regions:
        atlas_region_idx = np.where(atlas_data == float(r + 1))
        output_data[atlas_region_idx] = data[r]

    output_img = nib.Nifti1Image(output_data, atlas_affine)
    if output_file:
        nib.save(output_img, output_file)
    return output_img

def threshold_data(data,thresh):
    """Threshold data based on thresh in positive and negative directions
    """
    data_thresh = copy.deepcopy(data)
    data_thresh[np.where(np.abs(data_thresh) < thresh)] = 0.0
    return data_thresh

def SchaeferTian218_subcortical(data,output_file,thresh):
    """Use nilearn to plot axial and sagittal views of subcortical regions
    Parameters
    ----------
    data            :   1d array. data with same number and order of values as labels
    output_file     :   string. path to output png
    thresh          :   float. absolute value at which to threshold data
    Returns
    -------
    None
    """
    output_file = Path(output_file)

    def split_data_neg_pos_abs(data):
        """Split data into negative and positive subsets and assign absolute value to
        negative subset
        """
        data_neg = copy.deepcopy(data)
        data_pos = copy.deepcopy(data)
        data_neg[np.where(data > 0.0)] = 0.0
        data_neg *= -1.0
        data_pos[np.where(data < 0.0)] = 0.0
        return data_neg, data_pos

    data_neg, data_pos = split_data_neg_pos_abs(data)
    data_neg_img = assign_to_atlas_nifti(data_neg,SUBCORT_ATLAS_FILE)
    data_neg_max = np.max(data_neg)
    print('all_neg_max',data_neg_max)
    data_pos_img = assign_to_atlas_nifti(data_pos,SUBCORT_ATLAS_FILE)
    data_pos_max = np.max(data_pos)
    print('all_pos_max',data_pos_max)
    data_max = np.max([data_neg_max,data_pos_max])

    # Nilearn plotting method
    BG_img = nib.load(ATLAS_FILE)
    subcort_overlay = nib.load(SUBCORT_ATLAS_FILE)

    greyscale_cmap = LinearSegmentedColormap.from_list('greyscale', [(0,0,0),(1,1,1)],N=1000)
    black_cmap = LinearSegmentedColormap.from_list('greyscale', [(0,0,0),(0,0,0)],N=1000)
    yellow_red_cmap = LinearSegmentedColormap.from_list('greyscale', [(.906,.761,.106),(.941,.094,0)],N=1000)
    green_blue_cmap = LinearSegmentedColormap.from_list('greyscale', [(.698,.757,.463),(.224,.604,.694)],N=1000)

    def make_subcort_images(BG_img,subcort_overlay,neg_img,pos_img,img_thresh,vmax_neg,vmax_pos,zoom_scale):

        def zoom(img_file,zoom_scale):
            tmp_img = Path('/tmp/figure.png')
            img_file.savefig(tmp_img)
            img = Image.open(tmp_img)
            img_np = np.array(img.getdata()).reshape(img.size[1],img.size[0], 4)
            img_np_zoomed = ndimage.zoom(img_np, (zoom_scale,zoom_scale,1), order=0)
            return img_np_zoomed


        def subcort_plotting(display_mode,cut_coords):
            subcort_img = plotting.plot_img(    BG_img, 
                                                display_mode=display_mode,
                                                cut_coords=cut_coords,
                                                draw_cross=False,
                                                cmap=greyscale_cmap,
                                                figure=1,
                                                annotate=False,
                                                )

            subcort_img.add_overlay(            subcort_overlay,
                                                cmap=black_cmap,
                                                threshold=1.0,
                                                )
            if vmax_pos > 0:
                subcort_img.add_overlay(        pos_img,
                                                vmin=img_thresh,
                                                vmax=vmax_pos,
                                                cmap=yellow_red_cmap,
                                                threshold=img_thresh,                 
                                                )
            if vmax_neg > 0:
                subcort_img.add_overlay(        neg_img, 
                                                vmin=img_thresh,
                                                vmax=vmax_neg,
                                                cmap=green_blue_cmap,
                                                threshold=img_thresh,                 
                                                )
            return subcort_img
        
        LH_sag_coord = -28.0
        RH_sag_coord = 30.0
        axial_coord = 0.0

        subcort_img_LH = subcort_plotting('x',[LH_sag_coord])

        xxadj = -15
        xyadj = -9
        xbase = 35
        subcort_img_LH.axes[LH_sag_coord].ax.set_xlim(-xbase+xxadj,xbase+xxadj)
        subcort_img_LH.axes[LH_sag_coord].ax.set_ylim(-xbase+xyadj,xbase+xyadj)
        img_np_zoomed_LH = zoom(subcort_img_LH,zoom_scale)[:,:,:-1]

        subcort_img_RH = subcort_plotting('x',[RH_sag_coord])

        xxadj = -15
        xyadj = -9
        xbase = 35
        subcort_img_RH.axes[RH_sag_coord].ax.set_xlim(-xbase+xxadj,xbase+xxadj)
        subcort_img_RH.axes[RH_sag_coord].ax.set_ylim(-xbase+xyadj,xbase+xyadj)
        img_np_zoomed_RH = zoom(subcort_img_RH,zoom_scale)[:,:,:-1]
        
        subcort_img_axial = subcort_plotting('z',[axial_coord])

        zxadj = 1
        zyadj = -10
        zbase = 50
        subcort_img_axial.axes[axial_coord].ax.set_xlim(-zbase+zxadj,zbase+zxadj)
        subcort_img_axial.axes[axial_coord].ax.set_ylim(-zbase+zyadj,zbase+zyadj)
        img_np_zoomed_axial = zoom(subcort_img_axial,zoom_scale)[:,:,:-1]
        
        return img_np_zoomed_LH, img_np_zoomed_RH, img_np_zoomed_axial

    img_LH, img_RH, img_axial = make_subcort_images(BG_img, subcort_overlay, data_neg_img, data_pos_img,thresh,data_max,data_max,3)

    # Combined image for figures
    def combine_figures(img_LH,img_RH,img_axial):
        x,y,z = img_LH.shape
        img_new = np.empty((x,y*3,z))
        img_new[:,:y,:] = img_LH
        img_new[:,y:2*y,:] = img_axial
        img_new[:,2*y:,:] = np.flip(img_RH,axis=1)
        return img_new

    img_combined = combine_figures(img_LH,img_RH,img_axial)
    Image.fromarray(img_combined.astype('uint8'),'RGB').save(output_file)

    def save_colorbar(cmap,cmap_file):
        plt.figure()
        a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
        plt.imshow(a, cmap=cmap, origin='lower')
        plt.axis("off")
        plt.savefig(cmap_file)

    cmap_file_pos = output_file.parent.joinpath(f'{output_file.stem}_pos_colorbar.png')
    cmap_file_neg = output_file.parent.joinpath(f'{output_file.stem}_neg_colorbar.png')
    save_colorbar(yellow_red_cmap,cmap_file_pos)
    save_colorbar(green_blue_cmap,cmap_file_neg)
