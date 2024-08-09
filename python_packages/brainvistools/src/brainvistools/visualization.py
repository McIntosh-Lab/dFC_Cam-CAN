import subprocess
import copy
import platform
import tempfile
import nibabel as nib
import numpy as np
import importlib.resources
from pathlib import Path
from scipy import ndimage
from PIL import Image
from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

# Tested on linux. Should also work on Mac and Windows because of the following lines.
TMP_DIR = Path(tempfile.gettempdir())
if platform.system() in ['Linux','Darwin']:
    RSCRIPT = Path('Rscript')
elif platform.system() in ['Windows']:
    RDIR = Path("c:\Program Files\R")
    RVERSION_DIRS = sorted([d for d in RDIR.glob('**\*') if d.is_dir()], reverse=True)
    RSCRIPT = RVERSION_DIRS[0].joinpath('bin/Rscript.exe')

with importlib.resources.path("brainvistools.data","MNI152_T1_0.5mm.nii.gz") as standard_file:
    STANDARD_FILE = standard_file
with importlib.resources.path("brainvistools.data","ggseg_figure.r") as ggseg_r_file:
    GGSEG_R_FILE = ggseg_r_file
with importlib.resources.path("brainvistools.data","TVB_SchaeferTian_fixed_218.nii.gz") as atlas_file:
    REGIONS_218_ATLAS_FILE = atlas_file
with importlib.resources.path("brainvistools.data","TVB_SchaeferTian_fixed_220.nii.gz") as atlas_file:
    REGIONS_220_ATLAS_FILE = atlas_file
with importlib.resources.path("brainvistools.data","TVB_SchaeferTian_fixed_418.nii.gz") as atlas_file:
    REGIONS_418_ATLAS_FILE = atlas_file
with importlib.resources.path("brainvistools.data","TVB_SchaeferTian_fixed_420.nii.gz") as atlas_file:
    REGIONS_420_ATLAS_FILE = atlas_file
with importlib.resources.path("brainvistools.data","ST218_Labels.csv") as label_file:
    LABEL_218_FILE = label_file
with importlib.resources.path("brainvistools.data","ST220_Labels.csv") as label_file:
    LABEL_220_FILE = label_file
with importlib.resources.path("brainvistools.data","ST418_Labels.csv") as label_file:
    LABEL_418_FILE = label_file
with importlib.resources.path("brainvistools.data","ST420_Labels.csv") as label_file:
    LABEL_420_FILE = label_file

ATLAS_DICT = {
    'ST218': REGIONS_218_ATLAS_FILE,
    'ST220': REGIONS_220_ATLAS_FILE,
    'ST418': REGIONS_418_ATLAS_FILE,
    'ST420': REGIONS_420_ATLAS_FILE,
}

SUBCORT_REGIONS_DICT = {
    'ST218': [list(range(101,110))+list(range(210,219))],
    'ST220': [list(range(101,111))+list(range(211,221))],
    'ST418': [list(range(201,210))+list(range(410,419))],
    'ST420': [list(range(201,211))+list(range(411,421))],
}

ATLAS_LABEL_DICT = {
    'ST218': LABEL_218_FILE,
    'ST220': LABEL_220_FILE,
    'ST418': LABEL_418_FILE,
    'ST420': LABEL_420_FILE,
}

ATLAS_GGSEG_DICT = {
    'ST218': 'schaefer17_200',
    'ST220': 'schaefer17_200',
    'ST418': 'schaefer17_400',
    'ST418': 'schaefer17_400',
}

def vis_cortex(data,output_file,thresh,atlas_name='ST218',atlas=None,labels=None,int_fig_thresh=False,neg_colour=False,pos_colour=False,bg_colour='white',tmp_dir=TMP_DIR,rscript=RSCRIPT):
    """Use ggseg in R to plot TVBSchaeferTian218 parcelation cortex.
    There will be R warnings about missing/non-matching data if there are extra labels
    not matching the cortex atlas (e.g., when using TVBSchaeferTian atlas with the
    Tian subcortical regions included). This is fine, as it will ignore these and
    just plot the cortical regions.
    R library requirements: ggseg, ggsegSchaefer, tidyverse, ggplot2, dplyr
    See here https://github.com/ggseg/ggsegSchaefer for ggsegSchaefer install instructions
    Parameters
    ----------
    data            :   1D array. Data with same number and order of values as labels
    output_file     :   string. Path to output png
    thresh          :   float. Absolute value at which to threshold data
    atlas_name      :   str. Default='ST218'. Key to use which will access the data
                        bundled in this package. 'ST218' and 'ST220' will use the
                        Schaefer 200 atlas while 'ST418' and 'ST420' will use the
                        Schaefer 400 atlas. Use the atlas and labels parameters
                        below to use your own atlas selection and corresponding
                        labels
    atlas           :   str. Default='schaefer17_200'. Atlas setting to be assigned
                        for ggsegSchaefer. Options are 'schaefer17_100', 'schaefer7_100',
                        'schaefer17_200', 'schaefer7_200', 'schaefer17_300', 'schaefer7_300',
                        'schaefer17_400', 'schaefer7_400'. More can be added by
                        editting 'data/ggseg_figure.r' file
    labels          :   str or pathlib Path. Default='data/ST218_Labels.csv'.
                        Path to csv file with first column called 'region' containing 
                        region name, and second column called 'hemi' containing 
                        either 'left' or 'right'
    int_fig_thresh  :   bool. Default=False. whether to round to integers for color bar
    neg_colour	   :   str. hex representation of color to apply for any negative values
    pos_colour      :   str. hex representation of color to apply for any positive values
    bg_colour	   :   str. background colour to use in r. default 'white'
    tmp_dir         :   string or pathlib Path. Default to default system tmp dir.
                        Directory to save data to for use by R
    rscript         :   str or pathlib Path. Default set to 'Rscript' for Linux
                        and OSX and "C:\Program Files\R\R-#.#.#\bin\Rscript.exe"
                        for Windows.
                        Set to your specific location if default does not work
    Returns
    -------
    None
    """
    data_file = tmp_dir.joinpath('vis_cortex_data.csv')
    np.savetxt(data_file,data,delimiter=',')

    if not labels:
        labels = ATLAS_LABEL_DICT[atlas_name]
    if not atlas:
        atlas = ATLAS_GGSEG_DICT[atlas_name]

    if neg_colour:
    	neg_colour = '\#' + neg_colour
    if pos_colour:
    	pos_colour = '\#' + pos_colour

    subprocess.call(f'{str(rscript)} --vanilla {str(GGSEG_R_FILE)} \
                    {data_file} \
                    {output_file} \
                    {str(thresh)} \
                    {atlas} \
                    {labels} \
                    {str(int_fig_thresh).upper()} \
                    {str(neg_colour).upper()} \
                    {str(pos_colour).upper()} \
                    {bg_colour}', 
                    shell=True)
    
def assign_to_atlas_img(data,atlas_img,output_file=None,regions=None):
    """Creates nibabel image file after assigning values in data to atlas
    Parameters
    ----------
    data        :   1D ndarray. Data to assign to atlas, with index equal to
                    region number - 1
    atlas_img   :   nibabel Nifti1Image image. Atlas image to assign data to
    output_file :   str or pathlib Path. Default=None. Path to output nifti file
    regions     :   list. Default=None. Regions to assign data to. If none are
                    given then will assign to all ROIs (assuming ROIs are labelled
                    1 to len(data) in atlas_file nifti)
    Returns
    -------
    output_img  :   Nifti1Image file returned by nibabel.Nifti1Image with atlas regions
                    set to data values
    """
    if not regions:
        regions = list(range(len(data)))
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
    Parameters
    ----------
    data            :   1D array. Data with same number and order of values as labels
    thresh          :   float. Threshold value
    Returns
    -------
    data_thresh     :   1D array. Thresholded data (thresholded values set to 0.0)
    """
    data_thresh = copy.deepcopy(data)
    data_thresh[np.where(np.abs(data_thresh) < thresh)] = 0.0
    return data_thresh

def vis_subcortical(data,output_file,thresh,atlas_name='ST218',atlas_file=None,subcort_regions=None,tmp_dir=TMP_DIR, save_cmap=False, pos_colours=None, neg_colours=None):
    """Use nilearn to plot axial and sagittal views of subcortical regions and save
    as individual images as well as a combined image of these views.
    Parameters
    ----------
    data            :   1D array. Data where index+1 data corresponds to subcortical
                        regions in atlas_file nifti
    output_file     :   plathlib Path or string. Path to output png
    thresh          :   float. Absolute value at which to threshold data
    atlas_name      :   str. Default='ST218'. Can be 'ST218', 'ST220', 'ST418', 
                        or 'ST420'
    atlas_file      :   pathlib Path or str. Default=None. If value is given this
                        will override the atlas_name option and use your own defined 
                        atlas containing subcortical regions
    subcort_regions :   list of ints. Default=None. Should be optionally provided if a custom
                        atlas_file is used. The list of ints should correspond
                        to the subcortical region numbers in the atlas
    tmp_dir         :   pathlib Path or str. Default '/tmp/brainvistools'. Directory
                        to use for temporary files
    save_cmap       :   bool. Default=False. Option to save colormaps as png
    pos_colours	   :   list of float tuples (RGB). colour map for positive values
    neg_colours	   :   list of float tuples (RGB). colour map for negative values
    Returns
    -------
    None
    """
    output_file = Path(output_file)
    if not atlas_file:
        atlas_file = ATLAS_DICT[atlas_name]
    if not subcort_regions:
        subcort_regions = SUBCORT_REGIONS_DICT[atlas_name]
        
    if not pos_colours:
        pos_colours = [(.886,.761,.133),(.847,.631,.031),(.922,.0,.02)]
    if not neg_colours:
        neg_colours = [(.698,.757,.463),(.188,.533,.639)]

    def make_subcort_atlas_file(atlas_file,subcort_regions):
        """Make subcortical atlas from atlas file by setting regions not in subcort_regions
        to 0.0
        Parameters
        ----------
        atlas_file      :   str or pathlib Path. Path to atlas nifti
        subcort_regions :   list of ints. List of subcortical region numbers (as
                            they appear in atlas_file data)
        """
        atlas_img = nib.load(atlas_file)
        atlas_affine = atlas_img.affine
        subcort_data = atlas_img.get_fdata()
        subcort_data[np.where(np.isin(subcort_data,subcort_regions,invert=True))] = 0.0
        subcort_img = nib.Nifti1Image(subcort_data,atlas_affine)
        return subcort_img

    subcort_atlas_img = make_subcort_atlas_file(atlas_file, subcort_regions)

    def split_data_neg_pos_abs(data):
        """Split data into negative and positive subsets and assign absolute value to
        negative subset
        Parameters
        ----------
        data        :   1D array. Data with same number and order of values as labels
        Returns
        -------
        data_neg    :   1D array. Absolute value of negative values of data, with
                        positive values set to 0.0
        data_pos    :   1D array. Positive values of data left as is, with negative
                        values set to 0.0
        """
        data_neg = copy.deepcopy(data)
        data_pos = copy.deepcopy(data)
        data_neg[np.where(data > 0.0)] = 0.0
        data_neg = np.abs(data_neg)
        data_pos[np.where(data < 0.0)] = 0.0
        return data_neg, data_pos

    data_neg, data_pos = split_data_neg_pos_abs(data)
    data_neg_img = assign_to_atlas_img(data_neg,subcort_atlas_img)
    data_neg_min = np.min(data_neg)
    data_neg_max = np.max(data_neg)
    print('all_neg_max',data_neg_max)
    data_pos_img = assign_to_atlas_img(data_pos,subcort_atlas_img)
    data_pos_min = np.min(data_pos)
    data_pos_max = np.max(data_pos)
    print('all_pos_max',data_pos_max)
    data_max = np.max([data_neg_max,data_pos_max])

    # Nilearn plotting method
    BG_img = nib.load(STANDARD_FILE)

    greyscale_cmap = LinearSegmentedColormap.from_list('greyscale', [(0,0,0),(1,1,1)],N=1000)
    black_cmap = LinearSegmentedColormap.from_list('greyscale', [(0,0,0),(0,0,0)],N=1000)
#    yellow_red_cmap = LinearSegmentedColormap.from_list('greyscale', [(.886,.761,.133),(.847,.631,.031),(.922,.0,.02)],N=100)
#    green_blue_cmap = LinearSegmentedColormap.from_list('greyscale', [(.698,.757,.463),(.188,.533,.639)],N=100)
    pos_cmap = LinearSegmentedColormap.from_list('greyscale', pos_colours,N=100)
    neg_cmap = LinearSegmentedColormap.from_list('greyscale', neg_colours,N=100)

    def make_subcort_images(BG_img,subcort_overlay_img,neg_img,pos_img,img_thresh,vmax_neg,vmax_pos,vmin=0.0,zoom_scale=3):
        """Create subcortical image for a three views (LH and RH sagittal plus axial)
        Parameters
        ----------
        BG_img              :   Nifti1Image image from nibabel.load. Background
                                image
        subcort_overlay_img :   Nifti1Image image from nibabel.load. Overlay from
                                subcortical atlas. Will be set to black before applying
                                neg_img and pos_img data
        neg_img             :   Nifti1Image image from nibabel.load. Assigned negative
                                data values (but absolute value)
        pos_img             :   Nifti1Image image from nibabel.load. Assigned positive
                                data values
        img_thresh          :   float. Threshold value (positive)
        vmax_neg            :   float. Max absolute value of negative data points
        vmax_pos            :   float. Max value of positive data points
        vmin                :   float. Default=0. Min value for colorbar
        zoom_scale          :   int. Zoom scaling to use for ndimage.zoom. Shouldn't
                                change this from default=3
        Returns
        -------
        img_np_zoomed_LH    :   2D array. LH sagittal view
        img_np_zoomed_RH    :   2D array. RH sagittal view
        img_np_zoomed_axial :   2D array. Axial view
        """
        def zoom(plot_img,zoom_scale):
            """
            Parameters
            ----------
            plot_img        :   nilearn plotting.plot_img output
            zoom_scale      :   int. zoom scaling to use for ndimage.zoom
            Returns
            -------
            img_np_zoomed   :   plot image zoomed to subcortical regions as 2d array
            """
            tmp_img = tmp_dir.joinpath('subcort_figure.png')
            plot_img.savefig(tmp_img)
            img = Image.open(tmp_img)
            img_np = np.array(img.getdata()).reshape(img.size[1],img.size[0], 4)
            img_np_zoomed = ndimage.zoom(img_np, (zoom_scale,zoom_scale,1), order=0)
            return img_np_zoomed


        def subcort_plotting(display_mode,cut_coords):
            """Plot data using nilearn.plot_img
            Parameters
            ----------
            display_mode    :   nilearn.plotting.plot_img parameter
            cut_coords      :   nilearn.plotting.plot_img parameter
            Returns
            -------
            subcort_fig     :   nilearn.plotting.plot_img plot
            """
            subcort_fig = plotting.plot_img(    BG_img, 
                                                display_mode=display_mode,
                                                cut_coords=cut_coords,
                                                draw_cross=False,
                                                cmap=greyscale_cmap,
                                                figure=1,
                                                annotate=False,
                                                )

            subcort_fig.add_overlay(            subcort_overlay_img,
                                                cmap=black_cmap,
                                                threshold=1.0,
                                                )
            if vmax_pos > 0:
                subcort_fig.add_overlay(        pos_img,
                                                vmin=vmin,
                                                vmax=vmax_pos,
                                                cmap=pos_cmap,
                                                threshold=img_thresh,                 
                                                )
            if vmax_neg > 0:
                subcort_fig.add_overlay(        neg_img, 
                                                vmin=vmin,
                                                vmax=vmax_neg,
                                                cmap=neg_cmap,
                                                threshold=img_thresh,                 
                                                )
            return subcort_fig
        
        LH_sag_coord = -28.0
        RH_sag_coord = 30.0
        axial_coord = 0.0

        subcort_fig_LH = subcort_plotting('x',[LH_sag_coord])

        xxadj = -15
        xyadj = -9
        xbase = 35
        subcort_fig_LH.axes[LH_sag_coord].ax.set_xlim(-xbase+xxadj,xbase+xxadj)
        subcort_fig_LH.axes[LH_sag_coord].ax.set_ylim(-xbase+xyadj,xbase+xyadj)
        zoomed_LH_np = zoom(subcort_fig_LH,zoom_scale)[:,:,:-1]

        subcort_fig_RH = subcort_plotting('x',[RH_sag_coord])

        xxadj = -15
        xyadj = -9
        xbase = 35
        subcort_fig_RH.axes[RH_sag_coord].ax.set_xlim(-xbase+xxadj,xbase+xxadj)
        subcort_fig_RH.axes[RH_sag_coord].ax.set_ylim(-xbase+xyadj,xbase+xyadj)
        zoomed_RH_np = zoom(subcort_fig_RH,zoom_scale)[:,:,:-1]
        
        subcort_fig_axial = subcort_plotting('z',[axial_coord])

        zxadj = 1
        zyadj = -10
        zbase = 50
        subcort_fig_axial.axes[axial_coord].ax.set_xlim(-zbase+zxadj,zbase+zxadj)
        subcort_fig_axial.axes[axial_coord].ax.set_ylim(-zbase+zyadj,zbase+zyadj)
        zoomed_axial_np = zoom(subcort_fig_axial,zoom_scale)[:,:,:-1]
        
        return zoomed_LH_np, zoomed_RH_np, zoomed_axial_np

    vmin = 0
    if data_pos_max == 0:
        vmin = thresh if thresh > 0 else data_neg_min
    elif data_neg_max == 0:
        vmin = thresh if thresh > 0 else data_pos_min
    LH_np, RH_np, axial_np = make_subcort_images(   BG_img=BG_img,
                                                    subcort_overlay_img=subcort_atlas_img,
                                                    neg_img=data_neg_img,
                                                    pos_img=data_pos_img,
                                                    img_thresh=thresh,
                                                    vmax_neg=data_max,
                                                    vmax_pos=data_max,
                                                    vmin=vmin)

    def combine_figures(LH_np,RH_np,axial_np):
        """Combine LH and RH sagittal views and axial view into one image
        Parameters
        ----------
        LH_np       :   2D array. LH sagittal image
        RH_np       :   2D array. RH sagittal image
        axial_np    :   2D array. axial image
        Returns
        -------
        comb_np     :   2D array. combined image
        """
        x,y,z = LH_np.shape
        comb_np = np.empty((x,y*3,z))
        comb_np[:,:y,:] = LH_np
        comb_np[:,y:2*y,:] = axial_np
        comb_np[:,2*y:,:] = np.flip(RH_np,axis=1)
        return comb_np

    comb_np = combine_figures(LH_np,RH_np,axial_np)
    Image.fromarray(comb_np.astype('uint8'),'RGB').save(output_file)

    def save_colorbar(cmap,cmap_file):
        """Save colorbar used for figure
        Parameters
        ----------
        cmap        :   matplotlib.colors.LinearSegmentedColormap. Defined above 
                        from list
        cmap_file   :   pathlib Path or str. File location to save cmap png
        Returns
        -------
        None
        """
        plt.figure()
        a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
        plt.imshow(a, cmap=cmap, origin='lower')
        plt.axis("off")
        plt.savefig(cmap_file)
        
    if save_cmap:
        cmap_file_pos = output_file.parent.joinpath(f'{output_file.stem}_pos_colorbar.png')
        cmap_file_neg = output_file.parent.joinpath(f'{output_file.stem}_neg_colorbar.png')
        save_colorbar(pos_cmap,cmap_file_pos)
        save_colorbar(neg_cmap,cmap_file_neg)
