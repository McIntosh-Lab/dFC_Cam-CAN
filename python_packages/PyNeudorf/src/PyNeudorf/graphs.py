import numpy as np
import math

def flat_upper_tri_regions_n(flat_upper_tri_len):
    """Get number of regions in original matrix from number of cells in upper triangle (only valid for square matrix)
    Derived by solving for x in y = (x)*(x-1)/2 where x is the number of regions and y is the number of cells in the upper triangle
    Parameters
    ----------
    flat_upper_tri_len  :   int
                            must be positive
    Returns
    -------
    original_regions_n:   int
    """
    original_regions_n = 1/2 + math.sqrt(1/4 + 2*flat_upper_tri_len)
    return int(original_regions_n)

def matrix_to_flat_triu(matrix):
    """Take numpy matrix and get flat upper triangle
    Parameters
    ----------
    matrix              :   ndarray
                            square adjacency matrix
    Returns
    -------
    triu                :   1-D ndarray with all cells in upper triangle
    """
    regions_n = matrix.shape[0]
    triu = matrix[np.triu_indices(regions_n,k=1)]
    return triu

def flat_to_square_matrix(triu_flat):
    """Load flattened upper triangle from txt and reshape to square matrix
    Parameters
    ----------
    triu_flat           :   1-D ndarray
                            flattened upper triangle data
    Returns
    -------
    matrix                 :   2-dimensional square ndarray
    """
    regions_n = flat_upper_tri_regions_n(triu_flat.size)
    matrix = np.zeros((regions_n,regions_n))
    matrix[np.triu_indices(regions_n,k=1)] = triu_flat.copy()
    matrix += matrix.T
    return matrix

def threshold_matrix(matrix,thresh,thresh_direction):
    """Take in `matrix` and threshold according to `thresh`, in the direction indicated by `thresh_direction`
    Parameters
    ----------
    matrix              :   ndarray
    thresh              :   float, positive (even if using neg thresh_direction, will switch to -1*thresh in function)
    thresh_direction    :   string
                            'pos'   :   positive, will set all values less than thresh to 0.0
                            'neg'   :   negative, will set all values greater than thresh to 0.0
                            'both'  :   positive and negative, will set all values between -thresh and +thresh to 0.0
    Returns
    -------
    matrix_thresholded  :   ndarray
    """
    matrix_thresholded = np.zeros_like(matrix)
    if thresh_direction == 'both':
        matrix_thresholded[np.where(matrix >= thresh)] = np.copy(matrix[np.where(matrix >= thresh)])
        matrix_thresholded[np.where(matrix <= -1*thresh)] = np.copy(matrix[np.where(matrix <= -1*thresh)])
    elif thresh_direction == 'pos':
        matrix_thresholded[np.where(matrix >= thresh)] = np.copy(matrix[np.where(matrix >= thresh)])
    elif thresh_direction == 'neg':
        matrix_thresholded[np.where(matrix <= -1*abs(thresh))] = np.copy(matrix[np.where(matrix <= -1*abs(thresh))])
    return matrix_thresholded

def stride_diag_remove(matrix):
    """This function removes the diagonal and shifts the upper right triangle to the left
    New dimensions will be (regions_n,regions_n-1)
    Parameters
    ----------
    matrix              :   2-D square ndarray
    Returns
    -------
    out                 :   2-D (regions_n-1,regions_n)
    """
    regions_n = matrix.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = matrix.strides
    out = strided(matrix.ravel()[1:], shape=(regions_n-1,regions_n), strides=(s0+s1,s1)).reshape(regions_n,-1)
    return out

def density(matrix):
    """Report the density of a matrix as a proportion of the nonzero upper triangle cells over the total upper triangle cells (excluding diagonal)
    Parameters
    ----------
    matrix              :   2-D square ndarray
    Returns
    -------
    density            :   float
    """
    matrix_triu = np.triu(matrix,k=1)
    nonzero_off_diag_cells_n = np.nonzero(matrix_triu)[0].size
    total_off_diag_cells_n = matrix_triu.size
    density = nonzero_off_diag_cells_n / total_off_diag_cells_n
    return density

def hemisphere_analysis(matrix):
    """Returns the number of nonzero connections in a symmetric matrix stratefied by hemisphere
    Regions 0 to (matrix.shape[0] // 2) should be LH, and the rest RH
    Parameters
    ----------
    matrix                                      :   2-D square symmetric ndarray with even N regions
    Returns
    -------
    interhemi_n, lh_intrahemi_n, rh_intrahemi_n :   tuple of ints
    """
    half_regions_n = matrix.shape[0] // 2
    interhemi_matrix = matrix[:half_regions_n,half_regions_n:]
    lh_intrahemi_matrix = np.triu(matrix[:half_regions_n,:half_regions_n],k=1)
    rh_intrahemi_matrix = np.triu(matrix[half_regions_n:,half_regions_n:],k=1)
    interhemi_n = np.nonzero(interhemi_matrix)[0].size
    lh_intrahemi_n = np.nonzero(lh_intrahemi_matrix)[0].size
    rh_intrahemi_n = np.nonzero(rh_intrahemi_matrix)[0].size
    return interhemi_n, lh_intrahemi_n, rh_intrahemi_n

def matrix_weight_invert(adjacency_matrix):
    """Transform cell values to 1/x, avoiding division by zero and leaving as 0.0 (will be counted as non-edge by graph_tools)
    Not efficient by any means, but not that intensive for smallish matrices.
    Parameters
    ----------
    adjacency_matrix    :   2-dimensional ndarray
                            weighted adjacency matrix
    Returns
    ----------
    adjacency_matrix_inv:   2-dimensional ndarray (symmetric)
    """
    regions_num = adjacency_matrix.shape[0]
    adjacency_matrix_inv = np.zeros_like(adjacency_matrix)
    for i in range(regions_num):
        for j in range(regions_num):
            if adjacency_matrix[i][j] != 0:
                adjacency_matrix_inv[i][j] = 1/adjacency_matrix[i][j]
    return adjacency_matrix_inv