import numpy as np
import networkx as nx
import scipy
import math
import random
# Install graph_tool with `conda install -c conda-forge graph-tool`
import graph_tool as gt
from graph_tool import topology

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

def mean_first_passage_time(adjacency_matrix):
    """Mean first passage time (eq'n 7.14 from "Fundamentals of Brain Network Analysis")
    Results in a directed (asemetric) adjacency matrix
    Parameters
    ----------
    adjacency_matrix    :   2-dimensional ndarray
                            symmetrical weighted adjacency matrix with 0.0 diagonal values
    Returns
    ----------
    Xmean               :   2-dimensional ndarray (asymmetric)
    """
    W = adjacency_matrix.copy()
    regions_num = W.shape[0]
    I = np.identity(regions_num,dtype="float")
    S = np.zeros_like(W)
    strengths = np.sum(W,axis=0)
    for s in range(regions_num):
        S[s,s] = strengths[s]

    Sinv = np.linalg.inv(S)

    U = np.matmul(W,Sinv)

    Xmean = np.zeros((regions_num,regions_num))

    for j in range(regions_num):
        Uj = U.copy()
        Uj[j,:] = 0.0
        I_sub_Uj_inv_row_sums = np.sum(np.linalg.inv(I - Uj),axis=0)
        for i in range(regions_num):
            if i != j:
                #I have checked, and this gives identical results to summing each element individually as in the equation
                Xmean[i,j] = I_sub_Uj_inv_row_sums[i]
    
    return Xmean

def communicability(adjacency_matrix):
    """Communicability, calculated using the matrix exponential of the normalized U matrix (eq'n 7.18 "Fundamentals of Brain Network Analysis" with substitution for A as mentioned on p 245)
    Results in undirected (symmetric) adjacency matrix
    Parameters
    ----------
    adjacency_matrix    :   2-dimensional ndarray
                            symmetrical weighted adjacency matrix with 0.0 diagonal values
    Returns
    ----------
    com_ij              :   2-dimensional ndarray (symmetric)
    """
    W = adjacency_matrix.copy()
    regions_num = W.shape[0]
    S = np.zeros_like(W)
    strengths = np.sum(W,axis=0)
    for s in range(regions_num):
        S[s,s] = strengths[s]

    Sinv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(S))

    Unorm = np.matmul(np.matmul(Sinv_sqrt,W),Sinv_sqrt)

    com_ij = scipy.linalg.expm(Unorm)
    return com_ij

def numpy_to_graph_tool(adjacency_matrix):
    """Read numpy matrix to graph-tool
    Parameters
    ----------
    adjacency_matrix    :   2-dimensional ndarray
                            weighted adjacency matrix
    Returns
    ----------
    g                 :   graph_tool.Graph
    """
    g = gt.Graph(directed=False)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    nnz = np.nonzero(np.triu(adjacency_matrix,1))
    nedges = len(nnz[0])
    g.add_edge_list(np.hstack([np.transpose(nnz),np.reshape(adjacency_matrix[nnz],(nedges,1))]),eprops=[edge_weights])
    return g

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

def shortest_path_lengths(adjacency_matrix):
    """Calculates the shortest path length using graph_tools (much more efficient than networkx)
    Expects "inverted" weights corresponding to resistence (use the matrix_weight_invert() function)
    Assumes a symmetric matrix and therefore only calculates for upper triangle and copies these transposed values to bottome triangle
    I have tested, and these results are equivalent to nx.shortest_path_length
    Parameters
    ----------
    adjacency_matrix    :   2-dimensional ndarray
                            weighted adjacency matrix (inverted cells, lower values mean less resistance and better connectivity)
    Returns
    ----------
    spl_np:             :   2-dimensional ndarray (symmetric)
    """
    regions_num = adjacency_matrix.shape[0]
    graph = numpy_to_graph_tool(adjacency_matrix.copy())
    graph.vertex_properties['spl'] = topology.shortest_distance(graph,weights=graph.edge_properties["weight"])
    spl_np = np.zeros_like(adjacency_matrix,dtype=float)
    for i in range(regions_num):
        #try/except was added in case of disconnected networks, assigning np.inf in no connection (1/np.inf will give 0 for efficiency)
        try:
            v = graph.vertex(i)
            for j in range(regions_num):
                if j > i:
                    try:
                        spl_np[i,j] = graph.vp.spl[v][j]
                    except IndexError:
                        spl_np[i,j] = np.inf
        except ValueError:
            for j in range(regions_num):
                if j > i:
                    spl_np[i,j] = np.inf
    # Transpose upper triangle and copy to lower triangle
    spl_np += spl_np.T
    return spl_np

def rubinov_sporns_null_model(adjacency_matrix,seed=None):
    """Null hypothesis: Randomized SC Rubinov Sporns 2011
    Rubinov & Sporns (2011) null model from eq'n 10.4 in Fundamentals of Brain Network Analysis
    Step one for swapping pos/neg connections not applied as our SC matrix has only positive values
    Keeps degree and weights as constant as possible, while using algorithm to approximately maintain strength of nodes
    Parameters
    ----------
    adjacency_matrix    :   2-D array representing undirected/symmetric adjacency matrix for network with 0.0 diagonal values
    seed                :   int. default = None, will result in random seed. Change to set your own initial seed value for the `random` library that will be used to shuffle connection weights
    Returns
    -------
    null_matrix         :   2-D array representing undirected/symmetric null model adjacency matrix
    """
    regions_n = adjacency_matrix.shape[0]
    
    G = nx.from_numpy_matrix(adjacency_matrix)
    random_conn = list(G.edges)
    
    if seed:
        random.seed(seed)
    random.shuffle(random_conn)
    weights = [G.edges[r[0],r[1]]['weight'] for r in random_conn]
    weights = list(np.sort(np.array(weights))[::-1])
    
    strengths = np.sum(adjacency_matrix,axis=0)
    
    null_matrix = np.zeros((regions_n,regions_n))
    null_strengths = np.sum(null_matrix,axis=0)
    # Calculate surplus weights based on Rubinov and Sporns (2011) algorithm, then rank these surplus weights
    surplus_weights = [(strengths[r[0]] - null_strengths[r[0]]) * (strengths[r[1]] - null_strengths[r[1]]) for r in random_conn]
    # Using two np.argsorts() so that the value at each index represents the rank of that surplus_weight
    surplus_ranks = np.array(surplus_weights).argsort()[::-1].argsort()
    
    for x,n in enumerate(random_conn):
        null_matrix[n[0],n[1]] = weights.pop(surplus_ranks[0])
        null_matrix[n[1],n[0]] = null_matrix[n[0],n[1]]
        null_strengths = np.sum(null_matrix,axis=0)
        # Recalculate and rank surplas weights at each step
        surplus_weights = [(strengths[r[0]] - null_strengths[r[0]]) * (strengths[r[1]] - null_strengths[r[1]]) for r in random_conn[x+1:]]
        surplus_ranks = np.array(surplus_weights).argsort()[::-1].argsort()
    return null_matrix