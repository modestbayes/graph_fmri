import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
sys.path.append('/Users/linggeli/graph_fmri/')
from graph_fmri.src.data_utils import *


def spatial_distance_graph(adj_matrix_path, brain_regions, pct):
    """
    Graph based on spatial distance between brain regions.

    :param adj_matrix_path: (string) path to already calculated csv file
    :param brain_regions: (1d numpy array) of brain region indices
    :param pct: (float) percentage between 0 and 100 to make the graph sparse
    :return: (2d numpy array) graph adjacency matrix
    """
    adj_matrix = np.genfromtxt(adj_matrix_path, delimiter=',')
    adj_matrix[np.isnan(adj_matrix)] = 0
    thres = np.percentile(adj_matrix, pct)
    A_spatial = adj_matrix[brain_regions, :][:, brain_regions]
    A_spatial[A_spatial < thres] = 0
    return A_spatial


def random_graph(n, pct, random_seed=0):
    """
    Graph that is randomly generated.

    :param random_seed: (int) seed for randomness
    :param n: (int) graph size
    :param pct: (float) percentage between 0 and 100 to make the graph sparse
    :return: (2d numpy array) graph adjacency matrix
    """
    np.random.seed(random_seed)
    A_random = np.random.uniform(low=0, high=1, size=(n, n))
    thres = 1 - 0.01 * pct
    A_random[A_random > thres] = 0
    A_random = A_random / thres
    np.fill_diagonal(A_random, 1)
    return A_random


def calculate_pearson_correlation(time_series_dir, brain_regions):
    """
    Calculate Pearson correlation between time series across subjects.

    :param time_series_dir: (string) directory of time series
    :param brain_regions: (1d numpy array) of brain region indices
    :return: (2d numpy array) mean Pearson correlation matrix
    """
    subject_id_list = get_subject_id(time_series_dir)
    n = len(subject_id_list)
    size = brain_regions.shape[0]
    all_pearson = np.zeros((n, size, size))
    for i, subject_id in enumerate(tqdm(subject_id_list)):
        time_series_data = load_time_series(time_series_dir, subject_id)[brain_regions, :]
        time_series_data = preprocess_time_series(time_series_data)
        for j in range(size):
            for k in range(j, size):
                all_pearson[i, j, k] = pearsonr(time_series_data[j, :], time_series_data[k, :])[0]
                all_pearson[i, k, j] = all_pearson[i, j, k]
    mean_pearson = np.mean(all_pearson, axis=0)
    return mean_pearson


def pearson_correlation_graph(mean_pearson, pct):
    """
    Graph based on Pearson correlation between time series.

    :param mean_pearson: (2d numpy array) mean Pearson correlation matrix
    :param pct: (float) percentage between 0 and 100 to make the graph sparse
    :return: (2d numpy array) graph adjacency matrix
    """
    thres = np.percentile(mean_pearson, pct)
    A_pearson = mean_pearson
    A_pearson[A_pearson < thres] = 0
    return A_pearson


def mask_connections(A_matrix, pct):
    """
    Mask a percentage of edges on a graph with zeros.

    :param A_matrix: (2d numpy array) of size [100, 100] for graph adjacency matrix
    :param pct: (float) percentage between 0 and 100 to remove edges
    :return: (2d numpy array) graph adjacency matrix with zero masks
    """
    edges = A_matrix[(A_matrix < 1) & (A_matrix > 0)].flatten()
    thres = np.percentile(edges, 100 - pct)
    select = (A_matrix < 1) & (A_matrix > thres)
    A_matrix_mask = A_matrix.copy()
    A_matrix_mask[select] = 0
    return A_matrix_mask


def cut_connections(A_matrix, p_values, k):
    """
    Remove connections to brain regions based on p-values.

    :param A_matrix: (2d numpy array) of size [100, 100] for graph adjacency matrix
    :param p_values: (1d numpy array) of brain region p-values
    :param k: (int) number of brain regions to cut
    :return: (2d numpy array) graph adjacency matrix with cut connections
    """
    top_regions = p_values.argsort()[:k]
    A_matrix_cut = A_matrix.copy()
    for r in top_regions:
        A_matrix_cut[r, :] = 0
        A_matrix_cut[:, r] = 0
    np.fill_diagonal(A_matrix_cut, 1)
    return A_matrix_cut
