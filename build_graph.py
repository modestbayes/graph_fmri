import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from data_utils import *


def spatial_distance_graph(adj_matrix_path, pct):
    """
    Graph based on spatial distance between brain regions.

    :param adj_matrix_path: (string) path to already calculated csv file
    :param pct: (float) percentage between 0 and 100 to make the graph sparse
    :return: (2d numpy array) of size [100, 100] for graph adjacency matrix
    """
    adj_matrix = np.genfromtxt(adj_matrix_path, delimiter=',')
    adj_matrix[np.isnan(adj_matrix)] = 0
    thres = np.percentile(adj_matrix, pct)
    A_spatial = adj_matrix[:100, :100]
    A_spatial[A_spatial < thres] = 0
    return A_spatial


def random_graph(random_seed, pct):
    """
    Graph that is randomly generated.

    :param random_seed: (int) seed for randomness
    :param pct: (float) percentage between 0 and 100 to make the graph sparse
    :return: (2d numpy array) of size [100, 100] for graph adjacency matrix
    """
    np.random.seed(random_seed)
    A_random = np.random.uniform(low=0, high=1, size=(100, 100))
    thres = 1 - 0.01 * pct
    A_random[A_random > thres] = 0
    A_random = A_random / thres
    np.fill_diagonal(A_random, 1)
    return A_random


def pearson_correlation_graph(time_series_dir, pct):
    """
    Graph based on Pearson correlation between time series.

    :param time_series_dir: (string) directory of time series
    :param pct: (float) percentage between 0 and 100 to make the graph sparse
    :return: (2d numpy array) of size [100, 100] for graph adjacency matrix
    """
    subject_id_list = get_subject_id(time_series_dir)
    n = len(subject_id_list)
    all_pearson = np.zeros((n, 100, 100))
    for i, subject_id in enumerate(tqdm(subject_id_list)):
        time_series_data = load_time_series(time_series_dir, subject_id)
        time_series_data = preprocess_time_series(time_series_data)
        for j in range(100):
            for k in range(j, 100):
                all_pearson[i, j, k] = pearsonr(time_series_data[j, :], time_series_data[k, :])[0]
                all_pearson[i, k, j] = all_pearson[i, j, k]
    mean_pearson = np.mean(all_pearson, axis=0)
    thres = np.percentile(mean_pearson, pct)
    A_pearson = mean_pearson
    A_pearson[A_pearson < thres] = 0
    return A_pearson
