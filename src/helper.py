import numpy as np
import json
from collections import Iterable
import os
import sys
from sklearn import preprocessing
from scipy import sparse
from ..lib import models, graph, coarsening


def prepare_data(X, y, split, data_format, random_seed):
    """
    Helper function for data preparation.
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(X.shape[0])

    n_train = int(X.shape[0] * split)
    n_val = X.shape[0] - n_train
    X_train = X[indices[:n_train]]
    X_val = X[indices[n_train:n_train + n_val]]
    y_train = y[indices[:n_train]]
    y_val = y[indices[n_train:n_train + n_val]]

    if data_format == 'vector':
        X_vec = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X_vec = preprocessing.scale(X_vec)
        X_train_vec = X_vec[indices[:n_train]]
        X_val_vec = X_vec[indices[n_train:n_train + n_val]]
        train_val_data = [X_train_vec, y_train, X_val_vec, y_val]
    elif data_format == 'image':
        X_train = (X_train - np.mean(X)) / np.std(X)
        X_val = (X_val - np.mean(X)) / np.std(X)
        X_train_image = np.expand_dims(X_train, axis=-1)
        X_val_image = np.expand_dims(X_val, axis=-1)
        train_val_data = [X_train_image, y_train, X_val_image, y_val]
    else:
        X_train = (X_train - np.mean(X)) / np.std(X)
        X_val = (X_val - np.mean(X)) / np.std(X)
        train_val_data = [X_train, y_train, X_val, y_val]

    return train_val_data


def structure_data(A_matrix, X_train, X_val):
    """
    Structure input data for graph CNN including the graph Laplacians and permutated data.
    """
    A = sparse.csr_matrix(A_matrix).astype(np.float32)
    graphs, perm = coarsening.coarsen(A, levels=3, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    X_train_list = []
    X_val_list = []
    for i in range(X_train.shape[2]):
        X_train_list.append(coarsening.perm_data(X_train[:, :, i], perm))
        X_val_list.append(coarsening.perm_data(X_val[:, :, i], perm))

    X_train_graph = np.stack(X_train_list, axis=-1)
    X_val_graph = np.stack(X_val_list, axis=-1)
    return L, X_train_graph, X_val_graph


def graph_model_params(n_filter, dense_size, n_graph, num_epochs, batch_size, n_train):
    """
    Parameters of graph CNN.
    """
    params = dict()
    params['dir_name'] = 'demo'
    params['num_epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['eval_frequency'] = n_train / batch_size

    # Building blocks.
    params['filter'] = 'chebyshev5'
    params['brelu'] = 'b1relu'
    params['pool'] = 'apool1'

    # Number of classes.
    C = 2

    # Architecture.
    params['F'] = [n_filter, n_filter]  # Number of graph convolutional filters.
    params['K'] = [10, 10]  # Polynomial orders.
    params['p'] = [4, 2]  # Pooling sizes.
    params['M'] = [dense_size, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 5e-4
    params['dropout'] = 0.9
    params['learning_rate'] = 1e-3
    params['decay_rate'] = 0.95
    params['momentum'] = 0.9
    params['decay_steps'] = n_train / batch_size

    params['n_graph'] = n_graph
    return params


def flatten_list(items):
    """
    Yield items from any nested iterable.
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten_list(x):
                yield sub_x
        else:
            yield x


def save_details(details, directory):
    """
    Save details in a json file and creates a unique id.

    :param details: (dict) model details {'model': 'graph_cnn', 'epochs': 50} (must be non-nested)
    :param directory: (string) directory path
    :return: (int) unique model id
    """
    details_list = list(flatten_list(details.values()))
    hash_id = hash(frozenset(details_list)) % ((sys.maxsize + 1) * 2)
    js = json.dumps(details)
    path = os.path.join(directory, 'model_details_{}.json'.format(model_id))
    with open(path, 'w') as f:
        f.write(js)
    return hash_id
