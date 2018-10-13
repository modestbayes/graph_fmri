import numpy as np
import os
import sys
sys.path.append('/Users/linggeli/graph_fmri/')
from graph_fmri.src.helper import *
from graph_fmri.src.build_graph import *
from graph_fmri.src.graph_models import multi_cgcnn


data_dir = '/Users/linggeli/graph_fmri/clas_data/'
n_coef = 16

brain_regions = np.genfromtxt(os.path.join(data_dir, 'brain_regions.csv'), dtype=int, delimiter=',')[:40] - 1
X = np.load(os.path.join(data_dir, 'features_259subjects_filtered.npy'))[:, brain_regions, :n_coef]
y = np.load(os.path.join(data_dir, 'labels_259subjects.npy'))
A_spatial = spatial_distance_graph(os.path.join(data_dir, 'adj_matrix.csv'), brain_regions, 70)

X_train, y_train, X_val, y_val = prepare_data(X, y, 0.7)
L, X_train_graph, X_val_graph = structure_data(A_spatial, X_train, X_val)

params = graph_model_params(n_filter=20, dense_size=20,
                            n_graph=n_coef, keep_prob=1.0,
                            epochs=20, batch_size=20,
                            n_train=X_train.shape[0], verbose=False)

for i in range(10):
    model = multi_cgcnn(L, **params)
    accuracy, loss, t_step = model.fit(X_train_graph, y_train, X_val_graph, y_val)
