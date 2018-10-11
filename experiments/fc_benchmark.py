import numpy as np
import sys
sys.path.append('/Users/linggeli/graph_fmri/')
from graph_fmri.src.helper import prepare_data, save_details
from graph_fmri.src.benchmark import fully_connected_model


X_path = '/Users/linggeli/graph_fmri/clas_data/features_259subjects_filtered.npy'
y_path = '/Users/linggeli/graph_fmri/clas_data/labels_259subjects.npy'
n_coef = 16
arch_set = [[100, 40], [100, 20], [200, 40], [200, 20], [40, 20],
            [200, 100, 40], [200, 40, 20], [100, 40, 20], [200, 100, 20], [200, 100, 40, 20]]
p_range = [0, 0.1, 0.2]

exp_details = {'data': '259subjects_filtered',
               'n_coef': 16,
               'arch_set': [[100, 40], [100, 20], [200, 40], [200, 20], [40, 20],
                            [200, 100, 40], [200, 40, 20], [100, 40, 20], [200, 100, 20], [200, 100, 40, 20]],
               'p_range': [0, 0.1, 0.2]}

exp_id = save_details(exp_details, '/Users/linggeli/graph_fmri/output/')
print(exp_id)

brain_regions = np.genfromtxt('/Users/linggeli/graph_fmri/clas_data/brain_regions.csv',
                              dtype=int, delimiter=',')[:40] - 1
X = np.load(X_path)[:, brain_regions, :n_coef]
y = np.load(y_path)
train_val_data = prepare_data(X, y, split=0.7, data_format='vector', random_seed=0)

for arch in arch_set:
    for p in p_range:
        for i in range(10):
            val_acc = fully_connected_model(train_val_data, layers=arch, drop_prob=p)
            print('Model architecture: {arch}; Dropout probability: {p}; '
                  'Validation accuracy: {val_acc}'.format(arch=arch, p=p, val_acc=val_acc))
