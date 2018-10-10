import numpy as np
from helper import prepare_data, save_details
from other_models import fully_connected_model

X_path = '/Users/linggeli/cnn_graph/fmri/clas_data/X_259sub_40reg_16coef.npy'
y_path = '/Users/linggeli/cnn_graph/fmri/clas_data/y_259sub.npy'
n_coef = 8
arch_set = [[100, 40], [100, 20], [200, 40], [200, 20], [40, 20],
            [200, 100, 40], [200, 40, 20], [100, 40, 20], [200, 100, 20], [200, 100, 40, 20]]
p_range = [0, 0.1, 0.2]

exp_details = {'data': '259sub_40reg_16coef',
               'n_coef': 8,
               'arch_set': [[100, 40], [100, 20], [200, 40], [200, 20], [40, 20],
                            [200, 100, 40], [200, 40, 20], [100, 40, 20], [200, 100, 20], [200, 100, 40, 20]],
               'p_range': [0, 0.1, 0.2]}

exp_id = save_details(exp_details, '/Users/linggeli/cnn_graph/fmri/results')
print(exp_id)

X = np.load(X_path)[:, :, :n_coef]
y = np.load(y_path)
train_val_data = prepare_data(X, y, split=0.7, data_format='vector', random_seed=0)

for arch in arch_set:
    for p in p_range:
        val_acc = fully_connected_model(train_val_data, layers=arch, drop_prob=p)
        print('Model architecture: {arch}; Dropout probability: {p}; '
              'Validation accuracy: {val_acc}'.format(arch=arch, p=p, val_acc=val_acc))
