from sklearn import preprocessing
from data_utils import *
from other_models import *
import pickle


N_COEF = 5
time_series_dir = '/Users/linggeli/cnn_graph/data/time_series/cup/'
behavioral_dir = '/Users/linggeli/cnn_graph/data/fMRIbehav/cup/'
subject_id_list = get_subject_id(time_series_dir)
X = get_all_features(time_series_dir, behavioral_dir, subject_id_list,
                     n_time=64, coef_idx=range(N_COEF))[:, :100, :]
y = get_all_labels(behavioral_dir, subject_id_list) - 1
print(X.shape)
print(y.shape)

# training and validation data split
n_train = 600
n_val = 200
np.random.seed(0)
indices = np.random.permutation(X.shape[0])

X_train = X[indices[:n_train]]
X_val = X[indices[n_train:n_train+n_val]]
y_train = y[indices[:n_train]]
y_val = y[indices[n_train:n_train+n_val]]

# flatten data to feature vectors
X_vec = X.reshape((X.shape[0], X.shape[1] * N_COEF))
X_vec = preprocessing.scale(X_vec)
X_train_vec = X_vec[indices[:n_train]]
X_val_vec = X_vec[indices[n_train:n_train+n_val]]
train_val_data = [X_train_vec, y_train, X_val_vec, y_val]

# logistic regression with L1 penalty
y_hat_logistic = logistic_model(train_val_data, verbose=True)
acc_logistic = np.mean((y_hat_logistic > 0.5) == y_val)
print('Logistic regression best accuracy: {}'.format(acc_logistic))

# feedforward neural net
arch_list = [[100, 20],
             [100, 40],
             [200, 40],
             [200, 20],
             [200, 100, 40],
             [200, 100, 20],
             [100, 40, 20],
             [200, 100, 40, 20]]

y_hat_nn_list = []
for arch in arch_list:
    for p in [0, 0.1, 0.2]:
        print('Model architecture: {arch}; Dropout probability: {p}'.format(arch=arch, p=p))
        current = feedforward_model(train_val_data, X.shape[1] * N_COEF, arch, p)[0]
        y_hat_nn_list.append(current)

# convert data to images
X_train = (X_train - np.mean(X)) / np.std(X)
X_val = (X_val - np.mean(X)) / np.std(X)
X_train_image = np.expand_dims(X_train, axis=-1)
X_val_image = np.expand_dims(X_val, axis=-1)
train_val_data = [X_train_image, y_train, X_val_image, y_val]

# convolutional neural network
y_hat_conv_list = []
for c in [10, 20, 30, 40]:
    for d in [10, 20, 40]:
        for p in [0, 0.1, 0.2]:
            print('Number of filters: {c}; Dense layer size: {d}; '
                  'Dropout probability: {p}'.format(c=c, d=d, p=p))
            current = convolutional_model(train_val_data, N_COEF, c, d, p)[0]
            y_hat_conv_list.append(current)

preds = y_hat_nn_list + y_hat_conv_list
preds.append(y_hat_logistic)
with open('benchmark_preds.pkl', 'wb') as f:
    pickle.dump(preds, f)
