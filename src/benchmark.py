import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, AveragePooling2D
from keras.callbacks import ModelCheckpoint


def logistic_model(train_val_data, verbose=False):
    """
    Logistic regression model with L1 penalty.
    """
    X_train_vec, y_train, X_val_vec, y_val = train_val_data
    model = LogisticRegressionCV(cv=10, penalty='l1', solver='liblinear')
    model = model.fit(X_train_vec, y_train)
    if verbose:
        print('10 fold cross-validation penalty C: {}'.format(model.C_[0]))
    y_hat = model.predict_proba(X_val_vec)[:, 1] > 0.5
    acc = np.mean(y_val == y_hat)
    return acc


def fully_connected_model(train_val_data, layers, drop_prob, verbose=False):
    """
    Fully connected neural network.
    """
    X_train_vec, y_train, X_val_vec, y_val = train_val_data
    input_size = X_train_vec.shape[1]
    model = Sequential()
    for i, l in enumerate(layers):
        if i == 0:
            model.add(Dense(l, input_shape=(input_size,), activation='relu'))
        else:
            model.add(Dense(l, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(1, activation='sigmoid'))
    if verbose:
        model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint('/Users/linggeli/graph_fmri/models/temp_model.h5',
                                   verbose=0, save_best_only=True, save_weights_only=True)
    hist = model.fit(X_train_vec, y_train, batch_size=100, epochs=20, verbose=0,
                     validation_data=(X_val_vec, y_val), callbacks=[checkpointer])
    model.load_weights('/Users/linggeli/graph_fmri/models/temp_model.h5')
    y_hat = model.predict(X_val_vec)[:, 0] > 0.5
    acc = np.mean(y_val == y_hat)
    return acc


def convolutional_model(train_val_data, n_filter, filter_size, dense_size, drop_prob, verbose=False):
    """
    Convolutional neural network with input images.
    """
    X_train_image, y_train, X_val_image, y_val = train_val_data
    input_image = Input(shape=(X_train_image.shape[1], X_train_image.shape[2], 1))
    x = Conv2D(n_filter, (filter_size, filter_size), padding='same', activation='relu')(input_image)
    x = AveragePooling2D((4, 4))(x)
    x = Flatten()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(drop_prob)(x)
    y = Dense(1, activation='sigmoid')(x)
    conv_model = Model(input_image, y)
    if verbose:
        conv_model.summary()
    conv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint('/Users/linggeli/graph_fmri/models/temp_model.h5',
                                   verbose=0, save_best_only=True, save_weights_only=True)
    hist = conv_model.fit(X_train_image, y_train, batch_size=100, epochs=20, verbose=0,
                          validation_data=(X_val_image, y_val), callbacks=[checkpointer])
    conv_model.load_weights('/Users/linggeli/graph_fmri/models/temp_model.h5')
    y_hat = conv_model.predict(X_val_image)[:, 0] > 0.5
    acc = np.mean(y_val == y_hat)
    return acc
