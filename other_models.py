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
    y_hat = model.predict_proba(X_val_vec)[:, 1]
    return y_hat


def feedforward_model(train_val_data, input_size, layers, drop_prob, verbose=False):
    """
    Feedforward neural network.
    """
    X_train_vec, y_train, X_val_vec, y_val = train_val_data
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
    checkpointer = ModelCheckpoint('/Users/linggeli/cnn_graph/fmri/temp_model.h5',
                                   verbose=0, save_best_only=True, save_weights_only=True)
    hist = model.fit(X_train_vec, y_train, batch_size=100, epochs=20, verbose=1,
                     validation_data=(X_val_vec, y_val), callbacks=[checkpointer])
    model.load_weights('/Users/linggeli/cnn_graph/fmri/temp_model.h5')
    y_hat = model.predict(X_val_vec)[:, 0]
    return y_hat, hist


def convolutional_model(train_val_data, n_feature, n_filter, dense_size, drop_prob, verbose=False):
    """
    Convolutional neural network with input images.
    """
    X_train_image, y_train, X_val_image, y_val = train_val_data
    input_image = Input(shape=(100, n_feature, 1))
    x = Conv2D(n_filter, (5, 5), padding='same', activation='relu')(input_image)
    x = AveragePooling2D((5, 5))(x)
    x = Flatten()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(drop_prob)(x)
    y = Dense(1, activation='sigmoid')(x)
    conv_model = Model(input_image, y)
    if verbose:
        conv_model.summary()
    conv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint('/Users/linggeli/cnn_graph/fmri/temp_model.h5',
                                   verbose=0, save_best_only=True, save_weights_only=True)
    hist = conv_model.fit(X_train_image, y_train, batch_size=100, epochs=20, verbose=1,
                          validation_data=(X_val_image, y_val), callbacks=[checkpointer])
    conv_model.load_weights('/Users/linggeli/cnn_graph/fmri/temp_model.h5')
    y_hat = conv_model.predict(X_val_image)[:, 0]
    return y_hat, hist
