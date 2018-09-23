import numpy as np
from scipy.io import loadmat
import os
from sklearn.preprocessing import scale
from numpy.fft import fft, rfft
from sklearn.linear_model import LinearRegression


# time_series_dir = '/Users/linggeli/Downloads/time_series/cup/'
# behavioral_dir = '/Users/linggeli/Downloads/fMRIbehav/cup/'
BURN_IN = 20


def get_subject_id(time_series_dir):
    """
    Get a list of subject ids based on time series file names.

    :param time_series_dir: (string) path to time series data directory
    :return: (list) of integer subject ids
    """
    time_series_files = os.listdir(time_series_dir)
    subject_id_list = [int(x[3:7]) for x in time_series_files]
    return subject_id_list


def load_time_series(time_series_dir, subject_id):
    """
    Load time series matrix per subject.

    :param time_series_dir: (string) path to time series data directory
    :param subject_id: (int) subject id
    :return: (2d numpy array) time series data of format [channel, time]
    """
    time_series_data = loadmat(os.path.join(time_series_dir, 'SUB{}.mat'.format(subject_id)))['TC_Data']
    time_series_data = np.transpose(time_series_data)
    return time_series_data


def preprocess_time_series(time_series_data):
    """
    Center and scale time series per channel then remove linear trend after burn-in.

    :param time_series_data: (2d numpy array) time series data of format [channel, time]
    :return: preprocessed time series
    """
    time_series_data = scale(time_series_data, axis=1)
    ts_no_trend = np.zeros(time_series_data.shape)
    lm = LinearRegression()
    for i in range(time_series_data.shape[0]):
        y = time_series_data[i, BURN_IN:]
        x = (np.arange(y.shape[0]) + 1).reshape(-1, 1)
        lm = lm.fit(x, y)
        y_hat = lm.predict(x)
        res = y - y_hat
        ts_no_trend[i, BURN_IN:] = res
    return ts_no_trend


def block_indices(behavioral_data, block_num):
    """
    Find indices for a block based on behavioral data.
    :param behavioral_data: (2d numpy array) behavioral data of format [trial, variable]
    :param block_num: (int) from 1 to 4
    :return: (int) start and end indices
    """
    block_time = behavioral_data[behavioral_data[:, 12] == block_num, 11] / 2
    return int(block_time[0]), int(block_time[-1])


def divide_time_series(time_series_data, behavioral_data):
    """
    Divide entire time series into four blocks based on behavioral data.

    :param time_series_data: (2d numpy array) time series data of format [channel, time]
    :param behavioral_data: (2d numpy array) behavioral data of format [trial, variable]
    :return: (list) of time series blocks
    """
    time_series_blocks = []
    for i in range(4):
        start, end = block_indices(behavioral_data, i + 1)
        if start < BURN_IN:
            start = BURN_IN
        time_series_blocks.append(time_series_data[:, start:end])
    return time_series_blocks


def summarize_time_series(time_series, n_time=64, coef_idx=[1, 2]):
    """
    Perform fourier transformation on time series and take low frequency coefficients.
    :param time_series: (2d numpy array) time series data of format [channel, time]
    :param n_time: (int) number of time points in powers of 2
    :param coef_idx: (list) of coefficient indices
    :return: (2d numpy array) channel-wise features of format [channel, coef]
    """
    fourier_coef = rfft(time_series, n=n_time)
    return np.absolute(fourier_coef)[:, coef_idx]


def create_features(time_series_data, behavioral_data, **kwargs):
    """
    Wrapper function to create fourier features for all four blocks.

    :param time_series_data: (2d numpy array) time series data of format [channel, time]
    :param behavioral_data: (2d numpy array) behavioral data of format [trial, variable]
    :return: (2d numpy array) channel-wise features of format [block, channel, coef]
    """
    blocks = divide_time_series(time_series_data, behavioral_data)
    features = []
    for block in blocks:
        current = summarize_time_series(block, **kwargs)
        features.append(current)
    return np.stack(features)


def load_behavioral(behavioral_dir, subject_id):
    """
    Load behavioral matrix per subject.

    :param behavioral_dir: (string) path to behavioral data directory
    :param subject_id: (int) subject id
    :return: (2d numpy array) behavioral data of format [trial, variable]
    """
    behavioral_files = os.listdir(behavioral_dir)
    subject_file = [x for x in behavioral_files if int(x[12:16]) == subject_id][0]
    behavioral_data = loadmat(os.path.join(behavioral_dir, subject_file))['CP']
    return behavioral_data


def create_labels(behavioral_data):
    """
    Create labels for win/loss in each block from 1 to 4.
    :param behavioral_data: (2d numpy array) behavioral data of format [trial, variable]
    :return: (1d numpy array) labels for the four blocks
    """
    labels = []
    for i in range(4):
        labels.append(behavioral_data[behavioral_data[:, 12] == (i + 1), 1][0])
    return np.asarray(labels)


def get_all_features(time_series_dir, behavioral_dir, subject_id_list, **kwargs):
    """
    Wrapper function for time series data processing.

    :param time_series_dir: (string) path to time series data directory
    :param behavioral_dir: (string) path to behavioral data directory
    :param subject_id_list: (list) of integer subject ids
    :return: (2d numpy array) all subject block features [block, channels]
    """
    all_features = []
    for subject_id in subject_id_list:
        time_series_data = load_time_series(time_series_dir, subject_id)
        behavioral_data = load_behavioral(behavioral_dir, subject_id)
        time_series_data = preprocess_time_series(time_series_data)
        features = create_features(time_series_data, behavioral_data, **kwargs)
        all_features.append(features)
    return np.concatenate(all_features)


def get_all_labels(behavioral_dir, subject_id_list):
    """
    Wrapper function for behavioral data processing.

    :param behavioral_dir: (string) path to behavioral data directory
    :param subject_id_list: (list) of integer subject ids
    :return: (1d numpy array) labels for subject blocks
    """
    all_labels = []
    for subject_id in subject_id_list:
        behavioral_data = load_behavioral(behavioral_dir, subject_id)
        labels = create_labels(behavioral_data)
        all_labels.append(labels)
    return np.concatenate(all_labels)
