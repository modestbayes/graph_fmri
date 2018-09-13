import numpy as np
from scipy.io import loadmat
import os
from sklearn.preprocessing import scale
from numpy.fft import fft


# time_series_dir = '/Users/linggeli/Downloads/time_series/cup/'
# behavioral_dir = '/Users/linggeli/Downloads/fMRIbehav/cup/'

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


# TODO: remove trend and smooth time series
def preprocess_time_series(time_series_data):
    """
    Center and scale time series per channel.

    :param time_series_data: (2d numpy array) time series data of format [channel, time]
    :return: preprocessed time series
    """
    time_series_data = scale(time_series_data, axis=1)
    return time_series_data


# TODO: how do the time series indices actually correspond to blocks
def divide_time_series(time_series_data, start=10, length=70):
    """
    Divide entire time series into four blocks.

    :param time_series_data: (2d numpy array) time series data of format [channel, time]
    :param start: (int) starting index of first block
    :param length: (int) length of each block
    :return: (list) of time series blocks
    """
    time_series_blocks = []
    for i in range(4):
        current = start + i * length
        time_series_blocks.append(time_series_data[:, current:(current + length)])
    return time_series_blocks


# TODO: something more legit
def summarize_time_series(time_series):
    """
    Perform fourier transformation on time series and use the lowest frequency coefficient.

    :param time_series: (3d numpy array) time series data of format [block, channel, time]
    :return: (2d numpy array) channel-wise features of format [block, channel]
    """
    fourier = fft(time_series)
    return np.take(fourier.real, indices=1, axis=-1)


def create_features(time_series_data, **kwargs):
    """
    Wrapper function.

    :param time_series_data: (2d numpy array) time series data of format [channel, time]
    :param kwargs: input parameters for divide_time_series
    :return: (2d numpy array) channel-wise features of format [block, channel]
    """
    blocks = divide_time_series(time_series_data, **kwargs)
    return summarize_time_series(np.stack(blocks))


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


def get_all_features(time_series_dir, subject_id_list, **kwargs):
    """
    Wrapper function for time series data processing.

    :param time_series_dir: (string) path to time series data directory
    :param subject_id_list: (list) of integer subject ids
    :param kwargs: input parameters for divide_time_series
    :return: (2d numpy array) all subject block features [block, channels]
    """
    all_features = []
    for subject_id in subject_id_list:
        time_series_data = load_time_series(time_series_dir, subject_id)
        time_series_data = preprocess_time_series(time_series_data)
        features = create_features(time_series_data, **kwargs)
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
