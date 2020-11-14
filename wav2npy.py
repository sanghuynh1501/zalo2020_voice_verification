import os
import random
import librosa
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from hdf5_data import HDF5DatasetWriter
from sklearn.model_selection import train_test_split


MAX_LEN_AUDIO_ORIGIN = 100000
FOLDER_NAME = "Train-Test-Data/dataset"
data_folder = os.listdir(FOLDER_NAME)

total = 0
total_filter = 0
mfcc_feature = 39

max_audio_len = 0
max_audio_link = ''
new_sample_rate = 16000


def padding_audio(audio, max_len=MAX_LEN_AUDIO_ORIGIN):
    return np.pad(audio, pad_width=(max_len - len(audio), 0), mode='constant', constant_values=(0, 0))


def read_audio(_audio_link):
    sample_rate, samples = wavfile.read(_audio_link)
    data = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
    if len(data.shape) == 2:
        data = np.mean(data, -1)
    if len(data) > 0:
        if len(data) >= MAX_LEN_AUDIO_ORIGIN:
            data = data[:MAX_LEN_AUDIO_ORIGIN]
        else:
            data = padding_audio(data)

        data = data.astype(np.float32)
        data = data / (2.0 ** (16 - 1) + 1)

        S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=mfcc_feature)
        mfcc = np.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))

        return mfcc

    return None


total = 0
data_links = []
label_links = []

LABELS = []
for container in data_folder:
    LABELS.append(container)

random.shuffle(data_folder)
total_names = []
total_samples = 1600000
image_per_folder = total_samples / 400
image_name = []
exists = []

for co_idx, container in enumerate(data_folder):
    folder_name = FOLDER_NAME + '/' + container
    folder_name = os.listdir(folder_name)
    for file_name in folder_name:
        origin = read_audio(FOLDER_NAME + '/' + container + '/' + file_name)
        if origin is not None:
            origin = np.expand_dims(origin, 0)
            np.save(FOLDER_NAME + '/' + container + '/' + file_name.split(".")[0] + '.npy', origin)

