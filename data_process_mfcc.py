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


def dum_data(audioes, labels, path):
    length = len(audioes)
    dump_data = HDF5DatasetWriter((length, 196, 39), path)

    with tqdm(total=length) as pbar:
        for _audio_link, _label in zip(audioes, labels):
            if len(_audio_link) > 0 and len(_label) > 0:
                [origin_link, positive_link, positive_label] = _audio_link.split(" ")
                [negative_link, negative_label] = _label.split(" ")

                origin = np.load(origin_link.split(".")[0] + ".npy")
                positive = np.load(positive_link.split(".")[0] + ".npy")
                negative = np.load(negative_link.split(".")[0] + ".npy")

                positive_label = np.array([[int(positive_label)]])
                negative_label = np.array([[int(negative_label)]])

                dump_data.add(origin, positive, negative, positive_label, negative_label)
            pbar.update(1)


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

f = open("data_links/data_merge.txt", "r")
data_links = f.read().split('\n')
f = open("data_links/label_merge.txt", "r")
label_links = f.read().split('\n')

audio_train, audio_test, label_train, label_test = train_test_split(data_links, label_links, test_size=0.05, random_state=42)

print(len(audio_train))
print(len(audio_test))
audio_train = audio_train[:3030000]
label_train = label_train[:3030000]
audio_test = audio_test[:150000]
label_test = label_test[:150000]

dum_data(audio_train, label_train, 'train_triplet_mfcc_1M.hdf5')
dum_data(audio_test, label_test, 'test_triplet_mfcc_1M.hdf5')

