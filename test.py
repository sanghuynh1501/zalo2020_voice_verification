import csv

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
import tensorflow as tf

devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
except IndexError:
    print('error')

from model import Encoder

PUBLIC_TEST = "public-test/"
FOLDER_NAME = "Train-Test-Data/"
MAX_LEN_AUDIO_ORIGIN = 100000

mfcc_feature = 39
new_sample_rate = 16000

base_network = Encoder(1, 64, 8, 128, 60000)
base_network.load_weights('weights/base_model')


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
        mfcc = np.expand_dims(mfcc, 0)

        return mfcc


def euclidean_distance(anchor, positive):
    p_dist = np.sum(np.square(anchor - positive))
    return p_dist


with open(FOLDER_NAME + 'public-test.csv') as csv_file:
    with open('result/result_test.csv', mode='w') as employee_file:
        result_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['audio_1', 'audio_2', 'label'])
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for idx, row in enumerate(csv_reader):
            if idx > 0:
                audio1 = FOLDER_NAME + PUBLIC_TEST + row[0]
                audio2 = FOLDER_NAME + PUBLIC_TEST + row[1]
                audio1 = read_audio(audio1)
                audio2 = read_audio(audio2)
                audio1_feature = base_network(audio1, False, None)
                audio2_feature = base_network(audio2, False, None)
                p_dist = euclidean_distance(audio1_feature, audio2_feature)
                print('p_dist ', p_dist)
                if p_dist < 0.8:
                    result_writer.writerow([row[0], row[1], '1'])
                else:
                    result_writer.writerow([row[0], row[1], '0'])