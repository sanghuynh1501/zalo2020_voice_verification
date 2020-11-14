import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from hdf5_data import HDF5DatasetGenerator

devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
except IndexError:
    print('error')

from model import Encoder

margin = 0.8
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

batch_size = 128
data_train = HDF5DatasetGenerator('train_triplet_mfcc_1M.hdf5', batch_size)
data_test = HDF5DatasetGenerator('test_triplet_mfcc_1M.hdf5', batch_size)

base_network = Encoder(1, 64, 8, 128, 60000)


def loss_object(origin, positive, negative, batch):
    p_dist = K.sum(K.square(origin - positive), axis=-1)
    n_dist = K.sum(K.square(origin - negative), axis=-1)
    return K.sum(K.maximum(p_dist - n_dist + margin, 0), axis=0) / batch


@tf.function
def train_step(_audio_origin, _audio_positive, _audio_negative):
    with tf.GradientTape() as tape:
        _audio_origin = base_network(_audio_origin, True, None)
        _audio_positive = base_network(_audio_positive, True, None)
        _audio_negative = base_network(_audio_negative, True, None)
        loss = loss_object(_audio_origin, _audio_positive, _audio_negative, batch_size)
    gradients = tape.gradient(loss, base_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, base_network.trainable_variables))
    train_loss(loss)


@tf.function
def test_step(_audio_origin, _audio_positive, _audio_negative):
    _audio_origin = base_network(_audio_origin, False, None)
    _audio_positive = base_network(_audio_positive, False, None)
    _audio_negative = base_network(_audio_negative, False, None)
    loss = loss_object(_audio_origin, _audio_positive, _audio_negative, batch_size)
    test_loss(loss)


def evaluate(image1, image2, image3):
    image1 = base_network(image1, False, None)
    image2 = base_network(image2, False, None)
    image3 = base_network(image3, False, None)
    p_dist = np.sum(np.square(image1 - image2), axis=-1)
    n_dist1 = np.sum(np.square(image1 - image3), axis=-1)
    n_dist2 = np.sum(np.square(image2 - image3), axis=-1)
    true_count = (p_dist < margin).sum()
    false_count = (n_dist1 >= margin).sum()
    false_count += (n_dist2 >= margin).sum()
    return len(p_dist), len(n_dist1) + len(n_dist2), true_count, false_count


checkpoint_path = "checkpoint"
ckpt = tf.train.Checkpoint(transformer=base_network,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

EPOCHS = 1000

min_loss = float('inf')
for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()

    with tqdm(total=data_train.get_total_samples()) as pbar:
        for origin, positive, negative, _, _ in data_train.generator():
            train_step(origin, positive, negative)
            pbar.update(batch_size)

    with tqdm(total=data_test.get_total_samples()) as pbar:
        for origin, positive, negative, _, _ in data_test.generator():
            test_step(origin, positive, negative)
            pbar.update(batch_size)

    total_true_all = 0
    total_false_all = 0
    true_count = 0
    false_count = 0
    with tqdm(total=data_test.get_total_samples()) as pbar:
        for origin, positive, negative, _, _ in data_test.generator():
            total_true, total_false, true, false = evaluate(origin, positive, negative)
            total_true_all += total_true
            total_false_all += total_false
            true_count += true
            false_count += false
            pbar.update(batch_size)

    print('Accurancy true: ', (true_count / total_true_all) * 100)
    print('Accurancy false: ', (false_count / total_false_all) * 100)

    print('Epoch {} Train Loss {:.4f} Test Loss {:.4f}'.format(
        epoch + 1, train_loss.result(), test_loss.result()))

    if test_loss.result() < min_loss:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
        min_loss = test_loss.result()
        base_network.save_weights('weights/base_model', 'tf')