import numpy as np
from tqdm import tqdm
import tensorflow as tf
import random
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from hdf5_data import HDF5DatasetGenerator

devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
except IndexError:
    print('error')

from model import Encoder

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


batch_size = 64
data_train = HDF5DatasetGenerator('train_triplet_mfcc.hdf5', batch_size, 750000)
data_test = HDF5DatasetGenerator('test_triplet_mfcc.hdf5', batch_size, 30000)

model = Encoder(2, 64, 8, 1024, 60000)


@tf.function
def train_step(audios, labels):
    audios = tf.convert_to_tensor(audios, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(audios, True, None)
        loss = tfa.losses.triplet_semihard_loss(labels, predictions, 0.5)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


@tf.function
def test_step(audios, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(audios, False, None)
    loss = tfa.losses.triplet_semihard_loss(labels, predictions, 0.5)
    test_loss(loss)


def evaluate(image1, image2, image3):
    image1 = model(image1, False, None)
    image2 = model(image2, False, None)
    image3 = model(image3, False, None)
    p_dist = np.sum(np.square(image1 - image2), axis=-1)
    n_dist1 = np.sum(np.square(image1 - image3), axis=-1)
    n_dist2 = np.sum(np.square(image2 - image3), axis=-1)
    true_count = (p_dist < 0.5).sum()
    false_count = (n_dist1 >= 0.5).sum()
    false_count += (n_dist2 >= 0.5).sum()
    return len(p_dist), len(n_dist1) + len(n_dist2), true_count, false_count


checkpoint_path = "checkpoint"
ckpt = tf.train.Checkpoint(transformer=model,
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

    factor = random.randint(1, 4)
    with tqdm(total=data_train.get_total_samples()) as pbar:
        for index, (origin, positive, negative, positive_label, negative_label) in enumerate(data_train.generator()):
            if (index % 2) == (epoch % 2):
                audios = np.concatenate([origin, negative], 0)
            else:
                audios = np.concatenate([positive, negative], 0)
            labels = np.concatenate([positive_label, negative_label], 0)
            indices = np.arange(audios.shape[0])
            np.random.shuffle(indices)
            audios = audios[indices]
            labels = labels[indices]
            # audios = audios[:batch_size]
            # labels = labels[:batch_size]
            labels = np.reshape(labels, labels.shape[0])
            train_step(audios.astype(np.float32), labels.astype(np.int32))
            pbar.update(batch_size)

    with tqdm(total=data_test.get_total_samples()) as pbar:
        for origin, positive, negative, positive_label, negative_label in data_test.generator():
            audios = np.concatenate([origin, positive, negative], 0)
            labels = np.concatenate([positive_label, positive_label, negative_label], 0)
            indices = np.arange(audios.shape[0])
            np.random.shuffle(indices)
            audios = audios[indices]
            labels = labels[indices]
            audios = audios[:batch_size]
            labels = labels[:batch_size]
            labels = np.reshape(labels, labels.shape[0])
            test_step(audios.astype(np.float32), labels.astype(np.int32))
            pbar.update(batch_size)

    total_true_all = 0
    total_false_all = 0
    true_count = 0
    false_count = 0
    with tqdm(total=data_test.get_total_samples()) as pbar:
        for origin, positive, negative, positive_label, negative_label in data_test.generator():
            total_true, total_false, true, false = evaluate(origin, positive, negative)
            total_true_all += total_true
            total_false_all += total_false
            true_count += true
            false_count += false

    print('Accurancy true: ', (true_count / total_true_all) * 100)
    print('Accurancy false: ', (false_count / total_false_all) * 100)

    print('Epoch {} Train Loss {:.4f} Test Loss {:.4f}'.format(
        epoch + 1, train_loss.result(), test_loss.result()))

    if test_loss.result() < min_loss:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
        min_loss = test_loss.result()


# total_count = 0
# true_count = 0
# with tqdm(total=data_test.get_total_samples()) as pbar:
#     for origin, positive, negative, positive_label, negative_label in data_test.generator():
#         total, true = evaluate(origin, positive, negative)
#         total_count += total
#         true_count += true